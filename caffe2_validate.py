import argparse
import numpy as np
from caffe2.python import core, workspace, model_helper
from caffe2.proto import caffe2_pb2
from data import create_loader, resolve_data_config, Dataset
from utils import AverageMeter
import time

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='spnasnet1_00',
                    help='model architecture (default: dpn92)')
parser.add_argument('--c2-init', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--c2-predict', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--crop-pct', type=float, default=None, metavar='PCT',
                    help='Override default crop pct of 0.875')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--tf-preprocessing', dest='tf_preprocessing', action='store_true',
                    help='use tensorflow mnasnet preporcessing')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    model = model_helper.ModelHelper(name="validation_net", init_params=False)

    # Bring in the init net from init_net.pb
    init_net_proto = caffe2_pb2.NetDef()
    with open(args.c2_init, "rb") as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = core.Net(init_net_proto)  # model.param_init_net.AppendNet(core.Net(init_net_proto)) #

    # bring in the predict net from predict_net.pb
    predict_net_proto = caffe2_pb2.NetDef()
    with open(args.c2_predict, "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)  # model.net.AppendNet(core.Net(predict_net_proto))

    # this is so obvious, wonderful interface </sarcasm>
    input_blob = model.net.external_inputs[0]
    output_blob = model.net.external_outputs[0]

    # CUDA is crashing, no idea why, awesome error message, give it a try for kicks
    #device_opts = core.DeviceOption(caffe2_pb2.PROTO_CUDA, args.gpu_id)
    #model.net.RunAllOnGPU(gpu_id=args.gpu_id, use_cudnn=False)
    #model.param_init_net.RunAllOnGPU(gpu_id=args.gpu_id, use_cudnn=False)

    workspace.FeedBlob(input_blob, np.random.normal((1, 3, 224, 224)))
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True, input_blobs=['0'])

    data_config = resolve_data_config('mobilenetv3_100', args)
    loader = create_loader(
        Dataset(args.data, load_bytes=args.tf_preprocessing),
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        tensorflow_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(loader):
        # run the net and return prediction
        caffe2_in = input.data.numpy()
        workspace.FeedBlob(input_blob, caffe2_in)
        workspace.RunNet(model.net, num_iter=1)
        output = workspace.FetchBlob(output_blob)

        # measure accuracy and record loss
        prec1, prec5 = accuracy_np(output.data, target.numpy())
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {rate_avg:.3f}/s, {ms_avg:.3f} ms/sample) \t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, rate_avg=input.size(0) / batch_time.avg,
                ms_avg=100 * batch_time.avg / input.size(0), top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} ({top1a:.3f}) Prec@5 {top5.avg:.3f} ({top5a:.3f})'.format(
        top1=top1, top1a=100-top1.avg, top5=top5, top5a=100.-top5.avg))

    # turn it into something we can play with and examine which is in a multi-dimensional array


def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


if __name__ == '__main__':
    main()
