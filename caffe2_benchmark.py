""" Caffe2 validation script
This script runs Caffe2 benchmark on exported model.
"""
import argparse
from caffe2.python import core, workspace, model_helper
from caffe2.proto import caffe2_pb2


parser = argparse.ArgumentParser(description='Caffe2 Model Benchmark')
parser.add_argument('--c2-init', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--c2-predict', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    model = model_helper.ModelHelper(name="le_net", init_params=False)

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

    # CUDA performance not impressive
    #device_opts = core.DeviceOption(caffe2_pb2.PROTO_CUDA, args.gpu_id)
    #model.net.RunAllOnGPU(gpu_id=args.gpu_id, use_cudnn=True)
    #model.param_init_net.RunAllOnGPU(gpu_id=args.gpu_id, use_cudnn=True)

    input_blob = model.net.external_inputs[0]
    model.param_init_net.GaussianFill(
        [],
        input_blob.GetUnscopedName(),
        shape=(args.batch_size, 3, args.img_size, args.img_size),
        mean=0.0,
        std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net, overwrite=True)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 20, True)


if __name__ == '__main__':
    main()
