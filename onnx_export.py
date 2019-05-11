import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np

import onnx
import caffe2.python.onnx.backend as onnx_caffe2

from model_factory import create_model

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('output', metavar='ONNX_FILE',
                    help='output model filename')
parser.add_argument('--model', '-m', metavar='MODEL', default='spnasnet1_00',
                    help='model architecture (default: dpn92)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def main():
    args = parser.parse_args()

    if not args.checkpoint:
        args.pretrained = True

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    model.eval()

    x = torch.randn((1, 3, args.img_size or 224, args.img_size or 224), requires_grad=True)

    torch_out = torch.onnx._export(model, x, args.output, export_params=True)

    onnx_model = onnx.load(args.output)

    caffe2_backend = onnx_caffe2.prepare(onnx_model)

    B = {onnx_model.graph.input[0].name: x.data.numpy()}

    c2_out = caffe2_backend.run(B)[0]

    np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)


if __name__ == '__main__':
    main()
