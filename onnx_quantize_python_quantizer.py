""" ONNX-runtime validation script

This script was created to verify accuracy and performance of exported ONNX
models running with the onnxruntime. It utilizes the PyTorch dataloader/processing
pipeline for a fair comparison against the originals.

Copyright 2020 Ross Wightman
"""
import argparse

import onnx
from furiosa.quantizer.frontend.onnx import (
    calibrate,
    optimize_model,
    post_training_quantize,
    quantizer,
)

from data import Dataset, create_loader, resolve_data_config

parser = argparse.ArgumentParser(description="Caffe2 ImageNet Validation")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--onnx-input",
    default="",
    type=str,
    metavar="PATH",
    help="path to onnx model/weights file",
)
parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--crop-pct",
    type=float,
    default=None,
    metavar="PCT",
    help="Override default crop pct of 0.875",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "--tf-preprocessing",
    dest="tf_preprocessing",
    action="store_true",
    help="use tensorflow mnasnet preporcessing",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    data_config = resolve_data_config(None, args)
    loader = create_loader(
        Dataset(args.data, load_bytes=args.tf_preprocessing),
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=data_config["crop_pct"],
        tensorflow_preprocessing=args.tf_preprocessing,
    )

    model = onnx.load_model(args.onnx_input)
    model = optimize_model(model)
    input_name = model.graph.input[0].name
    ranges = calibrate.calibrate(
        model, dataset=({input_name: image.numpy()} for image, _ in loader)
    )

    quant_model = quantizer.FuriosaONNXQuantizer(
        model=model,
        per_channel=True,
        static=True,
        mode=quantizer.QuantizationMode.FAKE,
        dynamic_ranges=ranges,
        raw_data=True,
    ).quantize()

    onnx.save_model(quant_model, f"{args.onnx_input.split('.onnx')[0]}_fake_quant.onnx")

    quant_model = post_training_quantize(
        model, ({input_name: image.numpy()} for image, _ in loader)
    )
    onnx.save_model(quant_model, f"{args.onnx_input.split('.onnx')[0]}_quant.onnx")


if __name__ == "__main__":
    main()
