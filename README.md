# PyTorch Generic MobileNet

A 'generic' implementation of MobileNets that covers most of the architectures derived from the MobileNet V1/V2 block sequence, including those found via automated neural architecture search. All models are implemented by the same class, with string based architecture definitions to configure the block layouts (idea from [here](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py))

## Models

Implemented models include:
  * MNASNet B1, A1 (Squeeze-Excite), and Small (https://arxiv.org/abs/1807.11626)
  * MobileNet-V1 (https://arxiv.org/abs/1704.04861)
  * MobileNet-V2 (https://arxiv.org/abs/1801.04381)
  * MobileNet-V3 (https://arxiv.org/abs/1905.02244) -- work in progress, validating config
  * ChamNet (https://arxiv.org/abs/1812.08934) -- specific arch details hard to find, currently an educated guess
  * FBNet-C (https://arxiv.org/abs/1812.03443) -- TODO A/B variants
  * Single-Path NAS (https://arxiv.org/abs/1904.02877) -- pixel1 variant
    
I originally implemented and trained some these models with code [here](https://github.com/rwightman/pytorch-image-models), this repository contains just the GenMobileNet models, validation, and associated ONNX/Caffe2 export code. 

## Pretrained

I've attempted to train some of these models, decent results below. I've found it fairly challenging to get stated (paper) accuracy for most of these models. Other implementations in PyTorch appear to mirror these difficulties.


|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling  |
|---|---|---|---|---|
| FBNet-C | 74.830 (25.170 | 92.124 (7.876) | 5.6M | bilinear |
| Single-Path NASNet 1.00 | 74.084 (25.916)  | 91.818 (8.182) | 4.42M | bilinear |

More pretrained models to come...


## Ported Weights

I ported the Tensorflow MNASNet weights to verify sanity of my model. For some reason I can't hit the stated accuracy with my port Google's tflite weights. 

Using a TF equivalent to 'SAME' padding was important to get > 70%, but something small is still missing. Note that the ported weights with the 'SAME' conv cannot be exported to ONNX. You'd be better off porting the TF model in this case anyways.

Enabling the Tensorflow preprocessing pipeline with `--tf-preprocessing` at validaiton time will improve these scores by 0.2-0.5%

|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling  | Source |
|---|---|---|---|---|---|
| MNASNet 1.00 (B1) | 72.398 (27.602) | 90.930 (9.070) |  4.36M | bicubic | [Google TFLite](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) |
| SE-MNASNet 1.00 (A1) | 73.086 (26.914) | 91.336 (8.664) | 3.87M  | bicubic | [Google TFLite](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) |


## TODO
* Train more models with better results
* Exported model validation in Caffe2
* More models (ShuffleNetV2)
