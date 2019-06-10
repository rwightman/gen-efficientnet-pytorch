# PyTorch (Generic) Efficient Networks 

A 'generic' implementation of EfficientNet, MobileNet, etc. that covers most of the compute/parameter efficient architectures derived from the MobileNet V1/V2 block sequence, including those found via automated neural architecture search. All models are implemented by the same class, with string based architecture definitions to configure the block layouts (idea from [here](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py))

## Models

Implemented models include:
  * EfficientNet (B0-B4) (https://arxiv.org/abs/1905.11946) -- validated, compat with TF weights
  * MNASNet B1, A1 (Squeeze-Excite), and Small (https://arxiv.org/abs/1807.11626)
  * MobileNet-V1 (https://arxiv.org/abs/1704.04861)
  * MobileNet-V2 (https://arxiv.org/abs/1801.04381)
  * MobileNet-V3 (https://arxiv.org/abs/1905.02244) -- native PyTorch model trained better than paper spec
  * ChamNet (https://arxiv.org/abs/1812.08934) -- specific arch details hard to find, currently an educated guess
  * FBNet-C (https://arxiv.org/abs/1812.03443) -- TODO A/B variants
  * Single-Path NAS (https://arxiv.org/abs/1904.02877) -- pixel1 variant
    
I originally implemented and trained some these models with code [here](https://github.com/rwightman/pytorch-image-models), this repository contains just the GenMobileNet models, validation, and associated ONNX/Caffe2 export code. 

## Pretrained

I've attempted to train some of these models, decent results below. I've found it challenging to get stated (paper) accuracy for most of these models. Other implementations in PyTorch appear to mirror these difficulties. My training code is here: https://github.com/rwightman/pytorch-image-models


|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling | Resolution |
|---|---|---|---|---|---|
| efficientnet_b0 | 76.912 (23.088) | 93.210 (6.790) | 5.29M | bicubic | 224x224 |
| mobilenetv3_100 | 75.634 (24.366) | 92.708 (7.292) | 5.5M | bicubic | 224x224 |
| fbnetc_100 | 75.124 (24.876) | 92.386 (7.614) | 5.6M | bilinear | 224x224 |
| spnasnet_100 | 74.084 (25.916)  | 91.818 (8.182) | 4.42M | bilinear | 224x224 |


More pretrained models to come...


## Ported Weights

I ported the Tensorflow MNASNet weights to verify sanity of my model. For some reason I can't hit the stated accuracy with my port Google's tflite weights. Using a TF equivalent to 'SAME' padding was important to get > 70%, but something small is still missing.

The weights ported from Tensorflow checkpoints for the EfficientNet models do pretty much match accuracy in Tensorflow once a SAME convolution padding equivalent is added, and the same crop factors, image scaling, etc are used.

Enabling the Tensorflow preprocessing pipeline with `--tf-preprocessing` at validaiton time will improve these scores by 0.1-0.5% as it's closer to what these models were trained with.

|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling  | Resolution | Crop | 
|---|---|---|---|---|---|---|
| tf_efficientnet_b3 *tfp  | 80.982 (19.018) | 95.332 (4.668) | 12.23 | bicubic | 300x300 | N/A |
| tf_efficientnet_b3       | 80.968 (19.032) | 95.274 (4.726) | 12.23 | bicubic | 300x300 | 0.903 |
| tf_efficientnet_b2 *tfp  | 79.782 (20.218) | 94.800 (5.200) | 9.11 | bicubic | 260x260 | N/A |
| tf_efficientnet_b2       | 79.606 (20.394) | 94.712 (5.288) | 9.11 | bicubic | 260x260 | 0.89 |
| tf_efficientnet_b1 *tfp  | 78.796 (21.204) | 94.232 (5.768) | 7.79 | bicubic | 240x240 | N/A |
| tf_efficientnet_b1       | 78.554 (21.446) | 94.098 (5.902) | 7.79 | bicubic | 240x240 | 0.88 |
| tf_efficientnet_b0 *tfp  | 76.828 (23.172) | 93.226 (6.774) | 5.29 | bicubic | 224x224 | N/A |
| tf_efficientnet_b0       | 76.528 (23.472) | 93.010 (6.990) | 5.29 | bicubic | 224x224 | 0.875 |
| tflite_semnasnet_100     | 73.086 (26.914) | 91.336 (8.664) | 3.87 | bicubic | 224x224 | 0.875 |
| tflite_mnasnet_100       | 72.398 (27.602) | 90.930 (9.070) |  4.36 | bicubic | 224x224 | 0.875 |

*tfp models validated with `tf-preprocessing` pipeline

Google tf and tflite weights ported from official Tensorflow repositories
* https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
* https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

## Exporting

Scripts to export models to ONNX and then to Caffe2 are included, along with a Caffe2 script to verify.

As an example, to export the MobileNet-V3 pretrained model and then run an Imagenet validation:
```
python onnx_export.py --model mobilenetv3_100
python onnx_to_caffe.py ./mobilenetv3_100.onnx --c2-prefix mobilenetv3
python caffe2_validate.py /imagenet/validation/ --c2-init ./mobilenetv3.init.pb --c2-predict ./mobilenetv3.predict.pb --interpolation bicubic
```
**NOTE** the ported weights with the 'SAME' conv padding activated cannot be exported to ONNX. You'd be better off porting from the TF model -> ONNX or other deployment format in this case anyways.

## TODO
* Train more models with better results
* Exported model validation in Caffe2
* More models (ShuffleNetV2)
