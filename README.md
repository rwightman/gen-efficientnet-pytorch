# PyTorch (Generic) Efficient Networks 

A 'generic' implementation of EfficientNet, MobileNet, etc. that covers most of the compute/parameter efficient architectures derived from the MobileNet V1/V2 block sequence, including those found via automated neural architecture search. All models are implemented by the same class, with string based architecture definitions to configure the block layouts (idea from [here](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py))

## Models

Implemented models include:
  * EfficientNet (B0-B5) (https://arxiv.org/abs/1905.11946) -- validated, compat with TF weights
  * MixNet (https://arxiv.org/abs/1907.09595) -- validated, compat with TF weights
  * MNASNet B1, A1 (Squeeze-Excite), and Small (https://arxiv.org/abs/1807.11626)
  * MobileNet-V1 (https://arxiv.org/abs/1704.04861)
  * MobileNet-V2 (https://arxiv.org/abs/1801.04381)
  * MobileNet-V3 (https://arxiv.org/abs/1905.02244) -- native PyTorch model trained better than paper spec
  * ChamNet (https://arxiv.org/abs/1812.08934) -- specific arch details hard to find, currently an educated guess
  * FBNet-C (https://arxiv.org/abs/1812.03443) -- TODO A/B variants
  * Single-Path NAS (https://arxiv.org/abs/1904.02877) -- pixel1 variant
    
I originally implemented and trained some these models with code [here](https://github.com/rwightman/pytorch-image-models), this repository contains just the GenMobileNet models, validation, and associated ONNX/Caffe2 export code. 

## Pretrained

I've managed to train several of the models to accuracies close to or above the originating papers and official impl. My training code is here: https://github.com/rwightman/pytorch-image-models


|Model | Prec@1 (Err) | Prec@5 (Err) | Param#(M) | MAdds(M) | Image Scaling | Resolution | Crop |
|---|---|---|---|---|---|---|---|
| efficientnet_b2 | 79.668 (20.332) | 94.634 (5.366) | 9.1 | 1003 | bicubic | 260x260 | 0.890 |
| efficientnet_b1 | 78.692 (21.308) | 94.086 (5.914) | 7.8 | 694 | bicubic | 240x240 | 0.882 |
| mixnet_m | 77.256 (22.744) | 93.418 (6.582) | 5.01 | 353 | bicubic | 224x224 | 0.875 |
| efficientnet_b0 | 76.912 (23.088) | 93.210 (6.790) | 5.3 | 390 | bicubic | 224x224 | 0.875 |
| mobilenetv3_100 | 75.634 (24.366) | 92.708 (7.292) | 5.5 | 219 | bicubic | 224x224 | 0.875 |
| mnasnet_a1 | 75.448 (24.552) | 92.604 (7.396) | 3.9 | 312 | bicubic | 224x224 | 0.875 |
| fbnetc_100 | 75.124 (24.876) | 92.386 (7.614) | 5.6 | 385 | bilinear | 224x224 | 0.875 |
| mnasnet_b1 | 74.658 (25.342) | 92.114 (7.886) | 4.4 | 315 | bicubic | 224x224 | 0.875 |
| spnasnet_100 | 74.084 (25.916)  | 91.818 (8.182) | 4.4 | TBV | bilinear | 224x224 | 0.875 |


More pretrained models to come...


## Ported Weights

I ported the Tensorflow MNASNet weights to verify sanity of my model. For some reason I can't hit the stated accuracy with my port Google's tflite weights. Using a TF equivalent to 'SAME' padding was important to get > 70%, but something small is still missing.

The weights ported from Tensorflow checkpoints for the EfficientNet models do pretty much match accuracy in Tensorflow once a SAME convolution padding equivalent is added, and the same crop factors, image scaling, etc are used.

Enabling the Tensorflow preprocessing pipeline with `--tf-preprocessing` at validaiton time will improve these scores by 0.1-0.5% as it's closer to what these models were trained with.

|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling  | Resolution | Crop | 
|---|---|---|---|---|---|---|
| tf_efficientnet_b5 *tfp  | 83.200 (16.800) | 96.456 (3.544) | 30.39 | bicubic | 456x456 | N/A |
| tf_efficientnet_b5       | 83.176 (16.824) | 96.536 (3.464) | 30.39 | bicubic | 456x456 | 0.934 |
| tf_efficientnet_b4       | 82.604 (17.396) | 96.128 (3.872) | 19.34 | bicubic | 380x380 | 0.922 |
| tf_efficientnet_b4 *tfp  | 82.604 (17.396) | 96.094 (3.906) | 19.34 | bicubic | 380x380 | N/A |
| tf_efficientnet_b3 *tfp  | 80.982 (19.018) | 95.332 (4.668) | 12.23 | bicubic | 300x300 | N/A |
| tf_efficientnet_b3       | 80.968 (19.032) | 95.274 (4.726) | 12.23 | bicubic | 300x300 | 0.903 |
| tf_efficientnet_b2 *tfp  | 79.782 (20.218) | 94.800 (5.200) | 9.11 | bicubic | 260x260 | N/A |
| tf_efficientnet_b2       | 79.606 (20.394) | 94.712 (5.288) | 9.11 | bicubic | 260x260 | 0.89 |
| tf_mixnet_l *tfp         | 78.846 (21.154) | 94.212 (5.788) | 7.33 | bilinear | 224x224 | N/A |
| tf_efficientnet_b1 *tfp  | 78.796 (21.204) | 94.232 (5.768) | 7.79 | bicubic | 240x240 | N/A |
| tf_mixnet_l              | 78.770 (21.230) | 94.004 (5.996) | 7.33 | bicubic | 224x224 | 0.875 |
| tf_efficientnet_b1       | 78.554 (21.446) | 94.098 (5.902) | 7.79 | bicubic | 240x240 | 0.88 |
| tf_mixnet_m *tfp         | 77.072 (22.928) | 93.368 (6.632) | 5.01 | bilinear | 224x224 | N/A |
| tf_mixnet_m              | 76.950 (23.050) | 93.156 (6.844) | 5.01 | bicubic | 224x224 | 0.875 |
| tf_efficientnet_b0 *tfp  | 76.828 (23.172) | 93.226 (6.774) | 5.29 | bicubic | 224x224 | N/A |
| tf_efficientnet_b0       | 76.528 (23.472) | 93.010 (6.990) | 5.29 | bicubic | 224x224 | 0.875 |
| tf_mixnet_s *tfp         | 75.800 (24.200) | 92.788 (7.212) | 4.13 | bilinear | 224x224 | N/A |
| tf_mixnet_s              | 75.648 (24.352) | 92.636 (7.364) | 4.13 | bicubic | 224x224 | 0.875 |


*tfp models validated with `tf-preprocessing` pipeline

Google tf and tflite weights ported from official Tensorflow repositories
* https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
* https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

## PyTorch Hub

Models can be accessed via the PyTorch Hub API

```
>>> torch.hub.list('rwightman/gen-efficientnet-pytorch')
['efficientnet_b0', ...]
>>> model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
>>> model.eval()
>>> output = model(torch.randn(1,3,224,224))
```

## Exporting

Scripts to export models to ONNX and then to Caffe2 are included, along with a Caffe2 script to verify.

As an example, to export the MobileNet-V3 pretrained model and then run an Imagenet validation:
```
python onnx_export.py --model mobilenetv3_100 ./mobilenetv3_100.onnx
python onnx_to_caffe.py ./mobilenetv3_100.onnx --c2-prefix mobilenetv3
python caffe2_validate.py /imagenet/validation/ --c2-init ./mobilenetv3.init.pb --c2-predict ./mobilenetv3.predict.pb --interpolation bicubic
```
**NOTE** the ported weights with the 'SAME' conv padding activated cannot be exported to ONNX. You'd be better off porting from the TF model -> ONNX or other deployment format in this case anyways.

