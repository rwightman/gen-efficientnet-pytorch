import torch.nn as nn
import torch.nn.functional as F

import math

from .helpers import load_state_dict_from_url
from .layers import Flatten
from .efficientnet_builder import *

__all__ = ['mobilenetv3_050', 'mobilenetv3_075', 'mobilenetv3_100']

model_urls = {
    'mobilenetv3_050': None,
    'mobilenetv3_075': None,
    'mobilenetv3_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
}


class MobileNetV3(nn.Module):
    """ MobileNetV3
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280,
                 channel_multiplier=1.0, pad_type='', act_layer=HardSwish, drop_rate=0., drop_connect_rate=0.,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=BN_ARGS_PT, weight_init='goog'):
        super(MobileNetV3, self).__init__()
        self.drop_rate = drop_rate

        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        in_chs = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier, pad_type=pad_type, act_layer=act_layer, se_gate_fn=hard_sigmoid,
            se_reduce_mid=True, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
            drop_connect_rate=drop_connect_rate)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = select_conv2d(in_chs, num_features, 1, padding=pad_type)
        self.act2 = act_layer(inplace=True)
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if weight_init == 'goog':
                initialize_weight_goog(m)
            else:
                initialize_weight_default(m)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([
            self.global_pool, self.conv_head,self.act2,
            Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _create_model(model_kwargs, variant, pretrained=False):
    as_sequential = model_kwargs.pop('as_sequential', False)
    model = MobileNetV3(**model_kwargs)
    if pretrained and model_urls[variant]:
        model.load_state_dict(load_state_dict_from_url(model_urls[variant]))
    if as_sequential:
        model = model.as_sequential()
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = _create_model(model_kwargs, variant, pretrained)
    return model


def mobilenetv3_050(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_050', 0.5, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_075', 0.75, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3('mobilenetv3_100', 1.0, pretrained=pretrained, **kwargs)
    return model
