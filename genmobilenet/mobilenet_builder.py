import re
from copy import deepcopy

import torch
from torch import nn as nn
from genmobilenet.conv2d_same import *

# Default args for PyTorch BN impl
BN_MOMENTUM_DEFAULT = 0.1
BN_EPS_DEFAULT = 1e-5


def round_channels(channels, depth_multiplier=1.0, depth_divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""
    if not depth_multiplier:
        return channels

    channels *= depth_multiplier
    min_depth = min_depth or depth_divisor
    new_channels = max(
        int(channels + depth_divisor / 2) // depth_divisor * depth_divisor,
        min_depth)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += depth_divisor
    return new_channels


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_fn=F.relu, noskip=False, pw_act=False,
                 bn_momentum=BN_MOMENTUM_DEFAULT, bn_eps=BN_EPS_DEFAULT,
                 folded_bn=False, padding_same=False):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.act_fn = act_fn
        dw_padding = padding_arg(kernel_size // 2, padding_same)
        pw_padding = padding_arg(0, padding_same)

        self.conv_dw = sconv2d(
            in_chs, in_chs, kernel_size,
            stride=stride, padding=dw_padding, groups=in_chs, bias=folded_bn)
        self.bn1 = None if folded_bn else nn.BatchNorm2d(in_chs, momentum=bn_momentum, eps=bn_eps)
        self.conv_pw = sconv2d(in_chs, out_chs, 1, padding=pw_padding, bias=folded_bn)
        self.bn2 = None if folded_bn else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)

        x = self.conv_pw(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x)

        if self.has_residual:
            x += residual
        return x


class CascadeConv3x3(nn.Sequential):
    # FIXME lifted from maskrcnn_benchmark blocks, haven't used yet
    def __init__(self, in_chs, out_chs, stride, act_fn=F.relu, noskip=False,
                 bn_momentum=BN_MOMENTUM_DEFAULT, bn_eps=BN_EPS_DEFAULT,
                 folded_bn=False, padding_same=False):
        super(CascadeConv3x3, self).__init__()
        assert stride in [1, 2]
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.act_fn = act_fn
        padding = padding_arg(1, padding_same)

        self.conv1 = sconv2d(in_chs, in_chs, 3, stride=stride, padding=padding, bias=folded_bn)
        self.bn1 = None if folded_bn else nn.BatchNorm2d(in_chs, momentum=bn_momentum, eps=bn_eps)
        self.conv2 = sconv2d(in_chs, out_chs, 3, stride=1, padding=padding, bias=folded_bn)
        self.bn2 = None if folded_bn else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        if self.has_residual:
            x += residual
        return x


class ChannelShuffle(nn.Module):
    # FIXME lifted from maskrcnn_benchmark blocks, haven't used yet
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x = torch.sigmoid(x_se) * x
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_fn=F.relu, exp_ratio=1.0, noskip=False,
                 se_ratio=0., shuffle_type=None, pw_group=1,
                 bn_momentum=BN_MOMENTUM_DEFAULT, bn_eps=BN_EPS_DEFAULT,
                 folded_bn=False, padding_same=False):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        dw_padding = padding_arg(kernel_size // 2, padding_same)
        pw_padding = padding_arg(0, padding_same)

        # Point-wise expansion
        self.conv_pw = sconv2d(in_chs, mid_chs, 1, padding=pw_padding, groups=pw_group, bias=folded_bn)
        self.bn1 = None if folded_bn else nn.BatchNorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        # Depth-wise convolution
        self.conv_dw = sconv2d(
            mid_chs, mid_chs, kernel_size, padding=dw_padding, stride=stride, groups=mid_chs, bias=folded_bn)
        self.bn2 = None if folded_bn else nn.BatchNorm2d(mid_chs, momentum=bn_momentum, eps=bn_eps)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(mid_chs, reduce_chs=max(1, int(in_chs * se_ratio)))

        # Point-wise linear projection
        self.conv_pwl = sconv2d(mid_chs, out_chs, 1, padding=pw_padding, groups=pw_group, bias=folded_bn)
        self.bn3 = None if folded_bn else nn.BatchNorm2d(out_chs, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act_fn(x)

        # FIXME haven't tried this yet
        # for channel shuffle when using groups with pointwise convs as per FBNet variants
        if self.shuffle_type == "mid":
            x = self.shuffle(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.act_fn(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        if self.bn3 is not None:
            x = self.bn3(x)

        if self.has_residual:
            x += residual

        # NOTE maskrcnn_benchmark building blocks have an SE module defined here for some variants

        return x


class MobilenetBuilder:
    """ Build Trunk Blocks for Mobile Networks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """

    def __init__(self, depth_multiplier=1.0, depth_divisor=8, min_depth=None,
                 bn_momentum=BN_MOMENTUM_DEFAULT, bn_eps=BN_EPS_DEFAULT,
                 folded_bn=False, padding_same=False, verbose=False):
        self.depth_multiplier = depth_multiplier
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.folded_bn = folded_bn
        self.padding_same = padding_same
        self.verbose = verbose
        self.in_chs = None

    def _round_channels(self, chs):
        return round_channels(chs, self.depth_multiplier, self.depth_divisor, self.min_depth)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = round_channels(ba['out_chs'])
        ba['bn_momentum'] = self.bn_momentum
        ba['bn_eps'] = self.bn_eps
        ba['folded_bn'] = self.folded_bn
        ba['padding_same'] = self.padding_same
        #  NOTE: could replace this with lambdas or functools binding if variety increases
        if bt == 'ir':
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'ca':
            block = CascadeConv3x3(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block
        return block

    def _make_stack(self, stack_args):
        blocks = []
        # each stack (stage) contains a list of block arguments
        for block_idx, ba in enumerate(stack_args):
            if self.verbose:
                print('block', block_idx, end=', ')
            if block_idx >= 1:
                # only the first block in any stack/stage can have a stride > 1
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, arch_def):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            arch_def: A list of lists, outer list defines stacks (or stages), inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        arch_args = _decode_arch_def(arch_def)  # convert and expand string defs to arg dicts
        if self.verbose:
            print('Building model trunk with %d stacks (stages)...' % len(arch_args))
        self.in_chs = in_chs
        blocks = []
        # outer list of arch_args defines the stacks ('stages' by some conventions)
        for stack_idx, stack in enumerate(arch_args):
            if self.verbose:
                print('stack', stack_idx)
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return blocks


def _decode_block_str(block_str):
    """ Decode block definition string

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act,
      ca = Cascade3x3, and possibly more)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    # FIXME validate args and throw

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            exp_ratio=int(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            noskip=('noskip' in block_str),
        )
        if 'g' in options:
            block_args['pw_group'] = options['g']
            if options['g'] > 1:
                block_args['shuffle_type'] = 'mid'
    elif block_type == 'ca':
        block_args = dict(
            block_type=block_type,
            out_chs=int(options['c']),
            stride=int(options['s']),
            noskip=('noskip' in block_str),
        )
    elif block_type == 'ds' or block_type == 'dsa':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            noskip=block_type == 'dsa' or 'noskip' in block_str,
            pw_act=block_type == 'dsa',
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    # return a list of block args expanded by num_repeat
    return [deepcopy(block_args) for _ in range(num_repeat)]


def _decode_arch_args(string_list):
    block_args = []
    for block_str in string_list:
        block_args.append(_decode_block_str(block_str))
    return block_args


def _decode_arch_def(arch_def):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            stack_args.extend(_decode_block_str(block_str))
        arch_args.append(stack_args)
    return arch_args
