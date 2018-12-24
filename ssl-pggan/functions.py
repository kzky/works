# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from nnabla import Variable
from nnabla.initializer import ConstantInitializer, NormalInitializer, UniformInitializer
from nnabla.initializer import calc_normal_std_he_forward, calc_normal_std_he_backward, calc_uniform_lim_glorot
from nnabla.parameter import get_parameter_or_create
from nnabla.parametric_functions import parametric_function_api

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


def minibatch_stddev(x, eps=1e-8):
    b, _, h, w = x.shape
    mean = F.mean(x, axis=0, keepdims=True)
    std = F.pow_scalar(F.mean(F.pow_scalar(F.sub2(x, F.broadcast(
        mean, x.shape)), 2.), axis=0, keepdims=True) + eps, 0.5)
    std_chanel = F.broadcast(F.mean(std, keepdims=True), (b, 1, h, w))
    x = F.concatenate(x, std_chanel, axis=1)
    return x


def pixel_wise_feature_vector_normalization(h, eps=1e-8):
    mean = F.mean(F.pow_scalar(h, 2), axis=1, keepdims=True)
    deno = F.pow_scalar(mean + eps, 0.5)
    return F.div2(h, F.broadcast(deno, h.shape))


def f_layer_normalization(inp, beta, gamma):
    use_axis = [x for x in range(1, inp.ndim)]
    inp = F.sub2(inp, F.mean(inp, axis=use_axis, keepdims=True))
    inp = F.div2(inp, F.pow_scalar(
        F.mean(F.pow_scalar(inp, 2), axis=use_axis, keepdims=True), 0.5))
    return inp * F.broadcast(gamma, inp.shape) + F.broadcast(beta, inp.shape)


@parametric_function_api("ln")
def layer_normalization(inp, fix_parameters=False):
    """
    """
    beta_shape = (1, inp.shape[1], 1, 1)
    gamma_shape = (1, inp.shape[1], inp.shape[2], inp.shape[3])
    beta = get_parameter_or_create(
        "beta", beta_shape, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", gamma_shape, ConstantInitializer(1), not fix_parameters)
    return f_layer_normalization(inp, beta, gamma)


def BN(h, use_bn=True, test=False):
    if use_bn:
        return PF.batch_normalization(h, batch_stat=not test)
    else:
        return h


def LN(h, use_ln):
    if use_ln:
        return layer_normalization(h)
    else:
        return h

@parametric_function_api("in")
def IN(inp, axes=[1], decay_rate=0.9, eps=1e-5, fix_parameters=False):
    """Instance Normalization
    """
    if inp.shape[0] == 1:
        return INByBatchNorm(inp, axes, decay_rate, eps, fix_parameters)

    b, c = inp.shape[0:2]
    spacial_shape = inp.shape[2:]

    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)

    # Instance normalization
    # normalize over spatial dimensions
    axis = [i for i in range(len(inp.shape)) if i > 1]
    mean = F.sum(inp, axis=axis, keepdims=True) / np.prod(axis)
    var = F.sum((inp - mean) ** 2.0, axis=axis, keepdims=True) / np.prod(axis)
                
    h = (inp - mean) / F.pow_scalar(var + eps, 0.5)
    return gamma * inp + beta


@parametric_function_api("in")
def INByBatchNorm(inp, axes=[1], decay_rate=0.9, eps=1e-5, fix_parameters=True):
    """Instance Normalization (implemented using BatchNormalization)
    Instance normalization is equivalent to the batch normalization if a batch size is one, in
    other words, it normalizes over spatial dimension(s), meaning all dimensions except for
    the batch and feature dimension.
    """
    assert len(axes) == 1

    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat=True, output_stat=False)


@parametric_function_api("adain")
def AdaIN(h, y, fix_parameters=False):
    """Adaptive Instance Normalization
    """
    # 
    b, c, _, _ = h.shape
    h = IN(h, name="in", fix_parameters=True)

    def embed_func(y, initializer):
        o = PF.affine(y, initializer=initializer, with_bias=False).reshape([b, c, 1, 1])
        return o
    with nn.parameter_scope("gamma"):
        gamma = embed_func(y, initializer=ConstantInitializer(1.))
    with nn.parameter_scope("beta"):
        beta = embed_func(y, initializer=ConstantInitializer(0.))
    h = gamma * h + beta
    return h


def normalize(h, y=None, norm="PFVN", test=False):
    if norm == "PFVN":
        return pixel_wise_feature_vector_normalization(h)
    elif norm == "BN":
        return PF.batch_normalization(h, batch_stat=not test)
    elif norm == "IN":
        return IN(h)
    elif norm == "CCBN":
        raise NotImplementedError("CCBN not implemented yet.")
    else:
        raise ValueError("`norm` in ['PFVN', 'BN', 'CCBN']")
            

def use_bias(norm):
    if norm == "PFVN":
        return True
    elif norm == "BN":
        return False
    elif norm == "CCBN":
        return False
    elif norm == "IN":
        return False
    else:
        raise ValueError("`norm` in ['PFVN', 'BN', 'CCBN']")


@parametric_function_api("conv")
def affine(inp, n_outmaps,
           base_axis=1,
           w_init=None, b_init=None,
           fix_parameters=False, rng=None, with_bias=True,
           use_wscale=True, use_he_backward=False):
    """
    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))

    # Use He backward
    if use_he_backward:
        std = calc_normal_std_he_backward(inp.shape[base_axis], n_outmap)
    else:
        std = calc_normal_std_he_forward(inp.shape[base_axis], n_outmap)

    # W init
    if w_init is None and use_wscale:
        # Equalized Learning Rate
        w_init = NormalInitializer(1.)
        w = get_parameter_or_create(
            "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, not fix_parameters)
        w *= std
    elif w_init is None and not use_wscale:
        w_init = NormalInitializer(std)
        w = get_parameter_or_create(
            "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, not fix_parameters)
    else:
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(inp.shape[base_axis], n_outmaps), rng=rng)
        w = get_parameter_or_create(
            "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, not fix_parameters)

    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)

    return F.affine(inp, w, b, base_axis)


@parametric_function_api("conv")
def conv(inp, outmaps, kernel,
         pad=None, stride=None, dilation=None, group=1,
         w_init=None, b_init=None,
         base_axis=1, fix_parameters=False, rng=None, with_bias=True,
         use_wscale=True, use_he_backward=False):
    """
    """
    # Use He backward
    if use_he_backward:
        std = calc_normal_std_he_backward(
            inp.shape[base_axis], outmaps, kernel=kernel)
    else:
        std = calc_normal_std_he_forward(
            inp.shape[base_axis], outmaps, kernel=kernel)

    # W init
    if w_init is None and use_wscale:
        # Equalized Learning Rate
        w_init = NormalInitializer(1.)
        w = get_parameter_or_create(
            "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
            w_init, not fix_parameters)
        w *= std
    elif w_init is None and not use_wscale:
        w_init = NormalInitializer(std)
        w = get_parameter_or_create(
            "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
            w_init, not fix_parameters)
    else:
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
        w = get_parameter_or_create(
            "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
            w_init, not fix_parameters)

    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)

    return F.convolution(inp, w, b, base_axis, pad, stride, dilation, group)
