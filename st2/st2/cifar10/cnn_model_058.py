import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.contrib.context import extension_context
import numpy as np

def ce_loss(ctx, pred, y_l):
    with nn.context_scope(ctx):
        loss_ce = F.mean(F.softmax_cross_entropy(pred, y_l))
    return loss_ce

def ce_loss_with_uncertainty(ctx, pred, y_l, log_var):
    r = F.randn(0., 1., log_var.shape)
    r = F.pow_scalar(F.exp(log_var), 0.5) * r
    h = pred + r
    with nn.context_scope(ctx):
        loss_ce = F.mean(F.softmax_cross_entropy(h, y_l))
    return loss_ce

def sr_loss(ctx, pred0, pred1):
    with nn.context_scope(ctx):
        loss_sr = F.mean(F.squared_error(pred0, pred1))
    return loss_sr

def sr_loss_with_uncertainty(ctx, pred0, pred1, log_var0, log_var1):
    #TODO: squared error/absolute error
    s0 = F.exp(log_var0)
    s1 = F.exp(log_var1)
    squared_error = F.squared_error(pred0, pred1)
    with nn.context_scope(ctx):
        loss_sr = F.mean(squared_error * (1 / s0 + 1 / s1) + (s0 / s1 + s1 / s0)) * 0.5
    return loss_sr

def er_loss(ctx, pred):
    with nn.context_scope(ctx):
        bs = pred.shape[0]
        d = np.prod(pred.shape[1:])
        denominator = bs * d
        pred_normalized = F.softmax(pred)
        pred_log_normalized = F.log(F.softmax(pred))
        loss_er = - F.sum(pred_normalized * pred_log_normalized) / denominator
    return loss_er

def sigma_regularization(ctx, log_var, one):
    with nn.context_scope(ctx):
        h = F.exp(log_var)
        h = F.pow_scalar(h, 0.5)
        r = F.mean(F.squared_error(h, one))
    return r

def sigmas_regularization(ctx, log_var0, log_var1):
    with nn.context_scope(ctx):
        h0 = F.exp(log_var0)
        h0 = F.pow_scalar(h0, 0.5)
        h1 = F.exp(log_var1)
        h1 = F.pow_scalar(h1, 0.5)
        r = F.mean(F.squared_error(h0, h1))
    return r
    
def cifar10_resnet23_prediction(ctx, image, test=False):
    """
    Construct ResNet 23
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

            return h

    # Random generator for using the same init parameters in all devices
    rng = np.random.RandomState(0)
    nmaps = 64
    ncls = 10

    # Conv -> BN -> Relu
    with nn.context_scope(ctx):
        with nn.parameter_scope("conv1"):
            h = PF.convolution(image, nmaps, kernel=(3, 3), pad=(1, 1),
                               with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)

        h = res_unit(h, "conv2", rng, False)    # -> 32x32
        h = res_unit(h, "conv3", rng, True)     # -> 16x16
        h = res_unit(h, "conv4", rng, False)    # -> 16x16
        h = res_unit(h, "conv5", rng, True)     # -> 8x8
        h = res_unit(h, "conv6", rng, False)    # -> 8x8
        h = res_unit(h, "conv7", rng, True)     # -> 4x4
        h = res_unit(h, "conv8", rng, False)    # -> 4x4
        h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
        pred = PF.affine(h, ncls)

    return pred

def cifar10_resnet23_loss(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss
