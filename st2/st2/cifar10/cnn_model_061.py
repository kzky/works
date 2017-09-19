import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.contrib.context import extension_context
import numpy as np

def conv_unit(x, scope, maps, k=4, s=2, p=1, act=F.elu, test=False):
    with nn.parameter_scope(scope):
        h = PF.convolution(x, maps, kernel=(k, k), stride=(s, s), pad=(p, p))
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)
        return h

def cnn_model_003(ctx, scope, x, act=F.elu, do=True, test=False):
    with nn.context_scope(ctx):
        with nn.parameter_scope(scope):
            # Convblock0
            h = conv_unit(x, "conv00", 128, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv01", 128, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv02", 128, k=3, s=1, p=1, act=act, test=test)
            h = F.max_pooling(h, (2, 2))  # 32 -> 16
            with nn.parameter_scope("bn0"):
                h = PF.batch_normalization(h, batch_stat=not test)
            if not test and do:
                h = F.dropout(h)
     
            # Convblock 1
            h = conv_unit(h, "conv10", 256, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv11", 256, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv12", 256, k=3, s=1, p=1, act=act, test=test)
            h = F.max_pooling(h, (2, 2))  # 16 -> 8
            with nn.parameter_scope("bn1"):
                h = PF.batch_normalization(h, batch_stat=not test)
            if not test and do:
                h = F.dropout(h)
     
            # Convblock 2
            h = conv_unit(h, "conv20", 512, k=3, s=1, p=0, act=act, test=test)  # 8 -> 6
            h = conv_unit(h, "conv21", 256, k=1, s=1, p=0, act=act, test=test)
            h = conv_unit(h, "conv22", 128, k=1, s=1, p=0, act=act, test=test)
            u = h
     
            # Convblock 3
            h = conv_unit(h, "conv23", 10, k=1, s=1, p=0, act=act, test=test)
            h = F.average_pooling(h, (6, 6))
            with nn.parameter_scope("bn2"):
                h = PF.batch_normalization(h, batch_stat=not test)
            pred = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
     
            # Uncertainty
            u = conv_unit(u, "u0", 10, k=1, s=1, p=0, act=act, test=test)
            u = F.average_pooling(u, (6, 6))
            with nn.parameter_scope("u0bn"):
                u = PF.batch_normalization(u, batch_stat=not test)
                log_var = F.reshape(u, (u.shape[0], np.prod(u.shape[1:])))

        return pred, log_var

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

def cifar10_resnet23_prediction(ctx, scope, image, test=False):
    """
    Construct ResNet 23
    """
    # Residual Unit
    def res_unit(x, scope_name, dn=False, test=False):
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
    nmaps = 64
    ncls = 10

    # Conv -> BN -> Relu
    with nn.context_scope(ctx):
        with nn.parameter_scope(scope):
            with nn.parameter_scope("conv1"):
                h = PF.convolution(image, nmaps, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
             
            h = res_unit(h, "conv2", False)    # -> 32x32
            h = res_unit(h, "conv3", True)     # -> 16x16
            h = res_unit(h, "conv4", False)    # -> 16x16
            h = res_unit(h, "conv5", True)     # -> 8x8
            h = res_unit(h, "conv6", False)    # -> 8x8
            h = res_unit(h, "conv7", True)     # -> 4x4
            h = res_unit(h, "conv8", False)    # -> 4x4
            h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
            pred = PF.affine(h, ncls)

    return pred

def softmax_with_temperature(ctx, x, t):
    with nn.context_scope(ctx):
        h = x / t
        h = F.softmax(h, axis=1)
    return h

def kl_divergence(ctx, pred, label, log_var):
    with nn.context_scope(ctx):
        s = F.pow_scalar(F.exp(log_var), 0.5)
        elms = softmax_with_temperature(ctx, label, s) \
               * F.log(F.softmax(pred, axis=1))
        loss = -F.mean(F.sum(elms, axis=1))
    return loss

    
