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

def cnn_model_003(ctx, x, act=F.elu, do=True, test=False):
    with nn.context_scope(ctx):
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

def sr_loss_with_uncertainty_and_coef(ctx, pred0, pred1, log_var0, log_var1):
    c0 = srwu_coef(ctx, log_var0)
    c1 = srwu_coef(ctx, log_var1)
    sc0 = sigmas_coef(log_var0, log_var1)
    sc1 = sigmas_coef(log_var1, log_var0)

    #TODO: squared error/absolute error
    s0 = F.exp(log_var0)
    s1 = F.exp(log_var1)
    squared_error = F.squared_error(pred0, pred1)
    with nn.context_scope(ctx):
        loss_sr = F.mean(
            squared_error * (c0 / s0 + c1 / s1) + (sc0 * s0 / s1 + sc1 * s1 / s0)) * 0.5
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

def srwu_coef(ctx, log_var):
    v = F.exp(log_var)
    v0_g = F.greater_scalar(v, 1.)
    v0_l = F.logical_not(v0_g)
    c = v0_g + v * v0_l
    return c
    
def sigmas_coef(ctx, log_var0, log_var1):
    v0 = F.exp(log_var0)
    v1 = F.exp(log_var1)
    v0_g = F.greater_scalar(v0, 1.)
    v0_l = F.logical_not(v0_g)
    v1_g = F.greater_scalar(v1, 1.)
    v1_l = F.logical_not(v1_g)
    v0_g_and_v1_g = F.logical_and(v0_g, v1_g)
    v0_g_and_v1_l = F.logical_and(v0_g, v1_l)
    v0_l_and_v1_g = F.logical_and(v0_l, v1_g)
    v0_l_and_v1_l = F.logical_and(v0_l, v1_l)
    c = v0_g_and_v1_g \
        + v0_g_and_v1_l * v1 \
        + v0_l_and_v1_g / v0 \
        + v0_l_and_v1_l * v1 / v0
    return c
