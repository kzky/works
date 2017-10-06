import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.contrib.context import extension_context
import numpy as np

def conv_unit(x, scope, maps, k=4, s=2, p=1, act=PF.prelu, test=False):
    with nn.parameter_scope(scope):
        h = PF.convolution(x, maps, kernel=(k, k), stride=(s, s), pad=(p, p))
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)
        return h

def cnn_model_003(ctx, x, act=PF.prelu, do=True, test=False):
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
    


