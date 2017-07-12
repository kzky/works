import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.contrib.context import extension_context
import numpy as np

def conv_unit(x, scope, maps, k=4, s=2, p=1, act=F.relu, test=False, cnt=0):
    with nn.parameter_scope(scope):
        h = PF.convolution(x, maps, kernel=(k, k), stride=(s, s), pad=(p, p))
        h = batch_normalization(h, test=not test)
        h = act(h)
        return h

def batch_normalization(h, cnt=0, test=False):
    with nn.parameter_scope("{}".format(cnt)):
        h = PF.batch_normalization(h, batch_stat=not test)
    return h

def cnn_model_003(ctx, x, act=F.relu, test=False, cnt=0):
    with nn.context_scope(ctx):
        # Convblock0
        h = conv_unit(x, "conv00", 128, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv01", 128, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv02", 128, k=3, s=1, p=1, act=act, test=test)
        h = F.max_pooling(h, (2, 2))  # 32 -> 16
        with nn.parameter_scope("bn0"):
            h = batch_normalization(h, cnt, test=not test)
        if not test:
            h = F.dropout(h)

        # Convblock 1
        h = conv_unit(h, "conv10", 256, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv11", 256, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv12", 256, k=3, s=1, p=1, act=act, test=test)
        h = F.max_pooling(h, (2, 2))  # 16 -> 8
        with nn.parameter_scope("bn1"):
            h = batch_normalization(h, cnt, test=not test)
        if not test:
            h = F.dropout(h)

        # Convblock 2
        h = conv_unit(h, "conv20", 512, k=3, s=1, p=0, act=act, test=test)  # 8 -> 6
        h = conv_unit(h, "conv21", 256, k=1, s=1, p=0, act=act, test=test)
        h = conv_unit(h, "conv22", 128, k=1, s=1, p=0, act=act, test=test)
        h = conv_unit(h, "conv23", 10, k=1, s=1, p=0, act=act, test=test)

        # Convblock 3
        h = F.average_pooling(h, (6, 6))
        with nn.parameter_scope("bn2"):
            h = batch_normalization(h, cnt, test=not test)
        h = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
        return h

def ce_loss(ctx, pred, y_l):
    with nn.context_scope(ctx):
        loss_ce = F.mean(F.softmax_cross_entropy(pred, y_l))
    return loss_ce

def sr_loss(ctx, pred0, pred1):
    with nn.context_scope(ctx):
        pred_x_u0 = F.softmax(pred0)
        pred_x_u1 = F.softmax(pred1)
        loss_sr = F.mean(F.squared_error(pred_x_u0, pred_x_u1))
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


class GradScaleContainer(object):
    def __init__(self, n):
        self.scales_supervised_loss = [None] * n
        self.scales_unsupervised_loss = [None] * n
        self.n = n

    def scale_grad(self, ctx, parameters):
        values = parameters.values()
        for i in range(self.n):
            p = values[i]
            scale = self.scales_supervised_loss[i] / self.scales_unsupervised_loss[i]
            p.g = p.g * scale
            p.grad.cast(p.g.dtype, ctx)

    def set_scales_supervised_loss(self, parameters):
        for i, p in enumerate(parameters.values()):
            self.scales_supervised_loss[i] = np.std(p.g)
    
    def set_scales_unsupervised_loss(self, parameters):
        for i, p in enumerate(parameters.values()):
            self.scales_unsupervised_loss[i] = np.std(p.g)
