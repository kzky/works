import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.contrib.context import extension_context
import numpy as np

def conv_unit(x, scope, maps, k=4, s=2, p=1, act=F.relu, test=False):
    with nn.parameter_scope(scope):
        h = PF.convolution(x, maps, kernel=(k, k), stride=(s, s), pad=(p, p))
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)
        return h

def cnn_model_003(ctx, x, act=F.relu, test=False):
    with nn.context_scope(ctx):
        # Convblock0
        h = conv_unit(x, "conv00", 128, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv01", 128, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv02", 128, k=3, s=1, p=1, act=act, test=test)
        h = F.max_pooling(h, (2, 2))  # 32 -> 16
        with nn.parameter_scope("bn0"):
            h = PF.batch_normalization(h, batch_stat=not test)
        if not test:
            h = F.dropout(h)

        # Convblock 1
        h = conv_unit(h, "conv10", 256, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv11", 256, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv12", 256, k=3, s=1, p=1, act=act, test=test)
        h = F.max_pooling(h, (2, 2))  # 16 -> 8
        with nn.parameter_scope("bn1"):
            h = PF.batch_normalization(h, batch_stat=not test)
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
            h = PF.batch_normalization(h, batch_stat=not test)
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

def cifar10_resnet23_prediction(ctx, image, test=False):
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

def kl_divergence(ctx, pred, label):
    with nn.context_scope(ctx):
        elms = - F.softmax(label, axis=1) * F.log(F.softmax(pred, axis=1))
        loss = F.mean(F.sum(elms, axis=1))
    return loss
