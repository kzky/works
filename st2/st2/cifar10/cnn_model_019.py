import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.contrib.context import extension_context
import numpy as np

# Residual Unit
def res_unit(x, scope_name, act=F.relu, dn=False, test=False):
    C = x.shape[1]

    with nn.parameter_scope(scope_name):
        # Conv -> BN -> Relu
        with nn.parameter_scope("conv1"):
            h = PF.convolution(x, C/2, kernel=(1, 1), pad=(0, 0), with_bias=False)
            h = PF.batch_normalization(h, decay_rate=0.9, batch_stat=not test)
            h = act(h)
        # Conv -> BN -> Relu
        with nn.parameter_scope("conv2"):
            h = PF.convolution(h, C/2, kernel=(3, 3), pad=(1, 1), with_bias=False)
            h = PF.batch_normalization(h, decay_rate=0.9, batch_stat=not test)
            h = act(h)
        # Conv -> BN
        with nn.parameter_scope("conv3"): 
            h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0), with_bias=False)
            h = PF.batch_normalization(h, decay_rate=0.9, batch_stat=not test)
    # Residual -> Relu
    h = F.add2(h, x)
    h = act(h)
    
    # Maxpooling
    if dn:
        h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
    
    return h

def resnet_model(ctx, x, inmaps=64, act=F.relu, test=False):
    # Conv -> BN -> Relu
    with nn.context_scope(ctx):
        with nn.parameter_scope("conv1"):
            h = PF.convolution(x, inmaps, kernel=(3, 3), pad=(1, 1), with_bias=False)
            h = PF.batch_normalization(h, decay_rate=0.9, batch_stat=not test)
            h = act(h)
        
        h = res_unit(h, "conv2", act, False) # -> 32x32
        h = res_unit(h, "conv3", act, True)  # -> 16x16
        with nn.parameter_scope("bn0"):
            h = PF.batch_normalization(h, batch_stat=not test)
        if not test:
            h = F.dropout(h)
        h = res_unit(h, "conv4", act, False) # -> 16x16
        h = res_unit(h, "conv5", act, True)  # -> 8x8
        with nn.parameter_scope("bn1"):
            h = PF.batch_normalization(h, batch_stat=not test)
        if not test:
            h = F.dropout(h)
        h = res_unit(h, "conv6", act, False) # -> 8x8
        h = res_unit(h, "conv7", act, True)  # -> 4x4
        with nn.parameter_scope("bn2"):
            h = PF.batch_normalization(h, batch_stat=not test)
        if not test:
            h = F.dropout(h)
        h = res_unit(h, "conv8", act, False) # -> 4x4
        h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
        
        pred = PF.affine(h, 10)
    return pred

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
