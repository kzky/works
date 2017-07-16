import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.contrib.context import extension_context
import numpy as np

def conv_unit(x, scope, maps, k=4, s=2, p=1, act=F.relu, test=False):
    with nn.parameter_scope(scope):
        h = PF.convolution(x, maps, kernel=(k, k), stride=(s, s), pad=(p, p))
        if act is None:
            return h
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)
        return h

def deconv_unit(x, scope, maps, k=4, s=2, p=1, act=F.relu, test=False):
    with nn.parameter_scope(scope):
        h = PF.deconvolution(x, maps, kernel=(k, k), stride=(s, s), pad=(p, p))
        if act is None:
            return h
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)
        return h

def cnn_ae_model_000(ctx, x, act=F.relu, test=False):
    with nn.parameter_scope("ae"):
        with nn.context_scope(ctx):
            # Convblock0
            h = conv_unit(x, "conv00", 32, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv01", 32, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv02", 32, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv03", 32, k=4, s=2, p=1, act=act, test=test)  # 32 -> 16
            if not test:
                h = F.dropout(h)
     
            # Convblock 1
            h = conv_unit(h, "conv10", 64, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv11", 64, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv12", 64, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv13", 64, k=4, s=2, p=1, act=act, test=test) # 16 -> 8
            if not test:
                h = F.dropout(h)
     
            # Deconvblock0
            h = deconv_unit(h, "deconv00", 64, k=4, s=2, p=1, act=act, test=test) # 8 -> 16
            h = deconv_unit(h, "deconv01", 64, k=3, s=1, p=1, act=act, test=test)
     
            h = deconv_unit(h, "deconv02", 64, k=3, s=1, p=1, act=act, test=test)
            h = deconv_unit(h, "deconv03", 64, k=3, s=1, p=1, act=act, test=test)  
     
            # Deconvblock 1
            h = deconv_unit(h, "deconv10", 32, k=4, s=2, p=1, act=act, test=test)  # 16 -> 32
            h = deconv_unit(h, "deconv11", 32, k=3, s=1, p=1, act=act, test=test)
            h = deconv_unit(h, "deconv12", 32, k=3, s=1, p=1, act=act, test=test)
            h = deconv_unit(h, "deconv13", 3, k=3, s=1, p=1, act=None, test=test)

        return h

def recon_loss(ctx, pred, x_l):
    with nn.context_scope(ctx):
        loss_recon = F.mean(F.squared_error(pred, x_l))
    return loss_recon
