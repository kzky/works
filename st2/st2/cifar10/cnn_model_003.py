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


