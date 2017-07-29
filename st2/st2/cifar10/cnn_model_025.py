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

def attention(k, q, v, div_dim=True, softmax=True):
    v_shape = v.shape
    k = F.identity(k)
    q = F.identity(q)
    k = F.reshape(k, (k.shape[0], np.prod(k.shape[1:])))
    q = F.reshape(q, (q.shape[0], np.prod(q.shape[1:])))
    v = q  # F.reshape is inplace
    cf = F.affine(q, F.transpose(k, (1, 0)))
    if div_dim:
        dim = np.prod(v_shape[1:])
        cf /= np.sqrt(dim)
    h = cf
    if softmax: 
        h = F.softmax(h)
    h = F.affine(h, v)
    h = F.reshape(h, v_shape)
    return h

def cnn_model_003_with_cross_attention(ctx, x_list, act=F.relu, test=False):
    """With attention before pooling
    """
    with nn.context_scope(ctx):
        # Convblock0
        h0_list = []
        for x in x_list:
            h = conv_unit(x, "conv00", 128, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv01", 128, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv02", 128, k=3, s=1, p=1, act=act, test=test)
            h0_list.append(h)

        # Corss attention
        ca0 = attention(h0_list[0], h0_list[1], h0_list[1], 
                        div_dim=True, softmax=True)
        ca1 = attention(h0_list[1], h0_list[0], h0_list[0], 
                        div_dim=True, softmax=True)

        # Maxpooing, Batchnorm, Dropout
        h0_list = []
        for h in [ca0, ca1]:
            h = F.max_pooling(h, (2, 2))  # 32 -> 16
            with nn.parameter_scope("bn0"):
                h = PF.batch_normalization(h, batch_stat=not test)
            if not test:
                h = F.dropout(h)
            h0_list.append(h)

        # Convblock 1
        h1_list = []
        for h in h0_list:
            h = conv_unit(h, "conv10", 256, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv11", 256, k=3, s=1, p=1, act=act, test=test)
            h = conv_unit(h, "conv12", 256, k=3, s=1, p=1, act=act, test=test)
            h1_list.append(h)

        # Corss attention
        ca0 = attention(h1_list[0], h1_list[1], h1_list[1], 
                        div_dim=True, softmax=True)
        ca1 = attention(h1_list[1], h1_list[0], h1_list[0], 
                        div_dim=True, softmax=True)
            
        # Maxpooing, Batchnorm, Dropout
        h1_list = []
        for h in [ca0, ca1]:
            h = F.max_pooling(h, (2, 2))  # 16 -> 8
            with nn.parameter_scope("bn1"):
                h = PF.batch_normalization(h, batch_stat=not test)
            if not test:
                h = F.dropout(h)
                h1_list.append(h)

        # Convblock 2
        h2_list = []
        for h in h1_list:
            h = conv_unit(h, "conv20", 512, k=3, s=1, p=0, act=act, test=test)  # 8 -> 6
            h = conv_unit(h, "conv21", 256, k=1, s=1, p=0, act=act, test=test)
            h = conv_unit(h, "conv22", 128, k=1, s=1, p=0, act=act, test=test)
            h = conv_unit(h, "conv23", 10, k=1, s=1, p=0, act=act, test=test)
            h2_list.append(h)

        # Corss attention
        ca0 = attention(h2_list[0], h2_list[1], h2_list[1], 
                        div_dim=True, softmax=True)
        ca1 = attention(h2_list[1], h2_list[0], h2_list[0], 
                        div_dim=True, softmax=True)

        # Convblock 3
        h3_list = []
        for h in [ca0, ca1]:
            h = F.average_pooling(h, (6, 6))
            with nn.parameter_scope("bn2"):
                h = PF.batch_normalization(h, batch_stat=not test)
            h = F.reshape(h, (h.shape[0], np.prod(h.shape[1:])))
            h3_list.append(h)
        return h3_list

def one_by_one_conv(h, scope, k=1, s=1, p=1):
    with nn.parameter_scope(scope):
        maps = h.shape[1]
        h = PF.convolution(h, maps, kernel=(k, k), stride=(s, s), pad=(1, 1))
    return h

def cnn_model_003(ctx, x, act=F.relu, test=False):
    with nn.context_scope(ctx):
        # Convblock0
        h = conv_unit(x, "conv00", 128, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv01", 128, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv02", 128, k=3, s=1, p=1, act=act, test=test)

        # Learned attention multiplication
        h = one_by_one_conv(h, "attend0")
        h = F.max_pooling(h, (2, 2))  # 32 -> 16
        with nn.parameter_scope("bn0"):
            h = PF.batch_normalization(h, batch_stat=not test)
        if not test:
            h = F.dropout(h)

        # Convblock 1
        h = conv_unit(h, "conv10", 256, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv11", 256, k=3, s=1, p=1, act=act, test=test)
        h = conv_unit(h, "conv12", 256, k=3, s=1, p=1, act=act, test=test)

        # Learned attention multiplication
        h = one_by_one_conv(h, "attend1")
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

        # Learned attention multiplication
        h = one_by_one_conv(h, "attend2")

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
