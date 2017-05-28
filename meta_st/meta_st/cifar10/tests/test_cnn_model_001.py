import numpy as np
from meta_st.cifar10.cnn_model_001 import Model
import chainer.functions as F
from collections import OrderedDict
from chainer import Variable

def test_forward():
    device = None
    act = F.leaky_relu
    model = Model(device, act)
    model_params = OrderedDict([x for x in model.namedparams()])

    x_data = np.random.rand(4, 3, 32, 32).astype(np.float32)
    y_data = np.random.randint(0, 10, 4).astype(np.int32)
    x = Variable(x_data)
    y = Variable(y_data)

    # forward
    y_pred = model(x, model_params, test=False)
    l = F.softmax_cross_entropy(y_pred, y)
    
    # backward
    l.backward()

    # change variable held in model_params
    for k, v in model_params.items():
        w = Variable(v.grad)
        model_params[k] = w

    # forward
    y_pred = model(x, model_params, test=False)
    l = F.softmax_cross_entropy(y_pred, y)
    
    # backward
    l.backward()

    # check
    for k, v in model_params.items():
        if v.grad is not None:
            print(k)

    
    


