from st2.cifar10.cnn_model_003 import cnn_model_003
from st2.cifar10.cnn_model_024 import resnet_model
from nnabla.contrib.context import extension_context
import nnabla as nn
import nnabla.functions as F
import numpy as np

def main():
    batch_size, m, h, w = 4, 3, 32, 32
    extension_module = "cpu"
    device_id = 0
    ctx = extension_context(extension_module, device_id=device_id)

    x_l_data = np.random.randn(batch_size, m, h, w)
    y_l_data = (np.random.rand(batch_size, 1) * 10).astype(np.int32)
    x_l = nn.Variable(x_l_data.shape)
    y_l = nn.Variable(y_l_data.shape)
    x_l.d = x_l_data
    y_l.d = y_l_data

    # CNN
    print("# CNN")
    pred = cnn_model_003(ctx, x_l)
    s = 0
    for n, v in nn.get_parameters().iteritems():
        n_params = np.prod(v.shape)
        print(n, n_params)
        s += n_params
    print("n_params={}".format(s))
    nn.clear_parameters()
    
    # Resnet
    print("# Resnet")
    inmaps = 256
    pred = resnet_model(ctx, x_l, inmaps=inmaps)
    s = 0
    for n, v in nn.get_parameters().iteritems():
        n_params = np.prod(v.shape)
        print(n, n_params)
        s += n_params
    print("n_params={}".format(s))
    nn.clear_parameters()

if __name__ == '__main__':
    main()
