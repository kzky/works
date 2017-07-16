from cnn_ae_model_000 import cnn_ae_model_000
from nnabla.contrib.context import extension_context
import nnabla as nn
import nnabla.functions as F
import numpy as np

def test_forward_backward():
    batch_size, m, h, w = 4, 3, 32, 32
    extension_module = "cpu"
    device_id = 0
    ctx = extension_context(extension_module, device_id=device_id)

    x_l_data = np.random.randn(batch_size, m, h, w)
    x_l = nn.Variable(x_l_data.shape)
    x_l.d = x_l_data
    y_l = cnn_ae_model_000(ctx, x_l)
    with nn.context_scope(ctx):
        loss = F.mean(x_l - y_l)
    loss.forward()
    loss.backward()
