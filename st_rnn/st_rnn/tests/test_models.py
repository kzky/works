from models import ElmanNet, ElmanRNN
import numpy as np

def test_onestep_elmannet():
    batch_size = 32
    dims = [784, 100, 10]
    model = ElmanNet(dims)
    x = np.random.rand(batch_size, dims[0]).astype(np.float32)
    
    states_0 = model.get_states()
    y = model(x)
    states_1 = [x.data for x in model.get_states()]
    for s_0, s_1 in zip(states_0, states_1):
        assert s_0 != s_1
    
def test_onestep_elmanrnn():
    batch_size = 32
    dims = [784, 100, 10]
    T = 3
    model = ElmanRNN(dims, T)
    x_list = list(np.random.rand(T, batch_size, dims[0]).astype(np.float32))
    y_list = model(x_list)

    for y in y_list:
        assert y.data != None
