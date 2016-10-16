from models import ElmanOnestep, ElmanNet
import numpy as np

def test_onestep_elmannet():
    batch_size = 32
    dims = [784, 100, 10]
    onestep = ElmanOnestep(dims)
    x = np.random.rand(batch_size, dims[0]).astype(np.float32)
    
    states_0 = onestep.get_states()
    y = onestep(x)
    states_1 = [x.data for x in onestep.get_states()]
    for s_0, s_1 in zip(states_0, states_1):
        assert s_0 != s_1
    
def test_onestep_elmanrnn():
    batch_size = 32
    dims = [784, 100, 10]
    T = 3
    onestep = ElmanOnestep(dims)
    elman_rnn = ElmanNet(onestep, T)
    x_list = list(np.random.rand(T, batch_size, dims[0]).astype(np.float32))
    y_list = elman_rnn(x_list)

    for y in y_list:
        assert y.data != None
