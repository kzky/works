from models import ElmanOnestep, ElmanNet, RNNLabeledLosses, RNNUnlabeledLosses
import numpy as np

def test_elman_onestep():
    batch_size = 32
    dims = [784, 100, 10]
    onestep = ElmanOnestep(dims)
    x = np.random.rand(batch_size, dims[0]).astype(np.float32)
    
    states_0 = onestep.get_states()
    y = onestep(x)
    states_1 = [x.data for x in onestep.get_states()]
    for s_0, s_1 in zip(states_0, states_1):
        assert s_0 != s_1
    
def test_elman_net():
    batch_size = 32
    dims = [784, 100, 10]
    T = 3
    onestep = ElmanOnestep(dims)
    elman_net = ElmanNet(onestep, T)
    x_list = list(np.random.rand(T, batch_size, dims[0]).astype(np.float32))
    y_list = elman_net(x_list)

    for y in y_list:
        assert y.data != None

def test_rnn_labeled_losses():
    batch_size = 32
    n_cls = 10
    dims = [784, 100, 10]
    T = 3
    onestep = ElmanOnestep(dims)
    elman_net = ElmanNet(onestep, T)
    x_list = list(np.random.rand(T, batch_size, dims[0]).astype(np.float32))
    y_list = elman_net(x_list)

    y = np.random.uniform(0, 9, batch_size).astype(np.int32)
    rnn_losses = RNNLabeledLosses(T)
    losses = rnn_losses(y_list, y)
    
    for loss in losses:
        assert loss.data != None
    
def test_rnn_unlabeled_losses():
    batch_size = 32
    n_cls = 10
    dims = [784, 100, 10]
    T = 3
    onestep = ElmanOnestep(dims)
    elman_net = ElmanNet(onestep, T)
    x_list = list(np.random.rand(T, batch_size, dims[0]).astype(np.float32))
    y_list = elman_net(x_list)

    y = np.random.uniform(0, 9, (batch_size, n_cls))
    rnn_losses = RNNUnlabeledLosses(T)
    losses = rnn_losses(y_list)
    
    for loss in losses:
        assert loss.data != None
    


    
