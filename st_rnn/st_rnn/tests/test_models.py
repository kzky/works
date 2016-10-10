from models import ElmanRNN
import numpy as np

def test_onestep_elmanrann():
    batch_size = 32
    dims = [784, 100, 10]
    model = ElmanRNN(dims)
    x = np.random.rand(batch_size, dims[0]).astype(np.float32)
    
    states_0 = model.get_states()
    y = model(x)
    states_1 = [x.data for x in model.get_states()]

    assert np.all(states_1 != states_0)
    
