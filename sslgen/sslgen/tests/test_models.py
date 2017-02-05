from sslgen.cnn_model import Encoder, Decoder
import numpy as np                    

def test_cnn():
    # Encoder
    encoder = Encoder()
    x = np.random.rand(8, 1, 28, 28).astype(np.float32)
    encoder(x)
        
    # Decoder
    decoder = Decoder()
    y = np.random.rand(8, 10).astype(np.float32)
    decoder(y)
    
