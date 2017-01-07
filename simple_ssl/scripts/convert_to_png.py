import cv2
import numpy as np
import os

def convert_mnist(fpath="~/datasets/mnist/test.npz"):
    data = np.load(fpath)
    dpath = os.path.dirname(fpath)
    imgs = data["x"]

    print("Writing as png")
    for i, img in enumerate(imgs):
        fpath_out = os.path.join(dpath, "test_{:05d}.png".format(i))
        print("Write to {}".format(fpath_out))
        cv2.imwrite(fpath_out, img.reshape(28, 28))
        

    
    

    

    
