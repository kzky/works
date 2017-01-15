import cv2
import numpy as np
import os
import shutil

def convert_mnist_test(fpath="~/datasets/mnist/test.npz"):
    data = np.load(fpath)
    dpath = "{}/test".format(os.path.dirname(fpath))
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
        os.makedirs(dpath)
    else:
        os.makedirs(dpath)
    
    imgs = data["x"]

    print("Writing as png")
    for i, img in enumerate(imgs):
        fpath_out = os.path.join(dpath, "test_{:05d}.png".format(i))
        print("Write to {}".format(fpath_out))
        cv2.imwrite(fpath_out, img.reshape(28, 28))

def main():
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/mnist/test.npz")
    convert_mnist_test(fpath)

if __name__ == '__main__':
    main()
    
        

    
    

    

    
