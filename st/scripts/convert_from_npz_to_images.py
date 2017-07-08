import cv2
import numpy as np
import os

def main():
    fpath_inp = "/home/kzk/tmp/sslgen2/l_train.npz"
    dpath_out = "/home/kzk/tmp/sslgen2/mnist_l/"

    data = np.load(fpath_inp)
    imgs = data["x"]

    os.makedirs(dpath_out)
    for i, img in enumerate(imgs):
        fpath_out = os.path.join(dpath_out, "l_sample_{:05d}.png".format(i))
        cv2.imwrite(fpath_out, img.reshape(28, 28))
            
    

if __name__ == '__main__':
    main()
