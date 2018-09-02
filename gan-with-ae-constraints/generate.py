import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C
from nnabla.utils.data_iterator import data_iterator
from nnabla.monitor import Monitor, MonitorSeries, MonitorImage, MonitorImageTile
from nnabla.ext_utils import get_extension_context
import nnabla.utils.save as save

from datasets import data_iterator_celebA
from args import get_args, save_args
from models import encoder, decoder, infer, loss_recon, loss_kl
from helpers import normalize_method


def evaluate(args):
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    nn.load_parameters(args.model_load_path)
    x = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    e = encoder(x, args.maps, test=True)
    z, mu, logvar, var = infer(e, sigma=args.sigma)
    z = z + nn.Variable.from_numpy_array(np.random.choice([-1, 1], z.shape))
    x_recon = decoder(z, args.maps * 32, test=True).apply(persistent=True)

    # Data iterator
    di = data_iterator_celebA(args.valid_data_path, args.batch_size)

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_image_tile = MonitorImageTile("Generate Image", monitor, interval=1, 
                                          num_images=args.batch_size * args.batch_size)
    
    # Generate
    images = []
    x_data = di.next()[0]
    x.d = x_data
    for _ in range(args.batch_size):
        x_recon.forward(clear_buffer=True)
        images.append(x_recon.d.copy())
    saved_images = []
    for b0 in range(args.batch_size):
        vimages = []
        vimages.append(x_data[b0, ...])
        for b1 in range(args.batch_size - 1):
            vimages.append(images[b1][b0, ...])
        saved_images.append(np.asarray(vimages))
    saved_images = np.concatenate(saved_images, axis=0)

    # Save generated images
    monitor_image_tile.add(0, saved_images)


def main():
    args = get_args()
    save_args(args, "eval")

    evaluate(args)


if __name__ == '__main__':
    main()
