import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImageTile
from nnabla.ext_utils import get_extension_context
import nnabla.utils.save as save
from functools import reduce

from datasets import data_iterator_lapsrn
from args import get_args, save_args
from models import get_loss, lapsrn
from helpers import (get_solver, upsample, downsample, 
                     split, to_BCHW, to_BHWC, normalize, ycbcr_to_rgb, 
                     normalize_method)

def train(args):
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model


    # Solver

    
    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)


    # DataIterator
    
    # Train loop
    for i in range(args.max_iter):
        # Feed data

        # Zerograd, forward, backward, weight-decay, update


        
        # Monitor and save


    # Monitor and save



def main():
    args = get_args()
    save_args(args, "train")

    train(args)

if __name__ == '__main__':
    main() 
