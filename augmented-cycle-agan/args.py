#TODO: set better default hyper parameters


def get_args(monitor_path='tmp.monitor', max_epoch=200, model_save_path=None,
             learning_rate=2*1e-4,
             batch_size=1, description=None):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os
    if model_save_path is None:
        model_save_path = monitor_path
    if description is None:
        description = ("NNabla implementation of CycleGAN. The following help shared among examples in this folder. "
                       "Some arguments are valid or invalid in some examples.")
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-epoch", "-e", type=int, default=max_epoch,
                        help='Max epoch of training. Epoch is determined by the max of the number of images for two domains.')
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path,
                        help='Path where model parameters are saved.')
    parser.add_argument("--model-load-path",
                        type=str,
                        help='Path where model parameters are loaded.')
    parser.add_argument("--dataset",
                        type=str, default="edges2shoes", choices=["cityscapes",
                                                                  "edges2handbags",
                                                                  "edges2shoes",
                                                                  "facades",
                                                                  "maps"],
                        help='Dataset to be used.')
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration (float or half)')
    parser.add_argument('--lambda-recon', type=float,
                        default=10.,
                        help="Coefficient for reconstruction loss.")
    parser.add_argument('--lambda-idt', type=float,
                        default=0,
                        help="Coefficient for identity loss. Default is 0, but set 0.5 to comply with the pytorch cycle-gan implementation.")
    parser.add_argument('--unpool', action='store_true')
    parser.add_argument('--init-method', default=None, type=str,
                        help="`None`|`paper`")
    parser.add_argument('--latent', default=256, 
                        help="Number of dimensions of latent variables.")
    
    args = parser.parse_args()
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    return args


def save_args(args):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments.txt".format(args.monitor_path)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))
