# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_args(batch_size=32, ih=128, iw=128, max_iter=158275, save_interval=1000, maps=16):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os

    description = "Example of cVAEs"
    parser = argparse.ArgumentParser(description)

    parser.add_argument("-d", "--device-id", type=str, default="0",
                        help="Device id.")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Context.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--train", type=bool, default=True,
                        help="Train mode")
    parser.add_argument("--ih", type=int, default=ih,
                        help="Image height.")
    parser.add_argument("--iw", type=int, default=iw,
                        help="Image width.")
    parser.add_argument("--max-iter", type=int, default=max_iter,
                        help="Max iterations. Default is 25 epoch with 32 batch size")
    parser.add_argument("--maps", type=int, default=maps, 
                        help="Initial feature maps")
    parser.add_argument("--save-interval", type=int, default=10000,
                        help="Interval for saving models.")
    parser.add_argument("--monitor-path", type=str, default="./result/example_0",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str,
                        help="Model load path to a h5 file used in generation and validation.")
    parser.add_argument("--sigma", type=float, default=1.0, 
                        help="Noise level")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="beta1 of Adam")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Lambda for weights of objectives.")
    parser.add_argument("--use-pfvn", action="store_true",
                        help="Use pixel-wise feature vector normalization for encodings.")
    parser.add_argument("--use-patch", action="store_true",
                        help="Use patch also for fft.")
    parser.add_argument("--weight-decay-rate", type=float, default=1e-4, 
                        help="Weight decay rate")
    parser.add_argument("--train-data-path", "-T", type=str, default="",
                        help='Path to training data')
    parser.add_argument("--valid-data-path", "-V", type=str, default="",
                        help='Validation data to be used')
    parser.add_argument("--valid-metric", type=str, default="",
                        choices=[""],
                        help="Validation metric for reconstruction.")
    parser.add_argument("--use-deconv", action="store_true",
                        help="Use deconvolution in decocer. Default is unpooling -> convolution.")
    args = parser.parse_args()

    return args


def save_args(args, mode="train"):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments-{}.txt".format(args.monitor_path, mode)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))
