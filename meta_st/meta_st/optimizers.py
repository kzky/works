import math

import numpy

from chainer import cuda
from chainer import optimizers as opts

class Adam(opts.Adam):

    """Adam optimization algorithm.

    See: https://arxiv.org/abs/1412.6980v8

    """

    def update(self, params_dict, lossfun=None, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        This method runs in two ways.

        - If ``lossfun`` is given, then use it as a loss function to compute
          gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.

        In both cases, the computed gradients are used to update parameters.
        The actual update routines are defined by the :meth:`update_one`
        method (or its CPU/GPU versions, :meth:`update_one_cpu` and
        :meth:`update_one_gpu`).

        """
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', False)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward()
            del loss

        self.reallocate_cleared_grads()

        self.call_hooks()
        self.prepare()

        self.t += 1
        states = self._states
        for name, param in params_dict.items():
            with cuda.get_device_from_array(param.data):
                self.update_one(param, states[name])
