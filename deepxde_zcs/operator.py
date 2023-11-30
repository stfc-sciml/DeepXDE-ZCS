"""
PDE operator extensions for ZCS support
"""

import deepxde as dde
import numpy as np


class PDEOperatorCartesianProdZCS(dde.data.PDEOperatorCartesianProd):
    """
    Derived `PDEOperatorCartesianProd` class for ZCS support
    """

    def _losses(self, outputs, loss_fn, inputs, model, num_func):
        bkd = dde.backend

        # PDE
        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(model.zcs_parameters, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        error_f = [fi[:, bcs_start[-1]:] for fi in f]
        losses = [loss_fn(bkd.zeros_like(error), error) for error in error_f]  # noqa

        # BC
        for i, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(
                self.train_x[1],
                inputs[1],
                # add component dim for DirichletBC_ZCS and IC_ZCS
                outputs[:, :, None],
                beg,
                end,
                aux_var=model.net.auxiliary_vars,
            )
            losses.append(loss_fn(bkd.zeros_like(error), error))  # noqa
        return losses
