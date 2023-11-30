"""
Boundary extension for ZCS support
"""

import deepxde as dde


class DirichletBC_ZCS(dde.icbc.DirichletBC):  # noqa
    """ derived `DirichletBC` class for ZCS support """

    def error(self, x, inputs, outputs, beg, end, aux_var=None):
        values = self.func(x, beg, end, aux_var)
        return outputs[:, beg:end, self.component:self.component + 1] - values


class IC_ZCS(dde.icbc.IC):  # noqa
    """ derived `IC` class for ZCS support """

    def error(self, x, inputs, outputs, beg, end, aux_var=None):
        values = self.func(x, beg, end, aux_var)
        return outputs[:, beg:end, self.component:self.component + 1] - values
