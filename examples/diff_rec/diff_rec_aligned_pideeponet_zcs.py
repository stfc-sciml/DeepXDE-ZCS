"""
Kuangdai:
Adapted from
https://github.com/lululxvi/deepxde/blob/master/examples/operator/diff_rec_aligned_pideeponet.py
Changes are commented with big hash boxes
"""

"""Backend supported: tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

from ADR_solver import solve_ADR

########################
# import ZCS extension #
########################
# uncomment the following two lines to run in repo without installation
# import sys
# sys.path.append('../..')
import deepxde_zcs as ddez


#################################
# change PDE to ZCS formulation #
#################################
def pde(zcs_parameters, u_, v_):
    """ DR PDE with ZCS """
    D = 0.01
    k = 0.01
    grad_zcs = ddez.LazyGradZCS(u_, zcs_parameters)
    u_t = grad_zcs.compute((0, 1))
    u_xx = grad_zcs.compute((2, 0))
    return u_t - D * u_xx + k * u_ ** 2 - v_


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

####################
# boundary for ZCS #
####################
bc = dde.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

#######################################
# increase point sampling by 20 times #
#######################################
pde = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=200 * 20,
    num_boundary=40 * 20,
    num_initial=20 * 20,
    num_test=500,
)

# Function space
func_space = dde.data.GRF(length_scale=0.2)

# Data
eval_pts = np.linspace(0, 1, num=50)[:, None]
####################
# operator for ZCS #
####################
data = ddez.PDEOperatorCartesianProdZCS(
    pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=50
)

# Net
net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [2, 128, 128, 128],
    "tanh",
    "Glorot normal",
)

####################
# model for ZCS #
####################
model = ddez.ModelZCS(data, net)
model.compile("adam", lr=0.0005)

########################################
# decrease iterations to 1000 for demo #
########################################
losshistory, train_state = model.train(iterations=1000, display_every=100)
dde.utils.plot_loss_history(losshistory)

func_feats = func_space.random(1)
xs = np.linspace(0, 1, num=100)[:, None]
v = func_space.eval_batch(func_feats, xs)[0]
x, t, u_true = solve_ADR(
    0,
    1,
    0,
    1,
    lambda x: 0.01 * np.ones_like(x),
    lambda x: np.zeros_like(x),
    lambda u: 0.01 * u**2,
    lambda u: 0.02 * u,
    lambda x, t: np.tile(v[:, None], (1, len(t))),
    lambda x: np.zeros_like(x),
    100,
    100,
)
u_true = u_true.T
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((v_branch, x_trunk))
u_pred = u_pred.reshape((100, 100))
print(dde.metrics.l2_relative_error(u_true, u_pred))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
