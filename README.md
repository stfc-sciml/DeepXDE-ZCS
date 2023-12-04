# DeepXDE-ZCS

This package extends [DeepXDE](https://github.com/lululxvi/deepxde) to support the trick of
Zero Coordinate Shift (ZCS, [arxiv preprint](https://arxiv.org/abs/2311.00860)), which can
reduce GPU memory and wall time for training physics-informed DeepONets by an order of magnitude
without affecting the training results.

Supported backends:

* Pytorch
* TensorFlow
* PaddlePaddle

# Installation

```bash
pip install git+https://github.com/stfc-sciml/DeepXDE-ZCS
```

Note that the above line will upgrade your `DeepXDE` to 1.10.0 or higher, as required by our extension.
Also, make sure you have one of the backends installed, which are not included in our requirements.

# Usage

If you are familiar with DeepXDE, using DeepXDE-ZCS is straightforward with the following two steps.

### Step 1: Replacing classes

Replacing the classes listed in the following table:

| **FROM**                                | **TO**                                    |
|-----------------------------------------|-------------------------------------------|
| `deepxde.icbc.DirichletBC`              | `deepxde_zcs.DirichletBC_ZCS`             |
| `deepxde.icbc.IC`                       | `deepxde_zcs.IC_ZCS`                      |
| `deepxde.data.PDEOperatorCartesianProd` | `deepxde_zcs.PDEOperatorCartesianProdZCS` |
| `deepxde.Model`                         | `deepxde_zcs.ModelZCS`                    |

**NOTE**: Not all boundary conditions have been extended, and we are working on this. If you need
one of them, please submit an issue to prioritize it. Doing this on your own is also easy, mostly by
adding the function dimension to the outputs; see [deepxde_zcs/boundary.py](deepxde_zcs/boundary.py).

### Step 2: Changing PDE to ZCS format

Take the diffusion-reaction equation for example. The original DeepXDE format reads like

```python
import deepxde as dde


def pde(x, u, v):
    d = 0.01
    k = 0.01
    u_t = dde.grad.jacobian(u, x, j=1)
    u_xx = dde.grad.hessian(u, x, j=0)
    return u_t - d * u_xx + k * u ** 2 - v
```

The DeepXDE-ZCS format looks similar, using our `LazyGradZCS` class for derivative calculation:

```python
import deepxde_zcs as ddez


def pde(zcs_parameters, u, v):
    d = 0.01
    k = 0.01
    grad_zcs = ddez.LazyGradZCS(u, zcs_parameters)
    u_t = grad_zcs.compute((0, 1))
    u_xx = grad_zcs.compute((2, 0))
    return u_t - d * u_xx + k * u ** 2 - v
```

In the above code:
* the input `zcs_parameters` is a `dict` generated by our code, and users don't need to care about what's in it;  
* the tuples `(0, 1)` and `(2, 0)` give the wanted differential orders w.r.t. $(x,t)$, i.e., 
`(0, 1)` for $\frac{\partial}{\partial t}$ and `(2, 0)` for $\frac{\partial^2}{\partial x^2}$;
* the `LazyGradZCS` class is smart enough to reuse any computed lower orders; for example, after computing `u_xx`, 
`u_x` will also be computed and cached, which can be given upon reqeust or serve as the starting point for 
computing `u_xy`.

# Example

The diffusion-reaction equation is provided as a complete example under [examples/diff_rec](examples/diff_rec).

To run this example (change `pytorch` to the backend you want):

* Aligned (original)
  ```bash
  DDE_BACKEND=pytorch python diff_rec_aligned_pideeponet.py
  ```
* Unaligned (original)
  ```bash
  DDE_BACKEND=pytorch python diff_rec_unaligned_pideeponet.py
  ```
* ZCS (ours)
  ```bash
  DDE_BACKEND=pytorch python diff_rec_aligned_pideeponet_zcs.py
  ```
  
If you have multiple GPUs, prepend `CUDA_VISIBLE_DEVICES=0` to the above commands
when comparing your results with those reported below. 

### Results

The GPU memory and wall time we measured on a Nvidia V100 (with CUDA 12.2) are reported below.
Note that this example is a small-scale problem for quick demo; ZCS can save more (in ratio) memory
and time for larger-scale problems (e.g., those with more functions, more points, and
higher-order PDEs).

* PyTorch backend

  | **METHOD**           | **GPU / MB** | **TIME / s** | 
  |----------------------|--------------|--------------|
  | Aligned (original)   | 5779         | 186          |
  | Unaligned (original) | 5873         | 117          |
  | ZCS (ours)           | 655          | 11           |

* TensorFlow backend
  
  Our ZCS implementation is currently not jit-compiled. We are working on it for (maybe) faster running.

  | **METHOD**           | **GPU / MB** | **TIME / s** | 
  |----------------------|--------------|--------------|
  | Aligned (original)   | 9205         | 73 (jit)     |
  | Unaligned (original) | 11694        | 70 (jit)     |
  | ZCS (ours)           | 591          | 35 (no jit)  |


* PaddlePaddle backend

  | **METHOD**           | **GPU / MB** | **TIME / s** | 
  |----------------------|--------------|--------------|
  | Aligned (original)   | 5805         | 197          |
  | Unaligned (original) | 6923         | 385          |
  | ZCS (ours)           | 1353         | 15           |

Enjoy saving!