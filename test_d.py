"""
Test A
1D slab, 0<x<1 cm, 1 region

left boundary: vacuum
right boundary: vacuum

S_4 Double Gauss-Legendre Quadrature
10 0.1-cm cells

Region 1: [0, 1] cm
    sigma_t = 1/nu cm-1
    sigma_s = ((1/nu) - nu) cm-1
    sigma_f = 0 cm-1
    q       = nu

where the parameter nu is
(1) 1e-2
(2) 1e-4
"""
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py

from sn1d.cornervalues import CornerValues
from sn1d.quadrature import Quadrature, doubleGL
from sn1d.quasidiffusion import solve_fixed_source
from sn1d.xs import CrossSection

# Directory to save plots
OUT_DIR = Path(__file__).resolve().parent / "cp2_test_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NAME = "cp2_test_d"

a = 1.0
n_x = 10
n_mu = 4
epsilon_phi = 1e-10
is_reflective = False

# spatial mesh
x_boundary = np.linspace(0, 1, 11)
dx = x_boundary[1:] - x_boundary[:-1]
x_center = 0.5*(x_boundary[1:] + x_boundary[:-1])

# angular quadrature
quad = doubleGL(n_mu)
mu = quad.mu

# boundary condition
bc = np.zeros(2*n_mu)

# xs assignment
sigma_t = np.zeros(n_x)
sigma_s = np.zeros(n_x)
sigma_f = np.zeros(n_x)
nu = np.zeros(n_x)
q = np.zeros(n_x)



nu_vec = [1e-2, 1e-4]

for n in [0, 1]:
    nu = nu_vec[n]



    # Region 1
    idx = np.nonzero(x_center < 10)
    sigma_t[idx] = 1.0/nu
    sigma_s[idx] = (1.0 - nu**2)/nu
    q[idx] = nu


    xs = CrossSection(sigma_t, sigma_s, 0*sigma_s, 0*sigma_s)

    soln, convergence, closures = solve_fixed_source(epsilon_phi, q, xs, quad, dx, bc, False, update_interval=1)
    phi_avg = (1.0/6)*(4*soln.phi + soln.phi_edge[1:] + soln.phi_edge[:-1])

    file = h5py.File(f"{NAME}{n+1}.h5", "w")

    grid = file.create_group("grid")
    grid.create_dataset("cell-edges", data = x_boundary)
    grid.create_dataset("cell-centers", data = x_center)
    grid.create_dataset("dx", data= dx)

    gquad = file.create_group("quadrature")
    gquad.create_dataset("mu", data= quad.mu)
    gquad.create_dataset("w", data= quad.w)
    gquad.attrs["quadrature-set"] = "S4 Double Gauss Legendre"

    gsoln = file.create_group("solution")
    gsoln.create_dataset("scalar-flux/cell-center", data= soln.phi)
    gsoln.create_dataset("scalar-flux/cell-edge", data= soln.phi_edge)
    gsoln.create_dataset("scalar-flux/cell-average", data=phi_avg)

    gsoln.create_dataset("angular_flux/cell-center", data= np.array([soln.psi[m].cell_average() for m in range(n_mu*2)]))
    gsoln.create_dataset("current/cell-edge", data = soln.J_edge)

    gclosure = file.create_group("closures")
    gclosure.create_dataset("qd-factor/cell-center", data = closures.E_center)
    gclosure.create_dataset("qd-factor/cell-edge", data= closures.E_edge)
    gclosure.create_dataset("boundary-factor/left", data=closures.C_0)
    gclosure.create_dataset("boundary-factor/right", data=closures.C_X)

    gconvergence = file.create_group("convergence")
    gconvergence.create_dataset("number-iterations", data=convergence.N_iter)
    gconvergence.create_dataset("error", data=convergence.delta)
    gconvergence.create_dataset("spectral-radius", data=convergence.rho)




