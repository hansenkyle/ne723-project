"""
Test B
1D slab, 0<x<11 cm, 2 regions

left boundary: 
    psi(mu) = 1 for mu>0
right boundary: vacuum

S_8 Gauss-Legendre Quadrature
10 0.1-cm cells, 10 1-cm cells.

Region 1: [0, 1] cm
    sigma_t = 2 cm-1
    sigma_s = sigma_f = nu_f = 0
    q       = 0
Region 2: [1, 11] cm
    sigma_t = 100 cm-1
    sigma_s = 100 cm-1
    sigma_f = nu_f = 0
    q       = 0
"""
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sn1d.cornervalues import CornerValues
from sn1d.quadrature import Quadrature
from sn1d.source_iteration import solve_fixed_source
from sn1d.xs import CrossSection

# Directory to save plots
OUT_DIR = Path(__file__).resolve().parent / "test_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NAME = "test_b"

a = 11
n_x = 20
n_mu = 8
epsilon_phi = 1e-6
is_reflective = False

# spatial mesh
r1_bound = np.linspace(0, 1, 11)
r2_bound = np.linspace(2, 11, 10)
x_boundary = np.concatenate((r1_bound, r2_bound))
dx = x_boundary[1:] - x_boundary[:-1]
x_center = 0.5*(x_boundary[1:] + x_boundary[:-1])

# angular quadrature
[mu, w] = np.polynomial.legendre.leggauss(n_mu)
quad = Quadrature(mu, w)

# boundary condition
bc = np.zeros(n_mu)
bc[mu>0] = 1.0

# xs assignment
sigma_t = np.zeros(n_x)
sigma_s = np.zeros(n_x)
sigma_f = np.zeros(n_x)
nu = np.zeros(n_x)
q = np.zeros(n_x)

# Region 1
idx = np.nonzero(x_center < 1)
sigma_t[idx] = 2
# Region 2
idx = np.nonzero(x_center > 1)
sigma_t[idx] = 100
sigma_s[idx] = 100

xs = CrossSection(sigma_t, sigma_s, 0*sigma_s, 0*sigma_s)


phi_initial = CornerValues(np.zeros(n_x), np.zeros(n_x))


print("solving test A fixed source problem...")
solution, convergence, scheme, iterative = solve_fixed_source(epsilon_phi, phi_initial, q, xs, quad, dx, bc, is_reflective, 100)

# Output and Plots

##################
# Output and Plots
##################

# number of iterations N_iter
print(f"\nN_iter = {convergence.N_iter}")

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['lines.marker'] = '.'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.grid'] = True


# graph of Delta vs l
plt.figure()
plt.plot(convergence.delta)
plt.yscale('log')
plt.title("Absolute iterative error")
plt.xlabel("$\\ell$")
plt.ylabel("$|\\phi^{(\\ell)} - \\phi^{(\\ell - 1)}|_\\infty$")
plt.savefig(OUT_DIR / (NAME+"_iterative_error.svg"))

# graph of phi_rho vs l
plt.figure()
plt.plot(convergence.rho)
plt.yscale('log')
plt.title("Spectral Radius Estimate $\\rho_\\phi$")
plt.xlabel("$\\ell$")
plt.ylabel("$\\rho_\\phi^{(\\ell)}$")
plt.savefig(OUT_DIR / (NAME+"_spectral_radius.svg"))

# Cell-average scalar flux
plt.figure()
plt.plot(x_center, solution.phi.cell_average())
plt.xlabel("$x$ [cm]")
plt.ylabel("$\\overline{\\phi}_i$")
plt.title("Cell-average scalar flux")
plt.savefig(OUT_DIR / (NAME+"_cell_average_scalar.svg"))

# First spatial moment of scalar flux
plt.figure()
plt.plot(x_center, solution.phi.first_moment())
plt.xlabel("$x$ [cm]")
plt.ylabel("$\\hat{\\phi}_i$")
plt.title("First Legendre spatial moment of scalar flux")
plt.savefig(OUT_DIR / (NAME+"_scalar_moment.svg"))

# cell-edge scalar flux
plt.figure()
plt.plot(x_boundary, solution.phi_edge)
plt.xlabel("$x$ [cm]")
plt.ylabel("$\\phi_{i+1/2}$")
plt.title("Cell-edge scalar flux")
plt.savefig(OUT_DIR / (NAME+"_scalar_edge.svg"))

# cell-edge current
plt.figure()
plt.plot(x_boundary, solution.J_edge)
plt.axhline(y=0.0, color="black", linewidth=0.8)
plt.xlabel("$x$ [cm]")
plt.ylabel("$J_{i+1/2}$")
plt.title("Cell-edge current")
plt.savefig(OUT_DIR / (NAME+"_current_edge.svg"))

# cell-average angular flux
plt.figure()
for m in range(0, n_mu):
    plt.plot(x_center, solution.psi[m].cell_average(), label=f"$\\mu =$ {mu[m]:.4f}")
plt.legend()
plt.xlabel("$x$ [cm]")
plt.ylabel("$\\psi_{m, i}$")
plt.title("Cell-average angular flux")
plt.savefig(OUT_DIR / (NAME+"_cell_average_angular.svg"))

# Residuals vs position

# scheme, balance
plt.figure()
plt.plot(x_center, np.abs(scheme.balance))
plt.xlabel("$x$ [cm]")
plt.ylabel("$|\\delta\\overline{r}_{i}|$")
plt.title("Scheme residuals, balance")
plt.yscale('log')
plt.savefig(OUT_DIR / (NAME+"_schemeresid_balance.svg"))

# scheme, first moment
plt.figure()
plt.plot(x_center, np.abs(scheme.first_moment))
plt.xlabel("$x$ [cm]")
plt.ylabel("$|\\delta\\hat{r}_{i}|$")
plt.title("Scheme residuals, first Legendre spatial moment")
plt.yscale('log')
plt.savefig(OUT_DIR / (NAME+"_schemeresid_firstmoment.svg"))

# iterative, balance
plt.figure()
plt.plot(x_center, np.abs(iterative.balance))
plt.xlabel("$x$ [cm]")
plt.ylabel("$|\\delta\\overline{R}_{i}|$")
plt.title("Iterative residuals, balance")
plt.yscale('log')
plt.savefig(OUT_DIR / (NAME+"_iterativeresidu_balance.svg"))

plt.figure()
plt.plot(x_center, iterative.balance)
plt.xlabel("$x$ [cm]")
plt.ylabel("$\\delta\\overline{R}_{i}$")
plt.title("Iterative residuals, balance")
plt.savefig(OUT_DIR / (NAME+"_iterativeresidu_balance_linscale.svg"))

# iterative, first moment
plt.figure()
plt.plot(x_center, np.abs(iterative.first_moment))
plt.xlabel("$x$ [cm]")
plt.ylabel("$|\\delta\\hat{R}_{i}|$")
plt.title("Iterative residuals, first Legendre spatial moment")
plt.yscale('log')
plt.savefig(OUT_DIR / (NAME+"_iterativeresid_firstmoment.svg"))

plt.figure()
plt.plot(x_center, iterative.first_moment)
plt.xlabel("$x$ [cm]")
plt.ylabel("$\\delta\\hat{R}_{i}$")
plt.title("Iterative residuals, first Legendre spatial moment")
plt.savefig(OUT_DIR / (NAME+"_iterativeresid_firstmoment_linscale.svg"))

plt.show()