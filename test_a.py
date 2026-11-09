"""
Test A
1D slab, 0<x<8 cm, 5 regions

left boundary: reflective
right boundary: vacuum

S_8 Gauss-Legendre Quadrature
40 uniform spatial cells

epsilon = 1e-6

Region 1: [0, 2] cm
    sigma_t = 50 cm-1
    sigma_s = sigma_f = nu_f = 0
    q       = 50 cm-3 s-1
Region 2: [2, 3] cm
    sigma_t = 5 cm-1
    sigma_s = sigma_f = nu_f = 0
    q       = 0
Region 3: [3, 5] cm
    (vacuum)
Region 4: [5, 6] cm
    sigma_t = 1.1 cm-1
    sigma_s = 0.7 cm-1
    sigma_f = 0.1 cm-1
    nu_f    = 2
    q       = 0
Region 5: [6, 8] cm
    sigma_t = 1.1 cm-1
    sigma_s = 0.5 cm-1
    sigma_f = 0.16 cm-1
    nu      = 2.5 cm-1
    q       = 1 cm-3, s-1
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sn1d.cornervalues import CornerValues
from sn1d.quadrature import Quadrature
from sn1d.source_iteration import solve_fixed_source
from sn1d.xs import CrossSection

# Directory to save plots
OUT_DIR = Path(__file__).resolve().parent / "test_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NAME = "test_a"

# discretization parameters

a = 8
n_x = 40
n_mu = 8
epsilon_phi = 1e-6
is_reflective = True


# spatial mesh

x_boundary = np.linspace(0, a, n_x+1)
dx = x_boundary[1:] - x_boundary[:-1]
x_center = 0.5*(x_boundary[1:] + x_boundary[:-1])

# angular quadrature

[mu, w] = np.polynomial.legendre.leggauss(n_mu)
quad = Quadrature(mu, w)

# boundary condition
# (reflective bc left)

bc = np.zeros(n_mu)

# xs assignment

sigma_t = np.zeros(n_x)
sigma_s = np.zeros(n_x)
sigma_f = np.zeros(n_x)
nu = np.zeros(n_x)
q = np.zeros(n_x)

# Region 1
idx = np.nonzero( x_center < 2)
sigma_t[idx] = 50
q[idx] = 50

# Region 2
idx = np.nonzero((x_center > 2) * (x_center < 3))
sigma_t[idx] = 5

# Region 3
idx = np.nonzero((x_center > 2) * (x_center < 3))

# Region 4
idx = np.nonzero((x_center > 5) * (x_center < 6))
sigma_t[idx] = 1.1
sigma_s[idx] = 0.7
sigma_f[idx] = 0.1
nu[idx] = 2

# Region 5
idx = np.nonzero(x_center > 6)
sigma_t[idx] = 1.1
sigma_s[idx] = 0.5
sigma_f[idx] = 0.16
nu[idx] = 2.5
q[idx] = 1


xs = CrossSection(sigma_t, sigma_s, sigma_f, nu)

phi_initial = CornerValues(np.zeros(n_x), np.zeros(n_x))

print("solving test A fixed source problem...")
solution, convergence, scheme, iterative = solve_fixed_source(epsilon_phi, phi_initial, q, xs, quad, dx, bc, is_reflective)

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
