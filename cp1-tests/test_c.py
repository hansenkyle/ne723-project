"""
Test C
1D slab, 0<x<a cm, 1 region

vacuum boundary (k-eigenvalue problem)

S_8 Gauss-Legendre Quadrature
50 uniform spatial cells

Region 1: [0, 1] cm
    sigma_t = 1 cm-1
    sigma_s = 0.45 cm-1
    sigma_f = 0.3 cm-1
    nu_f = 2
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
from sn1d.eigenvalue import solve_eigenvalue
from sn1d.xs import CrossSection

# Directory to save plots
OUT_DIR = Path(__file__).resolve().parent / "test_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NAME = "test_c"




# discretization parameters

a_bounds = [1e-2, 1e1]
n_x = 50
n_mu = 8
epsilon_phi = 1e-6
epsilon_k = 1e-6
is_reflective = False


# angular quadrature

[mu, w] = np.polynomial.legendre.leggauss(n_mu)
quad = Quadrature(mu, w)


# spatial mesh -- defer

# x_boundary = np.linspace(0, a, n_x+1)
# dx = x_boundary[1:] - x_boundary[:-1]
# x_center = 0.5*(x_boundary[1:] + x_boundary[:-1])

# xs assignment

sigma_t = 1.0*np.ones(n_x)
sigma_s = 0.45*np.ones(n_x)
sigma_f = 0.3*np.ones(n_x)
nu = 2*np.ones(n_x)

xs = CrossSection(sigma_t, sigma_s, sigma_f, nu)

phi_initial = CornerValues(np.ones(n_x), np.ones(n_x))

bounds = a_bounds
epsilon_a = 1e-6
delta_a = 1

a_guesses = np.array(bounds)
while delta_a > epsilon_a:
    cut = 0.5*(bounds[0] + bounds[1])
    a_guesses = np.append(a_guesses, cut)
    dx = (cut/n_x)*np.ones(n_x)
    # evaluate k
    print(f"Evaluating with a = {cut:.6e} cm")
    k, solution, convergence, scheme, iterative \
         = solve_eigenvalue(epsilon_phi, epsilon_k, phi_initial, xs, quad, dx, is_reflective)
    if k > 1: bounds[1] = cut
    if k < 1: bounds[0] = cut
    delta_a = bounds[1] - bounds[0]
    phi_initial = solution.phi

x_boundary = np.linspace(0, a_guesses[-1], n_x+1)
x_center = 0.5*(x_boundary[1:] + x_boundary[:-1])




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

plt.figure()
plt.plot(a_guesses)
plt.title("Critical slab width estimations")
plt.xlabel("$s$")
plt.ylabel("$a_{crit}^{(s)}$")
plt.savefig(OUT_DIR / (NAME+"_a_crit.svg"))
# plt.yscale('log')


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