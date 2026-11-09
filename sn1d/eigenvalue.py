import numpy as np

from sn1d.cornervalues import CornerValues
from sn1d.quadrature import Quadrature
from sn1d.sweep import sweep
from sn1d.xs import CrossSection
from sn1d.source_iteration import Solution, Convergence, Residuals



# solve the fixed-source problem with source iteration
def solve_eigenvalue(
    epsilon_phi : float,
    epsilon_k : float,
    phi_initial : CornerValues,
    xs : CrossSection,
    quad : Quadrature,
    dx : np.array,
    lhs_reflective : bool
):

    BC = np.zeros(quad.M)
    q = np.zeros(len(dx))
    # initialize guess for scalar flux and eigenvalue
    phi = phi_initial.copy()
    k = np.array([1])


    delta_phi = np.array([1.0])
    rho_phi = np.array([1.0])

    delta_k = np.array([1.0])
    rho_k = np.array([1.0])

    # while not converged:
    l = 0
    while  (delta_phi[-1] > (epsilon_phi * ((1/rho_phi[-1]) - 1)) or delta_k[-1] > (epsilon_k * ((1/rho_k[-1]) - 1))):
        l += 1
        # print(f"Iteration {l}...")
        # solve for angular and scalar flux given known scattering source
        psi, phi_new = sweep(phi, q, xs, quad, dx, np.zeros(quad.M), lhs_reflective, k[-1])

        # compute k
        num = (xs.fission*xs.multiplicity)*phi_new.cell_average() @ dx
        denom = (xs.fission*xs.multiplicity)*phi.cell_average() @ dx
        k = np.append(k, k[-1]*num/denom)
        
        delta_k = np.append(delta_k, np.abs(k[l] - k[l-1]))
        rho_k = np.append(rho_k, delta_k[l]/delta_k[l-1])

        # compute delta_phi
        delta_phi = np.append(delta_phi, (phi_new - phi).norm_inf())
        rho_phi = np.append(rho_phi, delta_phi[l]/delta_phi[l-1])
        
        phi_old = phi.copy()
        phi = phi_new.copy()

        # print(f"finished iteration {l}; delta_phi = {delta_phi[-1]:.4e}, rho_phi = {rho_phi[-1]:.4e}")

    scale = phi.cell_average() @ dx
    phi /= scale
    phi_old /= scale
    psi /= scale
    k_used = k[-2]  # k used for the final sweep's fission source

    # cell-average angular flux
    cell_average_angular = np.array([psi_m.cell_average() for psi_m in psi])
    cell_edge_angular = np.zeros((quad.M, len(dx)+1))

    for m in range(quad.M):
        if quad.mu[m] < 0:
            cell_edge_angular[m, 0:-1] = psi[m].L
            cell_edge_angular[m, -1] = BC[m]
        if quad.mu[m] > 0:
            cell_edge_angular[m, 1:] = psi[m].R
            if lhs_reflective == True:
                cell_edge_angular[m, 0] = psi[quad.M-(m+1)].L[0]
            else:
                cell_edge_angular[m, 0] = BC[m]


    cell_average_scalar = np.transpose(cell_average_angular) @ quad.w
    cell_edge_scalar = np.transpose(cell_edge_angular) @ quad.w
    cell_average_scalar_old = phi_old.cell_average()
    legendre_scalar_old = phi_old.first_moment()
    legendre_scalar = phi.first_moment()
    cell_average_current = np.transpose(cell_average_angular) @ (quad.w * quad.mu)
    cell_edge_current = np.transpose(cell_edge_angular) @ (quad.w * quad.mu)

    # calculate residuals
    scheme_balance = np.longdouble(cell_edge_current[1:]) - np.longdouble(cell_edge_current[:-1]) \
        + np.longdouble(xs.total)*np.longdouble(cell_average_scalar)*np.longdouble(dx) \
        - (np.longdouble(xs.scatter) + (np.longdouble(xs.fission) * np.longdouble(xs.multiplicity))/np.longdouble(k_used))*np.longdouble(cell_average_scalar_old)*np.longdouble(dx) \
        - np.longdouble(q*dx)

    scheme_firstmoment = 3*np.longdouble(cell_edge_current[1:]) + 3*np.longdouble(cell_edge_current[:-1]) - 6*np.longdouble(cell_average_current) \
        + np.longdouble(xs.total)*np.longdouble(legendre_scalar)*np.longdouble(dx) \
        - (np.longdouble(xs.scatter) + np.longdouble(xs.fission)*np.longdouble(xs.multiplicity)/np.longdouble(k_used))*np.longdouble(legendre_scalar_old)*np.longdouble(dx)

    iterative_balance = np.longdouble(cell_edge_current[1:]) - np.longdouble(cell_edge_current[:-1]) \
        + (np.longdouble(xs.total)- np.longdouble(xs.scatter + (xs.fission * xs.multiplicity)/k[-1]))*np.longdouble(cell_average_scalar)*np.longdouble(dx) \
        - np.longdouble(q*dx)

    iterative_firstmoment = 3*np.longdouble(cell_edge_current[1:]) + 3*np.longdouble(cell_edge_current[:-1]) - 6*np.longdouble(cell_average_current) \
        + np.longdouble(xs.total - xs.scatter - (xs.fission * xs.multiplicity)/k[-1])*np.longdouble(legendre_scalar)*np.longdouble(dx)


    scheme_residuals = Residuals(scheme_balance, scheme_firstmoment)
    iterative_residuals = Residuals(iterative_balance, iterative_firstmoment)

    return k[-1], Solution(phi, psi, cell_edge_scalar, cell_edge_current), Convergence(l, delta_phi, rho_phi), scheme_residuals, iterative_residuals