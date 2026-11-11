import numpy as np
from numpy.linalg import solve as linsolve
from numba import njit

from sn1d.cornervalues import CornerValues
from sn1d.quadrature import Quadrature
from sn1d.sweep import sweep
from sn1d.xs import CrossSection
from sn1d.source_iteration import Solution, Residuals, Convergence

from matplotlib.pyplot import spy as spy
from matplotlib.pyplot import show as show
import matplotlib
matplotlib.use('Qt5Agg')

class Closure:
    def __init__(self, E_center, E_edge, C_0, C_X):
        self.E_center = E_center
        self.E_edge = E_edge
        self.C_0 = C_0
        self.C_X = C_X

def _solve_loqd(
    q : np.array,
    xs : CrossSection,
    dx : np.array,
    J_in : np.array,
    phi_in : np.array,
    E_b : np.array,
    E_c : np.array,
    C_0 : float,
    C_X : float,
    lhs_reflective : bool,
):
    xs_t = xs.total
    xs_s_eff = (xs.scatter + (xs.multiplicity * xs.fission))
    xs_a = xs_t - xs_s_eff

    n_x = dx.size
    A = np.zeros((3*n_x + 2, 3*n_x + 2))
    b = np.zeros(3*n_x + 2)

    # interior equations
    for i in range(n_x):
        p_c = i
        j_l = n_x + i
        j_r = n_x + i + 1
        p_l = (2*n_x + 1) + i
        p_r = (2*n_x + 1) + i + 1
        
        bal = (3*i)
        left = (3*i) + 1
        right = (3*i) + 2
        # balance equation
        A[bal, j_l:j_r+1] = [-1, 1]
        A[bal, p_c] = (2.0/3)*xs_a[i]*dx[i]
        A[bal, p_l:p_r+1] = (1.0/6)*xs_a[i]*dx[i]
        # left side
        A[left, p_c] = E_c[i]
        A[left, p_l] = -E_b[i]
        A[left, j_l] = 0.5*xs_t[i]*dx[i]
        # right side
        A[right, p_r] = E_b[i+1]
        A[right, p_c] = -E_c[i]
        A[right, j_r] = 0.5*xs_t[i]*dx[i]

        # balance eq'n source
        b[bal] = dx[i]*q[i]


    # boundary condition equations
    i = 0
    j_l = n_x + i
    p_l = (2*n_x + 1) + i
    bal = (3*i)
    left = (3*i) + 1
    right = (3*i) + 2
    if lhs_reflective == True:
        A[-2, j_l] = 1
        b[-2] = 0.0
    else:
        A[-2, j_l] = 1
        A[-2, p_l] = -C_0
        b[-2] = J_in[0] - C_0*phi_in[0]
    
    i = n_x - 1
    j_r = n_x + i + 1
    p_r = (2*n_x + 1) + i + 1
    A[-1, j_r] = 1
    A[-1, p_r] = -C_X
    b[-1] = J_in[1] - C_X*phi_in[1]

    result = linsolve(A, b)

    phi_i = result[0:n_x]
    J_b   = result[n_x:(2*n_x)+1]
    phi_b = result[(2*n_x)+1:]

    return phi_i, phi_b, J_b



def _calc_phi_in(
    psi_in: np.ndarray,
    quad: Quadrature
):
    # psi_in is a n_mu array
    # multiply by weights
    
    weighted = psi_in * quad.w
    phi_in = np.array([0.0, 0.0])
    phi_in[0] = np.sum(weighted[quad.mu>0])
    phi_in[1] = np.sum(weighted[quad.mu<0])

    return phi_in

def _calc_J_in(
    psi_in: np.ndarray,
    quad: Quadrature
):
    # psi_in is a n_mu array
    # multiply by weights
    
    weighted = psi_in * quad.w * quad.mu
    J_in = np.array([0.0, 0.0])
    J_in[0] = np.sum(weighted[quad.mu>0])
    J_in[1] = np.sum(weighted[quad.mu<0])

    return J_in

def _prolongate(
    phi_i : np.ndarray, 
    phi_b : np.ndarray
):
    n_x = phi_i.size
    
    left = np.zeros(n_x)
    right = np.zeros(n_x)

    left  = ((2.0/3)*phi_i) + ((2.0/3)*phi_b[:-1]) - ((1.0/3)*phi_b[1:])
    right = ((2.0/3)*phi_i) + ((-1.0/3)*phi_b[:-1]) + ((2.0/3)*phi_b[1:])


    return CornerValues(left, right)

def _calculate_QD_factor(
    psi  : list, 
    quad : np.ndarray,
    BC   : np.ndarray,
    is_reflective = False
):
    n_x = psi[0].L.size
    E_center_denom = np.zeros(n_x)
    E_center_num = np.zeros(n_x)
    E_edge_num = np.zeros(n_x + 1)
    E_edge_denom = np.zeros(n_x + 1)
    for m in range(quad.mu.size):
        # cell-centered; all angles
        E_center_num += quad.mu[m]**2 * quad.w[m] * (psi[m].L + psi[m].R)
        E_center_denom += quad.w[m] * (psi[m].L + psi[m].R)

        # cell-edge
        if quad.mu[m] > 0: # contribute to right edge
            E_edge_num[1:] += quad.mu[m]**2 * quad.w[m] * (psi[m].R)
            E_edge_denom[1:] +=  quad.w[m] * (psi[m].R)
        else:
            E_edge_num[:-1] += quad.mu[m]**2 * quad.w[m] * (psi[m].L)
            E_edge_denom[:-1] +=  quad.w[m] * (psi[m].L)
    # right face
    E_edge_num[-1] += np.sum(quad.mu**2 * quad.w * BC * (quad.mu < 0))
    E_edge_denom[-1] += np.sum(quad.w * BC * (quad.mu < 0))

    # left face
    if is_reflective == True:
        pass
    else:
        E_edge_num[0] += np.sum(quad.mu**2 * quad.w * BC * (quad.mu > 0))
        E_edge_denom[0] += np.sum(quad.w * BC * (quad.mu > 0))

    return (E_center_num/E_center_denom), (E_edge_num/E_edge_denom)

def _calculate_boundary_factors(
    psi  : list, 
    quad : np.ndarray
):
    C_0 = np.sum(
        [psi[m].L[0]*quad.w[m]*quad.mu[m]*(quad.mu[m]<0) for m in range(quad.mu.size)]
    ) / np.sum(
        [psi[m].L[0]*quad.w[m]*(quad.mu[m]<0) for m in range(quad.mu.size)]
    )

    C_X = np.sum(
        [psi[m].R[-1]*quad.w[m]*quad.mu[m]*(quad.mu[m]>0) for m in range(quad.mu.size)]
    ) / np.sum(
        [psi[m].R[-1]*quad.w[m]*(quad.mu[m]>0) for m in range(quad.mu.size)]
    )
    return C_0, C_X


# solve the fixed-source problem with source iteration
def solve_fixed_source(
    epsilon_phi : float,
    q : np.array,
    xs : CrossSection,
    quad : Quadrature,
    dx : np.array,
    BC : np.array,
    lhs_reflective : bool,
    update_interval = 1
):
    n_x = dx.size
    
    # initialize guess for scalar flux
    phi_i = np.ones(n_x)
    phi_b = np.ones(n_x+1)

    # compute LOQD boundary conditions
    phi_in = _calc_phi_in(BC, quad)
    J_in = _calc_J_in(BC, quad)

    # initialize closures
    E_c = np.ones(n_x)*(1.0/3)
    E_b = np.ones(n_x + 1)*(1.0/3)
    C_0 = -0.5
    C_X = 0.5

    # convergence log
    delta = np.array([1.0])
    rho = np.array([0.5])

    # while not converged:
    s = 0
    while (delta[-1] > (epsilon_phi * ((1/rho[-1]) - 1))):
        s += 1
        if s% update_interval == 0:
            print(f"Iteration {s}...")
        if s > 1:
            phi_ld = _prolongate(phi_i, phi_b)
            psi, _ = sweep(phi_ld, q, xs, quad, dx, BC, lhs_reflective, lumped=True)
            E_c, E_b = _calculate_QD_factor(psi, quad, BC, lhs_reflective)
            C_0, C_X = _calculate_boundary_factors(psi, quad)

            print(f"C_0 = {C_0}, C_X = {C_X}")

        # QD solve
        old_phi_center = phi_i.copy()
        phi_i, phi_b, J_b = _solve_loqd(q, xs, dx, J_in, phi_in, E_b, E_c, C_0, C_X, lhs_reflective)
        
        delta = np.append(delta, np.max(np.abs(phi_i - old_phi_center)))
        if s > 2:
            rho = np.append(rho, delta[-1]/delta[-2])
        else:
            rho = np.append(rho, 0.5)

    scheme_balance = np.longdouble(J_b[1:]) - np.longdouble(J_b[:-1]) \
        + np.longdouble(xs.total)*np.longdouble(phi_i)*np.longdouble(dx) \
        - (np.longdouble(xs.scatter) + (np.longdouble(xs.fission) * np.longdouble(xs.multiplicity)))*np.longdouble(old_phi_center)*np.longdouble(dx) \
        - np.longdouble(q*dx)
    print(scheme_balance)


    return Solution(phi_i, psi, phi_b, J_b), Convergence(s, delta, rho), Closure(E_c, E_b, C_0, C_X)