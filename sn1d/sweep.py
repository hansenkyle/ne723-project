import numpy as np
from numba import njit

from sn1d.cornervalues import CornerValues
from sn1d.quadrature import Quadrature
from sn1d.xs import CrossSection


# solve the transport equation in a single cell, single angle, discretized with LD finite elements
@njit
def ldSolve(
        mu : float, 
        dx : float,
        psi_in : float, # upwind angular flux (BC or neighboring cell)
        xs_t : float,
        xs_s : float,
        xs_f : float,
        nu_f : float,
        q    : float,
        phi  : tuple   # expected in (L, R) format. this function re-orders as necessary
):
    s_eff = (xs_s + nu_f*xs_f)
    m = abs(mu)
    if mu<0:
        phi_l = phi[1]
        phi_r = phi[0]
    else:
        phi_l = phi[0]
        phi_r = phi[1]
    psi_l = (
                dx**2 * xs_t * (q+s_eff*phi_l) 
                + (12*m**2 * psi_in) 
                + (dx*m*(s_eff*(phi_l - phi_r) + 8*xs_t*psi_in))
             )/(
                2*(6*m**2 + 4*dx*m*xs_t + dx**2 * xs_t**2)
             )
    psi_r = (
                0.5*dx*(s_eff*(phi_l/6 + phi_r/3)+0.5*q) + (0.5*m - (dx*xs_t)/6)*psi_l
            )/(
                0.5*m + (xs_t*dx)/3
            )
    if mu>0:
        return psi_l, psi_r
    else:
        return psi_r, psi_l


# sweep all angles, all cells, given known scattering and external source.
@njit
def sweep(
    phi : CornerValues,
    q : np.array,
    xs : CrossSection,
    quad : Quadrature,
    dx : np.array,
    BC : np.array,
    lhs_reflective : bool,
    k=1
):

    M = quad.M
    I = len(dx)
    
    psi = [CornerValues(np.zeros(I), np.zeros(I)) for _ in range(M)]
    phi_new = CornerValues(np.zeros(I), np.zeros(I))

    # for each pair of angles
        # for each cell
            # compute cell angular flux and add to vector
    
    for m in range(int(M/2)):
        m_neg = m
        m_pos = M-(m+1)

        # solve mu<0 angle

        for i in range(I-1, -1, -1):
            # print(f"solving transport equation for mu = {quad.mu[m_neg]} at i = {i}")
            if i == I-1:
                psi_in = BC[m_neg]
            else : 
                psi_in = psi[m_neg].L[i+1]
            psi[m_neg].L[i], psi[m_neg].R[i]  = ldSolve(
                        quad.mu[m_neg], dx[i], psi_in, xs.total[i], 
                        xs.scatter[i], xs.fission[i], xs.multiplicity[i]/k, 
                        q[i], (phi.L[i], phi.R[i]))

        phi_new.L += quad.w[m_neg]*psi[m_neg].L
        phi_new.R += quad.w[m_neg]*psi[m_neg].R

        # solve mu>1 angle
        for i in range(I):
            # print(f"solving transport equation for mu = {quad.mu[m_pos]} at i = {i}")
            if i == 0: 
                if lhs_reflective == True:
                    psi_in = psi[m_neg].L[0]
                else:
                    psi_in = BC[m_pos]
            else:
                psi_in = psi[m_pos].R[i-1]
            psi[m_pos].L[i], psi[m_pos].R[i] = ldSolve(
                quad.mu[m_pos], dx[i], psi_in, xs.total[i], xs.scatter[i], 
                xs.fission[i], xs.multiplicity[i]/k, 
                q[i], (phi.L[i], phi.R[i]))

        phi_new.L += quad.w[m_pos]*psi[m_pos].L
        phi_new.R += quad.w[m_pos]*psi[m_pos].R

    return psi, phi_new