import numpy as np

def calculateBalanceResidual(
        J_left : float,
        J_right : float,
        phi_LHS : float,
        phi_RHS : float,
        sigma_t : float,
        sigma_s : float,
        sigma_f : float,
        nu_f : float,
        q : float,
        dx : float
):
    streaming = J_right - J_left
    absorption = sigma_t*phi_LHS*dx
    scatter = (sigma_s + nu_f*sigma_f)*dx*phi_RHS
    source = dx*q
    return streaming + absorption - scatter - source

def calculateFirstMomentResidual(
        J_left : float,
        J_right : float,
        J_avg : float,
        phi_slope_LHS : float,
        phi_slope_RHS : float,
        sigma_t : float,
        sigma_s : float,
        sigma_f : float,
        nu_f : float,
        q : float,
        dx : float     
):
    streaming = 3*(J_right + J_left - 2*J_avg)
    absorption = sigma_t*phi_slope_LHS*dx
    scatter = (sigma_s + nu_f*sigma_f)*dx*phi_slope_RHS
    source = 0 # assumes constant source in each cell
    return streaming + absorption - scatter - source
