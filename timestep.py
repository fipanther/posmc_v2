"""
    ____                __  _________
    / __ \____  _____   /  \/  / ____/
    / /_/ / __ \/ ___/  / /\_/ / /
    / ____/ /_/ (__  )  / /  / / /___
    /_/    \____/____/  /_/  /_/\____/
    
    Calculate adaptive timestep
    Fiona H. Panther, Australian National University
    v. 1.2
"""
from __future__ import division, print_function

import sys
import numpy as np
import math
import os

c_dir = os.getcwd()
sys.path.append(c_dir)

import par_init as pr
import load as ld
import user_functions as usr

def dt_MC(lr, energy, p_elossmax, p_intmax, nH_tot, nHe_tot, x_H, x_He, x_He2, temperature, B, idx_now, idx_ext, arraytime):

#   Import the external timestep
    t_now = pr.timing_ext[idx_ext]
    t_next = pr.timing_ext[idx_ext+1]
    dt_now = t_next-t_now
    

#   Calculate the timestep set by the max energy losses
    def dt_en(energy, loss_rate, p_elossmax):
        return p_elossmax*energy/loss_rate
    #lr = pr.eloss_rate(energy, nH_tot, nHe_tot, x_H, x_He, x_He2, B, temperature)[5]
    dt_e = dt_en(energy, lr, p_elossmax)

#   Calculate the timestep set by the max interaction probability
    def denom(energy, nH_tot, nHe_tot, x_H, x_He, x_He2):
        return ld.c*usr.rel_bet(energy, ld.mp)*pr.norm_xsec(energy, nH_tot, nHe_tot, x_H, x_He, x_He2)

    if denom(energy, nH_tot, nHe_tot, x_H, x_He, x_He2) == 0:
        dt_int = 10**20
    else:
        dt_int = p_intmax/(denom(energy, nH_tot, nHe_tot, x_H, x_He, x_He2))


#   Now calculate the Monte Carlo timestep
    if arraytime[len(arraytime)-1]+min(dt_now, dt_e, dt_int)>=t_next:
        dt_MC = t_next - arraytime[len(arraytime)-1]
    else:
        dt_MC = min(dt_now, dt_e, dt_int)

    return dt_MC


