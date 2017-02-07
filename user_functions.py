"""  ____                __  _________
    / __ \____  _____   /  \/  / ____/
    / /_/ / __ \/ ___/  / /\_/ / /
    / ____/ /_/ (__  )  / /  / / /___
    /_/    \____/____/  /_/  /_/\____/
    
    User-defined functions by Fiona H. Panther
    Australian National University
"""
from __future__ import division, print_function
import numpy as np

def find_nearest(array, value):
    """
        Finds the nearest value to one given in an array
    """
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def rel_gam(energy, mass):
    """
    Relativistic gamma for particle of given mass and kinetic energy
    """
    return 1+(energy/mass)

def rel_bet(energy, mass):
    """ 
    Relativisitc beta for particle of given mass and kinetic energy
    """
    return np.sqrt(1-(1/(1+((energy)/mass)))**2)

def find_first(array, value):
    """
        finds the first instance of a value in an array and returns the index
        """
    for i in range(len(array)):
        if array[i]==value:
            out = i
            break
    return out

def adiabatic_gamma(energy, mass = 511E3):
    #   Thanks to Yuval Birnboim
    """
        Calculates adiabatic index as a function of energy
    """
    gamma_out = 1 + (1/3*(rel_bet(energy, mass)**2))/((1 - np.sqrt(1 - rel_bet(energy, mass)**2)))
    return gamma_out

