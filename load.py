"""
    ____                __  _________
    / __ \____  _____   /  \/  / ____/
    / /_/ / __ \/ ___/  / /\_/ / /
    / ____/ /_/ (__  )  / /  / / /___
    /_/    \____/____/  /_/  /_/\____/
    
    Import data
    Fiona H. Panther, Australian National University
    v. 1.2
"""
from __future__ import division, print_function

import sys
import numpy as np
import scipy as sp
import math
from scipy.integrate import quad
import os
import matplotlib.pyplot as plt

c_dir = os.getcwd()
sys.path.append(c_dir)

import user_functions as usr


#   Physical constants required in simulation
kbev = 8.6E-5       # Boltzmann constant in units eV/K
ec = 4.8E-19        # Electron charge in cgs units
hbar = 1.05E-27     # h/2pi in cgs units
kberg = 1.3E-16     # Boltzmann constant in cgs units
me = 9.11E-28       # Electron mass in cgs units
mp = 511E3      # positron rest mass in units eV/c^2
c = 3E10        # speed of light in cgs
r0 = 2.82E-13   # classical electron radius in units cm
a = 0.0072           # fine structure constant
kev = 1E3         # kiloelectronvolts



#   Load in the H-process cross-sections. Sources discussed in Murphy+2005
arr_H_ion = np.loadtxt(str(c_dir)+"/h_process/ion_xsec.txt",skiprows=0) #Jones+1993 and Hoffman+1997
arr_H_ex = np.loadtxt(str(c_dir)+"/h_process/ex_xsec.txt",skiprows=0) #Calc by Kernoghan+1996 and Stein+1998
arr_H_cx = np.loadtxt(str(c_dir)+"/h_process/cx_extrap.txt",skiprows=0) #Zhou+1997

#   Load in the He-process cross-sections. Sources discussed in Murpy+2005, update
arr_He_ion = np.loadtxt(str(c_dir)+"/he_process/heion_xsec.txt",skiprows=0) #Jacobsen+1995
arr_He_ex = np.loadtxt(str(c_dir)+"/he_process/heex_xsec.txt",skiprows=0)   #Mori+Sueoka 1994
arr_He_cx = np.loadtxt(str(c_dir)+"/he_process/hece_xsec.txt",skiprows=0)   #Overton+1993

#   Load in the radiative recombination cross-section (Gould1989)
arr_rad = np.loadtxt(str(c_dir)+"/e_process/radrecomb.txt", skiprows=0)

#def d_xsec(energy):
#    gamma = 1+((energy)/(mp))
#    beta = np.sqrt(1-(1/(1+((energy)/mp)))**2)
#    if energy<75*kev:
#        s = ((np.pi*r0**2)/beta)*(2*np.pi*0.0072/beta)/(1-np.exp(-2*np.pi*0.0072/beta))
#    else:
#        s = (np.pi*r0**2/(4*gamma**2*beta))*(2*(beta**2-2)+((3-beta**4)/beta)*np.log((1+beta)/(1-beta)))
#    
#    return s
#print(len(arr_rad))
#    
#d_free = []
#en = np.linspace(1,5000, 1000)
#for i in en:
#    aa = d_xsec(i)*10**5
#    d_free.append(aa)


#   Move these into a separate file - they're fairly redundant
#
#   if you would like to plot the raw cross-sections for these processes, please uncomment the following
#

##Helium cross-sections - Log-Log plot
#
#plt.figure
#plt.plot(np.log10(arr_He_ion[:,0]),np.log10(arr_He_ion[:,1]), 'r:', label = '$\mathrm{He\,Ionization}$')
#plt.plot(np.log10(arr_He_ex[:,0]), np.log10(arr_He_ex[:,1]), 'r--', label = '$\mathrm{He\,exitation}$')
#plt.plot(np.log10(arr_He_cx[:,0]), np.log10(arr_He_cx[:,1]), 'r-', label = '$\mathrm{He\,charge\,exchange}$')
##plt.title("Helium-positron process cross-sections")
#plt.xlabel("$\mathrm{log(E/eV)}$")
#plt.ylabel("$\mathrm{log{(\sigma/cm^2)}}$")
#plt.legend(loc = 'best', fontsize = '10')
#plt.ylim([-19,-15])
#
##Hydrogen cross-sections - Log-Log plot
#
#
#plt.plot(np.log10(arr_H_ion[:,0]),np.log10(arr_H_ion[:,1]), 'b:' , label = '$\mathrm{H\,Ionization}$')
#plt.plot(np.log10(arr_H_ex[:,0]), np.log10(arr_H_ex[:,1]), 'b--', label = '$\mathrm{H\,exitation}$')
#plt.plot(np.log10(arr_H_cx[:,0]), np.log10(arr_H_cx[:,1]), 'b-', label = '$\mathrm{H\,charge\,exchange}$')
#plt.plot(np.log10(en), np.log10(d_free), 'c-', label = '$\mathrm{Direct\,annihilation\,(x10}^5)$')
#plt.title("$\mathrm{Positron\,cross-sections}$")
##plt.xlabel("log(E/eV)")
##plt.ylabel("log($\sigma$/$\mathrm{cm}^2$)")
#plt.legend(loc = 'best', fontsize = '10')
#plt.ylim([-19,-15])
#plt.xlim([0,3.5])
#plt.show()



#   Load the rates at thermalization as a function of temperature

arr_df = np.loadtxt(str(c_dir)+"/rates_data/rate_dfree.txt",skiprows=0) #Crannell et al
arr_hcx = np.loadtxt(str(c_dir)+"/rates_data/rate_hcx.txt",skiprows=0)  #Computed from x-sec above
arr_hdb = np.loadtxt(str(c_dir)+"/rates_data/rate_hdab.txt",skiprows=0) #Murphy+2005
arr_hedb = np.loadtxt(str(c_dir)+"/rates_data/rate_hedab.txt",skiprows=0)   #Murphy+2005
arr_rr = np.loadtxt(str(c_dir)+"/rates_data/rate_rr.txt",skiprows=0)    #Ralph Sutherland, private communication
arr_hecx = np.loadtxt(str(c_dir)+"/rates_data/rate_hece.txt",skiprows=0)    #Murphy+2005



#   Move these to a new file
##   Normalized rates: Log-Log plot
#
#plt.figure
#plt.plot(np.log10(arr_df[:,1]),np.log10(arr_df[:,0]), 'r-' ,lw = 2, label = 'Direct free')
#plt.plot(np.log10(arr_hcx[:,0]), np.log10(arr_hcx[:,1,]), 'b--',lw = 2, label = 'Charge exchange w/ H')
#plt.plot(np.log10(arr_hdb[:,0]), np.log10(arr_hdb[:,1]), 'c:', lw = 2, label = 'Direct bound H')
#plt.plot(np.log10(arr_hedb[:,0]), np.log10(arr_hedb[:,1]), 'y-', lw = 2, label = 'Direct bound He')
#plt.plot(np.log10(arr_rr[:,0]), np.log10(arr_rr[:,1]), 'g-', lw = 2, label = 'Radiative Recombination')
#plt.plot(np.log10(arr_hecx[:,0]), np.log10(arr_hecx[:,1]), 'y--', lw = 2, label = 'CX on He')
#plt.title("reaction rates/target density")
#plt.xlabel("log(T/K)")
#plt.ylabel("log($\mathrm{cm}^3$/$\mathrm{s}$)")
#plt.legend(loc = 'best')
#plt.xlim([2,8])
#plt.ylim([-19,-5])
#plt.show()

#
#   initialize the formulae to calculate the energy loss processes
#
#   dE_io - ionization continuum losses
#   dE_pl - plasma losses
#   dE_sy - synchrotron losses
#   dE_br - bremsstrahlung losses
#   dE_ic - inverse Compton losses
#

#   ionization continuum losses (ref. Prantzos et al. 2011)
def dE_io(energy, n_neu, Z, e_ion):
    """
    Ionization continuum losses due to interactions with neutral atoms with density n_neu of species Z with ionization energy e_ion
    """
    return (7.7E-9)*(n_neu*Z/usr.rel_bet(energy, mp))*(np.log((usr.rel_gam(energy, mp)-1)*((usr.rel_gam(energy, mp)*usr.rel_bet(energy, mp)*mp)**2)/(2*e_ion**2))+1/8)

def dE_pl(energy, ne_free, temperature):
    """
    Plasma continuum losses due to e+ interactions with free electrons with density ne_free in plasma of T = temperature
    """
    if ne_free == 0:
        """
        Avoid weird NaNs if there are no free electrons
        """
        rt = 0
    elif energy>=1E3 or temperature >= 1E4:
        """
        For high energies and temperatures, just use the approximation from Prantzos+2011
        """
        rt  = ((7.7/1.45)*10**-9)*(ne_free/usr.rel_bet(energy, mp))*(np.log(usr.rel_bet(energy, mp)/ne_free)+73.6)
    elif energy<1E3 or temperature < 1E4:
        """
        For low energies and temperatures, use the full plasma losses formula from Huba 2004 (NRL Plasma Formulary)
        """
        u = np.abs(np.sqrt(2*energy/mp)-np.sqrt((8*kbev*temperature)/(np.pi*mp)))
        rmax = np.sqrt(kberg*temperature/(4*np.pi*ne_free*(ec**2)))
        if (ec**2/(me*u**2))>hbar/(2*me*u):
            rmin =(ec**2/(me*u**2))
        else:
            rmin =hbar/(2*me*u)
        cl = np.log(rmax/rmin)
        funcint = lambda x: ((x**(1/2))*math.exp(-x))-(((energy/(kbev*temperature))**(1/2))*math.exp(-energy/kbev*temperature))
        integ = quad(funcint,0,(energy/(kbev*temperature)))
        rt =((1.7*10**-8))*(ne_free/usr.rel_bet(energy, mp))*cl*integ[0]
    return rt


def dE_sy(energy, B):
    """
    Synchrotron losses (ref Blumenthal & Gould 1970) - B is specified in microGauss, averaged over pitch angle
    """
    return (9.9E-16)*B**2*usr.rel_gam(energy, mp)**2*usr.rel_bet(energy, mp)**2*(2/3)


def dE_bri(energy, n_ion, Z):
    """
    Bremsstrahlung energy losses from a fully ionized gas of species Z with density n_ion
    """
    return (3.6E-11)*Z*(Z+1)*n_ion*usr.rel_gam(energy, mp)*(np.log(2*usr.rel_gam(energy, mp))-0.33)

def dE_brn(energy, nH, nHe, xH, xHe, xHe2):
    """
    Bremsstrahlung energy losses from a neutral gas with total H density nH and total He density nHe with ionization fractions xH, xHe and xHe2
    """
    return ((4.1E-10)*(1-xH)*nH*usr.rel_gam(energy,mp))+((1.1E-9)*(1-xHe-xHe2)*nHe*usr.rel_gam(energy, mp))

def dE_ic(energy, urad):
    """
    Inverse Compton energy losses in radiation field with density urad - use the CMB field as a default. Blumenthal+Gould 1970
    """
    return (2.6E-14)*urad*(usr.rel_gam(energy, mp)**2)*(usr.rel_bet(energy, mp)**2)






