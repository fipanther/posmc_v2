"""
    ____                __  _________
    / __ \____  _____   /  \/  / ____/
    / /_/ / __ \/ ___/  / /\_/ / /
    / ____/ /_/ (__  )  / /  / / /___
    /_/    \____/____/  /_/  /_/\____/
    
    Main Science Module
    Fiona H. Panther, Australian National University
    v. 1.2
"""
from __future__ import division, print_function

import sys
import numpy as np
import math
from scipy.integrate import quad
import random
import os

c_dir = os.getcwd()
#os.getpid()
sys.path.append(c_dir)

import par_init as pr
import load as ld
import timestep as ts
import user_functions as usr

def do_science(energy,count, qu, hyd):
    """
        do_science(energy,count,qu,hyd)
        Carries out Monte Carlo simulation for positron annihilation in gas phase (no dust)
    """
    #   need to set this to generalize and import arrays from a txt or dat file
    e_packet = energy
    ext_index = 0
    #   Set up the arrays to store the ISM parameters during the simulation
    arr_energy=[]
    #arr_ne = []
    arr_nH = []
    arr_nHe = []
    arr_xH = []
    arr_xHe = []
    arr_xHe2 = []
    arr_temperature = []
    arr_B = []
    timing_sim = []
    arr_rho = []
    
    arr_nHsim = [hyd]*100000
    
    #   Initialize the arrays
    pr.initialize_MCmp([e_packet, arr_nHsim[0], pr.arr_nHesim[0], pr.arr_xHsim[0], pr.arr_xHesim[0], pr.arr_xHe2sim[0], pr.arr_temperaturesim[0], pr.arr_Bsim[0], pr.arr_rhosim[0]], arr_energy, arr_nH, arr_nHe, arr_xH, arr_xHe, arr_xHe2, arr_temperature, arr_B, timing_sim, arr_rho)
    dEad_init = ld.de_ad(e_packet, 0, pr.model_parameters[0], arr_rho[0], arr_rho[0], pr.model_parameters[1], mass = 511E3)
    print('simulation initialized, particle '+str(count)+','+str(dEad_init))
    
    
    #   Begin the while-loop
    while e_packet>0:
    #   Get the ISM parameters now:
        X = pr.X_ism(arr_energy, arr_nH, arr_nHe, arr_xH, arr_xHe, arr_xHe2, arr_temperature, arr_B, arr_rho)
    #   Calculate the continuous energy loss rate:
    #   add adiabatic losses:
        if len(arr_rho)==1:
            de_conts = pr.eloss_rate(X[0], X[1], X[2], X[3], X[4], X[5], X[7], X[6])[5]+dEad_init
            print(de_conts)
        else:
            de_conts = pr.eloss_rate(X[0], X[1], X[2], X[3], X[4], X[5], X[7], X[6])[5]+ld.de_ad(X[0], timing_sim[len(timing_sim)-1], pr.model_parameters[0], X[8], arr_rho[len(arr_rho)-2], pr.model_parameters[1], mass = 511E3)
            print(de_conts)

        if de_conts<0:
            print('writing to file: thermalization particle '+str(count))
            qu.put(str(count)+','+str(hyd)+', thermalization,'+str(X[0])+';\n')
            break

    #   Calculate the MC timestep
        dt = ts.dt_MC(de_conts, X[0], 0.1, 0.01, X[1], X[2], X[3], X[4], X[5], X[7], X[6], len(arr_nH)-1, ext_index, timing_sim)
        

    #   Calculate the energy change due to the continiuous energy losses
        del_e = X[0] - (X[0] - (de_conts*dt))
    #   Calculate the energy at the end of the timestep if only continuous losses take place
        e_low = X[0] - del_e
        

    #   Work around for big jump in cross-section around 700eV, or else everything just annihilates there
        if pr.norm_xsec(e_low, X[1], X[2], X[3], X[4], X[5])>10*pr.norm_xsec(X[0], X[1], X[2], X[3], X[4], X[5]):
            sigma = pr.norm_xsec(X[0], X[1], X[2], X[3], X[4], X[5])
        else:
            sigma = (pr.norm_xsec(X[0], X[1], X[2], X[3], X[4], X[5])+pr.norm_xsec(e_low, X[1], X[2], X[3], X[4], X[5]))/2

    #   4/8/16 Removed the interpolation function in favor of averaging the cross-sections and velocity. This speeds up the algorithm and steps around the bug in interp1d.
        vel = (usr.rel_bet(X[0], ld.mp)+usr.rel_bet(e_low, ld.mp))/2

    #   Evaluate whether a discrete interaction will occur before e_low (according to Prantzos+2011, eqn 16)
    #   include adiabatic losses here
        rtemp = quad(lambda x:  vel*sigma/(pr.eloss_rate(x, X[1], X[2], X[3], X[4], X[5], X[7], X[6])[5]+ld.de_ad(X[0], timing_sim[len(timing_sim)-1], pr.model_parameters[0], X[8], arr_rho[len(arr_rho)-2], pr.model_parameters[1], mass = 511E3)), e_low, X[0])
        
        rtest = 1-math.exp(-ld.c*rtemp[0])
        
    #   Generate a random number
        ran = random.uniform(0,1)
        
    #   The monte carlo step:
        if ran < rtest:
            etemp = np.linspace(e_low, X[0], 10)
            p = []
            for i in etemp:
                b = quad(lambda x: usr.rel_bet(i, ld.mp)*pr.norm_xsec(i, X[1], X[2], X[3], X[4], X[5])/pr.eloss_rate(i, X[1], X[2], X[3], X[4], X[5], X[7], X[6])[5], i, X[0])
                p.append(1-math.exp(-ld.c*b[0]))
            
            q = usr.find_nearest(np.asarray(p), ran)
            
            eint = etemp[q[0]]

            
            p1 = (X[1]+2*X[2])*pr.d_xsec(eint)/pr.norm_xsec(eint, X[1], X[2], X[3], X[4], X[5])
            
            p2 = p1+((X[1]*(1-X[3])*pr.Hxsec(eint, X[1])[0])/pr.norm_xsec(eint, X[1], X[2], X[3], X[4], X[5]))
            
            p3 = p2+((X[1]*(1-X[3])*pr.Hxsec(eint, X[1])[1])/pr.norm_xsec(eint, X[1], X[2], X[3], X[4], X[5]))
            
            p4 = p3+((X[1]*(1-X[3])*pr.Hxsec(eint, X[1])[2])/pr.norm_xsec(eint, X[1], X[2], X[3], X[4], X[5]))
            
            p5 = p4 + ((X[2]*(1-X[4]-X[5])*pr.Hexsec(eint, X[2])[0])/pr.norm_xsec(eint, X[1], X[2], X[3], X[4], X[5]))
            
            p6 =p5 + ((X[2]*(1-X[4]-X[5])*pr.Hexsec(eint, X[2])[1])/pr.norm_xsec(eint, X[1], X[2], X[3], X[4], X[5]))
            
            p7 = p6 + ((X[2]*(1-X[4]-X[5])*pr.Hexsec(eint, X[2])[2])/pr.norm_xsec(eint, X[1], X[2], X[3], X[4], X[5]))
            #   Generate a new random number to decide on the kind of interaction that took place
            s = random.uniform(0,1)
            #   There should probably be some <= or >= here to catch edge cases! Put them in but if it fucks up, this is why!
            if s<=p1:
                elost = 0
                dtint = (X[0]-eint)/de_conts
                e_con = de_conts*dtint
                enew = X[0]-e_con
                print('writing to file: direct annihilation particle '+str(count))
                qu.put(str(count)+','+str(hyd)+',direct,'+str(enew)+';\n')
                break
            elif p1<s<=p2 and eint>20:   #   shouldn't occur if E<20, but this catches it with an error in case!
                elost = 13.7/4  #mean consistent with Guessoum2005 28/4/16
                    #print('ionization H')
            elif p2<s<=p3 and eint>10.2:#   shouldn't occur if E<10.2, but this catches it with an error in case!
            #   energy losses are equal to excitation of H
                elost = 10.2
            #print('excitation H')
            elif p3<s<=p4 and eint>6.8:#   shouldn't occur if E<6.8, but this catches it with an error in case! ad nauseum
            #   energy losses are equal to charge xchange on H
                elost = 6.8 #and positron annihilates. Bin.
                dtint = (X[0]-eint)/de_conts
                e_con = de_conts*dtint
                enew = X[0]-elost-e_con
                print('writing to file: charge exchange particle '+str(count))
                qu.put(str(count)+','+str(hyd)+',cx,'+str(enew)+';\n')
                break
            elif p4<s<=p5 and eint>49:
            #   energy losses are equal to ionization of He
                elost = 49
            elif p5<s<=p6 and eint>20.6:
            #   energy losses are equal to excitation of He
                elost = 20.6
            #print('exitation He')
            elif p6<s<=p7 and eint>13.6:
            #     energy losses are equal to charge exchange on He
                elost = 13.6    #and positron annihilates. Bin.
                #print('CX He')
                break




    #   Calculate the time elapsed up to the interaction
            dtint = (X[0]-eint)/de_conts
            #print('dtint', dtint)
    #   Calculate the continous energy losses that occured up to this time
            e_con = de_conts*dtint
    #   Calculate the new packet energy
            enew = X[0]-elost-e_con
            
    #   Update the packet energy include adiabatic losses here
            if enew < ld.kbev*X[7] or enew < 6.8 or pr.eloss_rate(enew, X[1], X[2], X[3], X[4], X[5], X[7], X[6])[5]<0:
                # This is the thermalization condition.
                print('writing to file: thermalization2 particle '+str(count))
                qu.put(str(count)+','+str(hyd)+',thermalization,'+str(enew)+';\n')
                break
            else:
                # If the positron doesn't thermalize, update the positron energy
                pr.update_epacket(X[0], elost+e_con, arr_energy)

    #   Calculate the time at the end of the timestep taken
            tnew = timing_sim[len(timing_sim)-1]+dtint
            #print('tnew',tnew)
            
    #   Catch exiting particles HERE
            if tnew >= max(pr.timing_ext):
                print('writing to file: particle exiting box 1,'+str(count)+','+str(tnew))
                qu.put(str(count)+','+str(tnew)+',exit_box,'+str(enew)+';\n')
                break

        
    #   Update all the ISM parameters including rho
            pr.update_params(dtint, len(timing_sim)-1, ext_index+1, arr_nH, arr_nHe, arr_xH, arr_xHe, arr_xHe2, arr_temperature, arr_B, arr_rho, timing_sim, pr.timing_ext, arr_nHsim)
            
    #   Update the time
            pr.update_timing(dtint, timing_sim[len(timing_sim)-1], timing_sim)
            
    #   Roll the timing index forward if necessary
            if timing_sim[len(timing_sim)-1] == pr.timing_ext[ext_index+1]:
                ext_index += 1
    
    #   Or else there is no interaction
        else:
            
    #   Update the packet energy include adiabatic losses here
            if X[0] - del_e < ld.kbev*X[7] or X[0] - del_e < 6.8 or pr.eloss_rate(X[0]-del_e, X[1], X[2], X[3], X[4], X[5], X[7], X[6])[5] < 0:
                print('writing to file: thermalization3 particle '+str(count))
                qu.put(str(count)+','+str(hyd)+',thermalization,'+str(X[0]-del_e)+';\n')
                break
                    
            else:
                pr.update_epacket(X[0], del_e, arr_energy)
            
    #   Catch exiting particles HERE
            if timing_sim[len(timing_sim)-1]+dt >= max(pr.timing_ext):
                print('writing to file: particle exiting box 2,'+str(count)+','+str(timing_sim[len(timing_sim)-1]+dt))
                qu.put(str(count)+','+str(timing_sim[len(timing_sim)-1]+dt)+',exit_box,'+str(X[0] - del_e)+';\n')
                break
            
            
    #   Update the parameters include rho here
            pr.update_params(dt, len(timing_sim)-1, ext_index+1, arr_nH, arr_nHe, arr_xH, arr_xHe, arr_xHe2, arr_temperature, arr_B, arr_rho, timing_sim, pr.timing_ext, arr_nHsim)
            
            
    #   Update the time
            pr.update_timing(dt, timing_sim[len(timing_sim)-1], timing_sim)
            
    #   Roll the timing index forward if necessary
            if timing_sim[len(timing_sim)-1] == pr.timing_ext[ext_index+1]:
                ext_index += 1
    

    return


