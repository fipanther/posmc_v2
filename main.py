"""
    ____                __  _________
    / __ \____  _____   /  \/  / ____/
    / /_/ / __ \/ ___/  / /\_/ / /
    / ____/ /_/ (__  )  / /  / / /___
    /_/    \____/____/  /_/  /_/\____/
    
    This is the bit where you run the simulation
    Fiona H. Panther, Australian National University
    v. 2.0
"""
from __future__ import print_function, division

import sys
import numpy as np
import math
import multiprocessing as mp
import socket
import os

#   get the current working directory so you know where to import everything from
c_dir = os.getcwd()
sys.path.append(c_dir)

#   import the Pos-MC modules
import par_init as pr
import load as ld
import do_science as ds


#   Serial loop - you really don't want to do this because you will be dead by the time it finishes
#N = 100000
#j = 0
#while j<N:
#    ds.do_science(700)
#    j +=1

hst = socket.gethostname()


def writer(q,host,cdir):
    """
        writer(q, host, cdir)
        
        safely writes simulation outputs to a text file when using multiprocessing
        
        Parameters
        ----------
        q: multiprocessing queue manager
        # make host optional
        host: ensures an individual .txt is created for each node and is identifiable
        #make the directory optional
        cdir: puts the file into the current diretctory
        
        # make the file name user-supplied
        
        Returns:
        ---------
        creates and appends results to a text file named
        ncx_host.txt
    """
    with open(str(cdir)+"/ncx_"+str(host)+".txt",'a') as f:
        while True:
            s = q.get()
            if s == 'stop':
                return
            f.write(s)
            f.flush()
        f.close()

#   create the manager
manager = mp.Manager()

#   create the queue and pool the CPUs - this needs to be user variable
qu = manager.Queue()
pool = mp.Pool(mp.cpu_count()+1)

#   put the writer to work using apply_async as it takes multiple args
wp = pool.apply_async(writer, (qu,hst,c_dir))

#   how many particles do you want your MC to simulate (Note: 10,000 particles for good convergence, just don't do this on your own comp unless you want to die)
n_part = 50

#   here's the grid points for the convergence test
hy = [1, 1]

#   here's the parallel loop!

#   create a place to put your jobs
jobs   = []

#   Squeeze into the job cannon
for j in hy:
    for i in range(n_part):
        #   Fire jobs out of the job cannon
        job = pool.apply_async(ds.do_science, (1E6, i, qu, j))
        jobs.append(job)

#   get the results from the pile of jobs
for job in jobs:
    job.get()

#   when you run out of jobs, make sure the code doesn't hang by shutting off the parallel bit
qu.put('stop')
pool.close()
pool.join()

#   you're finished! let the user know this otherwise they will be sad
print('simulation complete')
