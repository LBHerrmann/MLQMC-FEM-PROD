import numpy as np
import sys
import os

"""
2018 Lukas Herrmann, SAM, ETH Zurich

Generates and saves random shifts for multilevel QMC sampling by randomly shifted lattice rules
input: 
    Lmax : maximal level to be considered
    rep : number of random shifts
"""

assert(len(sys.argv) == 3)

Lmax = int(sys.argv[2])
s_all = 2**(Lmax+3) - 1
rep = int(sys.argv[1])
outdir = "random_shifts"

if not os.path.exists(outdir):
    os.mkdir(outdir)

for k in range(0, rep):
    shiftfile = os.path.join(outdir, 'shift_%d.csv'%k)
    shift = np.random.random((s_all,))
    np.savetxt(shiftfile, shift)


