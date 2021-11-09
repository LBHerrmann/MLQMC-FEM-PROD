import numpy as np
import json
from collections import Sequence
from mpi4py import MPI


"""
2018 Lukas Herrmann, SAM, ETH Zurich
"""

def quadrature_MPI(lattice, f, transf = None):
    """

    :param lattice: QMC lattice
    :param f: integrand
    :param transf: transformation of QMC points
    :return: computed integral
    """
    assert(len(lattice)>0)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # determine range of indices
    blocksize = int(len(lattice) / size)
    diff = len(lattice)-size*blocksize
    imin = 0
    for i in range(rank):
        imin += blocksize
        if i < diff:
            imin += 1
    imax = imin+blocksize
    if rank < diff:
        imax += 1
    myrange = range(imin, imax)
    # quadrature loop
    S = 0.
    if transf == None:
        for i in myrange:
            S += f(lattice[i])
    else:
        for i in myrange:
            S += f(transf(lattice[i]))

    # MPI reduce
    sendbuf = np.array([S])
    recvbuf = np.array([0.])
    comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)
    if rank == 0:
        return recvbuf[0]/len(lattice)
    else:
        return 0


class Lattice(Sequence):
    def __init__(self, filename=None, smax=1):
        self.is_loaded = False
        self.smax = smax
        if filename != None: self.load(filename)

    def set_smax(self, smax):
        if self.is_loaded:
            assert(smax <= self.s)
        self.smax = smax

    def load(self, f):
        """Load generating vector from a file"""
        with open(f) as ff:
            data = json.load(ff)
            self._load_helper(data)

    def _load_helper(self,data):
        self.N = int(data["N"])
        self.s = int(data["s"])
        if self.s < self.smax:
            raise Exception("Generating vector not long enough! (max. dim: s=%d)"%self.s)
        self.C = data["C"]
        self.genvec = np.array(data["genvec"],dtype=int)
        self.is_loaded = True

    def __getitem__(self,n):
        """need this for Sequence"""
        if not self.is_loaded:
            raise Exception("Must load a lattice before accessing its points!")
        if n >= self.N:
            raise IndexError("Point set only contains N=%d points."%self.N)
        y = np.array([np.mod(1.*n*self.genvec[j]/self.N,1) for j in range(self.smax)])
        assert(all(y >= 0))
        assert(all(y <= 1))
        return y

    def __len__(self):
        """Number of points in the lattice"""
        # need this for 'Sequence'
        return self.N



