import argparse
from quadrature import *
from integrands import *
from helpers import *
import os
import numpy as np
import scipy.sparse as sparse
from mpi4py import MPI


"""
2018 Lukas Herrmann, SAM, ETH Zurich

Main script that realizes MLQMC-FEM with
PROD QMC weights parallized by mpi4py
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Lmax", type=int, help="maximal level of prewavelet expansion")
    parser.add_argument("--rep", type=int, help="number of random shift")
    parser.add_argument("--alpha", type=int, help="alpha related to smoothness of GRF", default=2.)
    parser.add_argument("--chi", type=float, help="QMC convergence rate", default=0.65)
    parser.add_argument("--tau", type=float, help="FEM convergence rate", default=1)
    parser.add_argument("--sigma0", type=float, help="standard deviation of GRF", default=1.0)
    parser.add_argument("--lambdaC", type=float, help="correlation length of GRF", default=0.1)
    parser.add_argument("--genvec_informed", help="true if genvecs were informed by random input", action='store_true')

    args = parser.parse_args()

    Lmax = args.Lmax
    rep = args.rep
    alpha = args.alpha
    chi = args.chi
    tau = args.tau
    sigma0 = args.sigma0
    lambdaC = args.lambdaC
    informed = False
    if args.genvec_informed:
        informed = True

    outdir = get_outdir(alpha, lambdaC, sigma0, informed)
    outdir = outdir + "/data_r%d"%rep
    if not os.path.exists(outdir): os.makedirs(outdir)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output = {}
    # load random shift
        
    shiftfile = "random_shifts/shift_%d.csv"%rep
    if os.path.exists(shiftfile):
        shift = loadtxt(shiftfile)
    else:
        print("ERROR: could not load shift from",shiftfile)
        exit(-1)

    shift = comm.bcast(shift, root=0) # broadcast from root
    res = 0.
    Sused = 0

    kappa, theta = compute_GRF_params(alpha, sigma0, lambdaC)


    # load precomputed L matrices (because package is not suppoted on EULER cluster)
    Lmatrices = []
    #loop over the levels
    for k in range(Lmax+1):
        if k <= 1:
            L = load("L_matrices/L_matrix%d.npy"%(k))
            Lmatrices.append( L )
        if k > 1:
            L = sparse.load_npz("L_matrices/L_matrix%d.npz"%(k))
            Lmatrices.append( L )


    for l in range(0,Lmax+1):

        s_leaf = 2**(l+2)


        if l==0:
            # instantiate integrand
            integrand = integrand_single(kappa, theta, alpha, l, Lmatrices, s_leaf, shift)

            # compute QMC sample numbers
            Nqmc = compute_QMC_sample_numbers(l, Lmax, chi, tau)

            latfile_dir = get_dir_genvec(alpha, sigma0, lambdaC, informed)
            latfile = latfile_dir + "/lat-n%ds%dalpha2.100000.json"%(Nqmc,s_leaf)
            if alpha == 4:
                latfile = latfile_dir + "/lat-n%ds%dalpha10.100000.json" % (Nqmc, s_leaf)
            if rank == 0:
                print("loading lattice from:", latfile)
            lat = Lattice(latfile, s_leaf)
            if rank == 0:
                print("starting quadrature loop with", len(lat), "points in", s_leaf, "dimensions")
            # quadrature
            res_level = quadrature_MPI(lat, integrand)
            res += res_level
            if rank == 0:
                print("result:", res_level)
        else:
            s_level = 2**(l+1)
            # compute QMC sample numbers
            Nqmc = compute_QMC_sample_numbers(l, Lmax, chi, tau)

            # instantiate integrand
            integrand = integrand_difference(kappa, theta, alpha, l, Lmatrices, s_leaf, s_level, Sused, shift)


            # load lattice
            latfile_dir = get_dir_genvec(alpha, sigma0, lambdaC, informed)
            latfile = latfile_dir + "/lat-n%ds%dalpha2.100000.json" % (Nqmc, s_leaf)
            if alpha == 4:
                latfile = latfile_dir + "/lat-n%ds%dalpha10.100000.json" % (Nqmc, s_leaf)
            if rank == 0:
                print("loading lattice from:",latfile)

            lat = Lattice(latfile, s_leaf)

            if rank == 0:
                print("starting quadrature loop with",len(lat),"points in",s_leaf,"dimensions")
            # quadrature
            res_level = quadrature_MPI(lat, integrand)
            res += res_level
            if rank == 0:
                print("result:",res_level)

        #count the used dimensions
        Sused += s_leaf

        


        output["N"] = Lmax
        output["latfile"] = latfile
        output["result"] = res
        parameters = {
            "alpha": alpha,
            "sigma0": sigma0,
            "lambdaC": lambdaC
        }
        output["params"] = parameters

        # write output
        st = 'logN'
        outfile = os.path.join(outdir,"%s_a%d_m%d_s%d.json"%(st,alpha,Lmax,s_leaf))
        if rank == 0:
            print("writing output to:",outfile)
            with open(outfile, "w") as f:
                json.dump(output, f)

