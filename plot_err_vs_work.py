import argparse
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

"""
2018 Lukas Herrmann, SAM, ETH Zurich
"""



def load_computed_data(folder, rand_shift, levels, regularity):
    """
    Loads results from folder given indices of random shift, maximal levels of interest, regularity of GRF
    Input:
        folder : folder name
        rand_shift : indices of random shifts
        levels : maximal levels of interest
        regularity : type=string, possible values are either "smooth" or "rough"
    """
    if regularity == "rough":
        a = "2"
    else:
        a = "4"
    filefct = lambda r, m, s: "data_r%d/logN_a" % r + a + "_m%d_s%d.json" % ( m, s)
    res = []
    for m in levels:
        reps = []
        for r in rand_shift:
            filename = os.path.join(folder, filefct(r, m, 2 ** (m + 2)))
            if not os.path.exists(filename):
                print("file does not exist:", filename)
                continue
            try:
                with open(filename) as f:
                    data = json.load(f)
                    reps.append(float(data["result"]))

            except:
                print("could not open:", filename)
        res.append(reps)

    return res


def preprocess_data(res, tau, chi):
    """
    The results are processed and the output the root mean square error and the computational work
    Input:
        res : raw results still explicitly dependent on random shift
        chi : convergence rate of QMC
        tau : convergence rate of FEM
    """
    # compute mean
    means = [np.mean(r) for r in res]

    refval = means[-1]
    # remove last measurements from plot
    res = res[:-1]

    # compute variance
    err = []
    for rep in res:
        # rep is a list of repetitions
        N = len(rep)
        err.append(np.sqrt(sum([(x - refval) ** 2. for x in rep])))
        if N > 1:
            err[-1] /= np.sqrt((N - 1))
    mvals_plt = np.array(range(1, len(err) + 1))
    err /= abs(refval)  # relative error
    work = 2 ** (tau / chi * mvals_plt)

    return err, work

def plot_values(axes, err, work, do_lambdas, lambdas, do_sigmas, sigmas, folder_idx, do_fit, lmin):
    """
    Create plot of error data points versus work data points
    Input:
        axes : axes instance of fugure
        err : error data points
        work : work data points
        do_lambdas : true if values of correlation lengths denoted by lambda should appear in legend
        lambdas : values of correlation lengths
        do_sigmas : true if values of sigma should appear in legend
        sigmas : parameters that controls variance of GRF
        folder_idx : index of folder used to select the style
        do_fit : True if a least square fit should be produced
        lmin : number of data points that are disregarded from fit; the pre-asymptotic regime
    """
    #colors and styles
    cols = ['0.1', '0.3', '0.5', '0.7', '0.8', '0.9']
    styles = ['o-', '^-', 's-', 'p-', '*-', '>-']
    col = cols[folder_idx]
    style = styles[folder_idx]
    # label of data
    lstr = ""
    if do_lambdas:
        lstr = "$\lambda =$" + str(lambdas[folder_idx])
    if do_sigmas:
        lstr = "$\sigma_0 =$" + str(sigmas[folder_idx])
    if do_lambdas and do_sigmas:
        lstr = "$\lambda =$" + str(lambdas[folder_idx]) + ", $\sigma_0 =$" + str(sigmas[folder_idx])

    axes.loglog(work, err, style, label=lstr, markersize=9, color=col)

    ## fit
    if do_fit:
        work_fit = work[args.lmin:]
        err_fit = err[args.lmin:]

        p = np.polyfit(np.log(work_fit), np.log(err_fit), 1)

        ## plot fit
        work = work[lmin_plt:]

        y = np.exp(np.polyval(p, np.log(work)))
        axes.loglog(work, y, '--', label="fit: $%1.3f$" % p[0], color="0.7")



def get_labels(folders):
    """
    currently not used
    """
    lambdas = []
    sigmas= []
    for folder in folders:
        lambda_str = folder.split("lambda")[1].split("_", 1)[0]
        lambdas.append(float(lambda_str[0] + "." + lambda_str[1:]))

        sigma_str = folder.split("sigma")[1].split("_", 1)[0]
        sigmas.append(float(sigma_str[0] + "." + sigma_str[1:]))


    return lambdas, sigmas

def get_folders(sigmas, lambdas, regularity, genvec_informed):
    """
    Outputs the folder name for given parameters.
    Input:
        sigmas : parameters that controls variance of GRF
        lambdas : values of correlation lengths
        regularity : type=string, possible values are either "smooth" or "rough"
        genvec_informed : True if generating vector was informed in experiment
    """
    folders = []
    dir_list = os.listdir('./results')
    for lamb, sigma in zip(lambdas, sigmas):
        sigma_str = str(sigma)
        sigma_str = sigma_str.rstrip('0').rstrip('.') if '.' in sigma_str else sigma_str
        sigma_str = sigma_str.replace(".",'')
        lamb_str = str(lamb)
        lamb_str = lamb_str.rstrip('0').rstrip('.') if '.' in lamb_str else lamb_str
        lamb_str = lamb_str.replace(".",'')
        folder_str = "lambda" + lamb_str + "_" + "sigma" + sigma_str
        if regularity == "smooth":
            folder_str = "alpha4_" + folder_str
        if regularity == "rough":
            folder_str = "alpha2_" + folder_str
        if genvec_informed:
            folder_str = folder_str + "_informed_genvecs"
        if folder_str in dir_list:
            folders.append("results/" + folder_str)
        else:
            print("Folder " + folder_str + " could not be found")

    return folders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularity", type=str, help="regularity of the log Gaussian input", default="rough")
    parser.add_argument("--sigmas", nargs=3 , type=float, help="values of sigma0", default=[1.0, 1.0, 1.0])
    parser.add_argument("--lambdas", nargs=3 , type=float, help="values of lambda", default=[0.1, 0.1, 0.1])
    parser.add_argument("--do_sigmas", help="true if sigmas should appear in legend", action='store_true')
    parser.add_argument("--do_lambdas", help="true if lambdas should appear in legend", action='store_true')
    parser.add_argument("--genvec_informed", help="true if genvecs were informed by random input", action='store_true')
    parser.add_argument("--legend_loc", type=str, default="upper right")
    parser.add_argument("--lmin", type=int, help="number of data points not considered in the empirical conv rate fit", default=4)
    parser.add_argument("--mmax", type=int, help="number of data points in the plot", default=12)
    parser.add_argument("--do_fit", nargs=3, type=int, help="1 if legend 0 if not legend", default=[1, 1, 1])
    parser.add_argument("--figfile", type=str, default="convergence_logN.pdf")
    parser.add_argument("--do_show", help="if true plot will be shown during computation", action='store_true')


    args = parser.parse_args()

    #figfile = "convergence_logN.pdf" # plot output file
    R = 20 # number of repetitions


    folders = get_folders(args.sigmas, args.lambdas, args.regularity, args.genvec_informed)

    ## plot settings
    do_savefig = True
    do_fit = [bool(do_fit_comp) for do_fit_comp in args.do_fit]
    labelargs = {"fontsize":15}
    tickargs = {"fontsize":12}

    # data points not considered in fit of convergence rate
    lmin_plt = args.lmin

    tau = 1.0
    chi = 0.5
    if args.regularity == "smooth":
        chi = 0.9
    if args.regularity == "rough":
        chi = 0.65

    mvals = np.r_[2:args.mmax+1]





    folder_idx = 0
    fig = plt.gca()
    for folder in folders:
        # LOAD DATA
        res = load_computed_data(folder, np.r_[0:R], mvals, args.regularity)

        # COMPUTE VALUES
        err, work = preprocess_data(res, tau, chi)

        # PLOT THE VALUES
        plot_values(fig, err, work, args.do_lambdas, args.lambdas, args.do_sigmas, args.sigmas, folder_idx, do_fit[folder_idx], args.lmin)

        folder_idx += 1

    # FINAL SETTINGS OF PLOT
    plt.setp(fig.get_xticklabels(), **tickargs)
    plt.setp(fig.get_yticklabels(), **tickargs)
    plt.legend(loc=args.legend_loc, prop={'size':tickargs["fontsize"]})
    plt.grid(True)
    plt.xlabel("Work", **labelargs)
    plt.ylabel("Relative Error", **labelargs)
    plt.title(r"MLQMC Convergence", **labelargs)
    plt.tight_layout()


    plt.draw()
    if do_savefig:
        print( "saving plot to file:", args.figfile)
        plt.savefig(args.figfile)
    if args.do_show:
        plt.show()




