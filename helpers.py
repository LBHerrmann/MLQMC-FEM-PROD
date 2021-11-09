import numpy as np

"""
2018 Lukas Herrmann, SAM, ETH Zurich

Several helper function that are used repeatedly
"""

def compute_GRF_params(alpha, sigma0, lambdaC):
    """
    Computes GRF parameters kappa and theta
    input:
        alpha : twice the order of the differential operator
        sigma0 : parameter that controls variance of GRF
        lambdaC : correlation length of GRF
    """
    kappa = 2 * np.sqrt(alpha - 0.5) / lambdaC
    sigma2 = 1. / kappa ** (2 * alpha)
    for k in range(1, 10000):
        sigma2 += 2. / (4 * np.pi ** 2 * k ** 2 + kappa ** 2) ** alpha

    sigma = np.sqrt(sigma2)
    theta = sigma0 / sigma

    return kappa, theta


def get_next_prime(n):
    """
    Computes in a naive way the smallest prime number
    smaller or equal to n
    input:
        n : positive integer
    """
    if n < 3:
        n = 3
    ## test if n is prime otherwise +=1
    state = True
    while state:
        state1 = True
        for i in range(2, n):
            if n % i == 0:
                state1 = False
        if state1:
            state = False
        n += 1
    n -= 1
    return n

def compute_QMC_sample_numbers(l, Lmax, chi, tau):
    """
    Computes the sample numbers for multivel QMC
    input:
        l : current level
        Lmax : maximal level considered
        chi : convergence rate of QMC
        tau : convergence rate of FEM
    """
    M = 2 ** (-((2 * tau + 1) / (2 * chi + 1)) * (l + 1))
    M *= (l + 1) ** (-1 / (2 * chi + 1))
    N0 = 2 ** ((Lmax + 1) * tau / chi)
    N = get_next_prime(int(np.ceil(N0 * M)))

    return N

def get_outdir(alpha, lambdaC, sigma, informed):
    """
    Outputs the out directory for given parameters
    input:
        alpha : twice the order of the differential operator
        lambdaC : correlation length of GRF
        sigma0 : parameter that controls variance of GRF
        informed : boolean, True if informed generating vectors are used otherwise False
    """
    alpha_str = "alpha" + str(alpha)
    sigma_str = str(sigma)
    sigma_str = sigma_str.rstrip('0').rstrip('.') if '.' in sigma_str else sigma_str
    sigma_str = sigma_str.replace(".", '')
    lambdaC_str = str(lambdaC)
    lambdaC_str = lambdaC_str.rstrip('0').rstrip('.') if '.' in lambdaC_str else lambdaC_str
    lambdaC_str = lambdaC_str.replace(".", '')
    informed_str = ""
    if informed:
        informed_str = "_informed_genvecs"
    return "results_new/" + alpha_str + "_lambda" + lambdaC_str + "_sigma" + sigma_str + informed_str

def get_dir_genvec(alpha, sigma, lambdaC, informed):
    """
    Outputs directory of the generating vector to be used for given parameters
    input:
        alpha : twice the order of the differential operator
        sigma0 : parameter that controls variance of GRF
        lambdaC : correlation length of GRF
        informed : boolean, True if informed generating vectors are used otherwise False
    """
    genvec_dir = ""
    if informed:
        sigma_str = str(sigma)
        sigma_str = sigma_str.rstrip('0').rstrip('.') if '.' in sigma_str else sigma_str
        sigma_str = sigma_str.replace(".", '')
        lambdaC_str = str(lambdaC)
        lambdaC_str = lambdaC_str.rstrip('0').rstrip('.') if '.' in lambdaC_str else lambdaC_str
        lambdaC_str = lambdaC_str.replace(".", '')
        genvec_dir = "genvecs_informed/genvecs_output_sigma" + sigma_str + "_lambda" + lambdaC_str
    elif alpha == 2 and not informed:
        genvec_dir = "genvecs_generic/genvecs_output_generic_0"
    else:
        genvec_dir = "genvecs_generic/genvecs_output_generic_2"

    return genvec_dir


