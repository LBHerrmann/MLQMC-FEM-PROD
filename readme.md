
# Description

Python source code that computes expectations of linear functionals of solutions to elliptic partial differential equations with log-Gaussian coefficients by the multilevel quasi-Monte Carlo finite element method (MLQMC-FEM). 
The log-Gaussian coefficients are represented in pre-wavelet expansions and QMC by randomly shifted lattice rules is used.


# Publication


Lukas Herrmann, Christoph Schwab. <br /> 
Multilevel quasi-Monte Carlo integration with product weights for elliptic PDEs with lognormal coefficients <br />
ESAIM: Mathematical Modelling and Numerical Analysis, 53/5, pp. 1507-1552, 2019 <br />
[DOI](https://doi.org/10.1051/m2an/2019016)

[accepted version of the preprint](https://www.sam.math.ethz.ch/sam_reports/counter/ct.php?file=/sam_reports/reports_final/reports2017/2017-19_rev2.pdf)



Please consult Section 7 of this paper for details on the numerical experiments. 
In brief, the code realizes a pre-wavelet discretization of a Gaussian random field, which is taken as an input for a univariate, elliptic boundary value problem. A MLQMC-FEM scheme is used to compute the expectation of a functional applied to the solution. The tests shall underpin the theoretical results on QMC by randomly shifted lattice rules with so called product weights, which have a computational cost that is linear with respect to the number of parameters.





When you find this code useful for your own research, please cite the mentioned paper above. 


# Compute results

Install the requirements for example by

```bash
conda create -n mlqmc python=3.6
conda activate mlqmc
conda install -c conda-forge --file requirements.txt 
```


Please execute the following commands from the root directory.

```bash
python source/make_shifts.py 20 13
```

```bash
bash jobs/compute_Lmatrices.sh
```

Start a job for example by setting number of processors 

```bash
NP=1
```

Note each jobs was executed on a computing cluster and was running for several hours with 720 CPUs. Execute a job for example by

```bash
bash jobs/alpha2_lambda01_sigma1.sh $NP
```


# Visualize results


To create the plots that appear in the paper with the provided results run the commands

```bash
bash jobs/compute_Lmatrices.sh

bash jobs/create_error_vs_work_plots.sh
```

To create the plots for the visualizations of the representation system of the Gaussian random field input for the localization run the command 

```bash
python source/visualize_localization.py
```

To create the plot on the decay of the representation system run the commands

```bash
Lmax=11

python source/compute_decay.py $Lmax 1.0 0.1
python source/compute_decay.py $Lmax 1.0 0.05
python source/compute_decay.py $Lmax 1.0 0.01

python source/visualize_decay.py $Lmax
```


 
