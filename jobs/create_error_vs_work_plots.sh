#!/bin/bash

python source/plot_err_vs_work.py --regularity "rough" --sigmas 0.5 0.5 0.5 --lambdas 0.1 0.05 0.01 --do_lambdas --legend_loc "upper right" --lmin 6 --mmax 12 --figfile "plot1.pdf" --do_show
python source/plot_err_vs_work.py --regularity "rough" --sigmas 1.0 1.0 1.0 --lambdas 0.1 0.05 0.01 --do_lambdas --legend_loc "upper right" --lmin 6 --mmax 12 --figfile "plot2.pdf" --do_show
python source/plot_err_vs_work.py --regularity "smooth" --sigmas 0.1 0.25 0.5 --lambdas 0.1 0.1 0.1 --do_sigmas --legend_loc "lower left" --lmin 4 --mmax 12 --figfile "plot3.pdf" --do_show
python source/plot_err_vs_work.py --regularity "rough" --sigmas 1.0 1.0 1.0 --lambdas 0.1 0.05 0.01 --do_lambdas --genvec_informed --legend_loc "upper right" --lmin 4 --mmax 10 --do_fit 1 1 0 --figfile "plot4.pdf" --do_show
