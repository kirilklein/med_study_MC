"""Compute beta_exp as described in 
Austin, Peter C. (2010).
 A Data-Generation Process for Data with Specified Risk Differences 
    or Numbers Needed to Treat. 
    Communications in Statistics - Simulation and Computation, 39(3), 563–577."""
import os
from pathlib import Path
from os.path import join
file_path = os.path.realpath(__file__)
base_dir = Path(file_path).parent
import numpy as np
import proplot as pplt
import string
from util import simulate_austin
import pickle as pkl
import scipy.optimize as so
import matplotlib.pyplot as plt


with open(join(base_dir, 'data_and_params', 'alpha0_meds.pkl'), 'rb') as f:
    alpha0_dic = pkl.load(f)

with open(join(base_dir, 'data_and_params', 'beta0_exp_meds.pkl'), 'rb') as f:
    beta0_exp_ls = pkl.load(f)

def simulate_risk_difference(beta_exp, X, alpha0, beta0_exp, iters=10):
    gammas = []
    for _ in range(iters):
            exposures = simulate_austin.simulate_exposure(X, alpha0)
            p = simulate_austin.compute_outcome_prob(X, beta0_exp, 
                beta_exp, exposures)
            p0 = np.mean(p[exposures==0])
            if (exposures==1).sum()==0:
                p1 = 0
            else:
                p1 = np.mean(p[exposures==1])
            gammas.append(p1 - p0)  # Different than described in the paper (p0-p1)
    return np.mean(gammas)

desired_gamma = 0.02
iters=1000
num_pats = 1000
beta_exp_dic = {}


def get_beta_exp_spec(input):
    """Helper function for multiprocessing"""
    setting, prevalence = input
    a0 = simulate_austin.get_alpha0(setting, prevalence, 
                num_vars=num_vars, num_pats=num_pats, 
                iters=iters, a=a, b=b)
    return a0
def get_beta_exp(alpha0, beta0_exp, iters, ):

if __name__ == '__main__':
    for i in range(6):
        bexp_ls = []
        beta0_exp = beta0_exp_ls[i]
        alpha0_ls = alpha0_dic[i]
        setting = string.ascii_lowercase[i]
        X = simulate_austin.simulate_pats(setting, num_patients=num_pats)
        for alpha0 in alpha0_ls:
            f = lambda y: simulate_risk_difference(y, X, alpha0, beta0_exp, iters) - desired_gamma
            bexp_res = so.bisect(f, a=-2, b=0)
            bexp_ls.append(bexp_res)
            print(bexp_res, f(bexp_res))
        beta_exp_dic[setting] = bexp_ls
    with open(join(base_dir, 'data_and_params', 'beta_exp_dic.pkl'), 'wb') as f:
        pkl.dump(beta_exp_dic, f)     
        