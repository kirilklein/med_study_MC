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
            p1 = np.mean(p[exposures==1])
            gammas.append(p1 - p0)  # Different than described in the paper
    return np.mean(gammas)

desired_gamma = 0.02
iters=100
num_pats = 1000
bexp_med_ls = []

fig, ax = pplt.subplots()
for i in range(1):
    setting = string.ascii_lowercase[i]
    X = simulate_austin.simulate_pats(setting, num_patients=num_pats)
    for beta0_exp, j in enumerate(range(len(beta0_exp_ls[:1]))):
        alpha0 = alpha0_dic[setting][j]
        f = lambda y: simulate_risk_difference(y, X, alpha0, beta0_exp, iters) #- desired_gamma
        #res_test = []
        #for s in np.linspace(-10,10,100):
        #    res_test.append(f(s))
        bexp_res = so.bisect(f, a=-10, b=10)
        print(bexp_res, f(bexp_res))
        
#plt.plot(np.linspace(-10,10,100),res_test)
#plt.savefig(join(base_dir, 'figs','test.png'))
        