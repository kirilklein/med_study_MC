import numpy as np
import proplot as pplt
import pickle as pkl
from os.path import join
import os
from pathlib import Path
import string
from util import simulate_austin as sa
file_path = os.path.realpath(__file__)
base_dir = Path(file_path).parent



with open(join(base_dir, 'data_and_params', 'alpha0_meds.pkl'), 'rb') as f:
    alpha0_dic = pkl.load(f) 
with open(join(base_dir, 'data_and_params', 'beta0_exp_meds.pkl'), 'rb') as f:
    beta0_exp_ls = pkl.load(f) 

num_iters = 10
num_pats = 100
beta0_test = 0.1 # This still has to be calculated

for i in range(2):
    setting = string.ascii_lowercase[i]
    alpha0_ls = alpha0_dic[setting] 
    beta0_exp = beta0_exp_ls[i]
    for j, prevalence in enumerate(np.logspace(0, 3, 4, base=2)/100):
        for k in range(num_iters):
            X = sa.simulate_pats(setting, num_patients=num_pats)
            exposures = sa.simulate_exposure(X, a0=alpha0_ls[j])
            outcomes = sa.simulate_outcome(X, beta0_exp, beta0_test, exposures)

            exposed = X[exposures==1]
            unexposed = X[exposed==0]
            # TODO: continue here


