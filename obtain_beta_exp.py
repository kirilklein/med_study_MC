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



num_pats = 1000
incidence = 0.1
bexp_med_ls = []


with open(join(base_dir, 'data_and_params', 'alpha0_meds.pkl'), 'rb') as f:
    alpha0_dic = pkl.load(f)

with open(join(base_dir, 'data_and_params', 'beta0_exp_meds.pkl'), 'rb') as f:
    beta0_exp_ls = pkl.load(f)


def compute_outcome_prob(X, beta0_exp, beta_exp, exposures,  
        al = np.log(1.25),
        am = np.log(1.5),
        ah = np.log(1.75),
        avh = np.log(2)):
    exponent = beta0_exp + beta_exp*exposures + al*X[:,1] + al*X[:,2]\
        + am*X[:,4] + am*X[:,5]\
            + ah*X[:,7]+ ah*X[:,8]\
                + avh*X[:,9]
    p = 1/(1+np.exp(-exponent))
    return p


fig, ax = pplt.subplots()
for i in range(1):
    setting = string.ascii_lowercase[i]
    X = simulate_austin.simulate_pats(setting, num_patients=num_pats)
    for beta0_exp, j in enumerate(range(len(beta0_exp_ls[:2]))):
        exposures = simulate_austin.simulate_exposure(X, alpha0_dic[setting][j])
        simulate_austin.compute_outcome_prob(X, beta0_exp,  )
        print(exposures.shape)

        