import os
from pathlib import Path
from os.path import join
file_path = os.path.realpath(__file__)
base_dir = Path(file_path).parent
import string
from util import simulate_austin
from util import metrics
import pickle as pkl
from multiprocessing import Pool
import numpy as np

with open(join(base_dir, 'data_and_params', 'alpha0_dic.pkl'), 'rb') as f:
    alpha0_dic = pkl.load(f)
with open(join(base_dir, 'data_and_params', 'beta0_exp_ls.pkl'), 'rb') as f:
    beta0_exp_ls = pkl.load(f)
with open(join(base_dir, 'data_and_params', 'beta_exp_dic.pkl'), 'rb') as f:
    beta_exp_dic = pkl.load(f)


num_pats = 1000
iters = 800
std_diffs = []
mean_diffs = []
for i, (setting, values) in enumerate(alpha0_dic.items()):
        if i==1:
            break
        i+=1
        beta0_exp = beta0_exp_ls[i]
        beta_exp_ls = beta_exp_dic[setting]
        for i, alpha0 in enumerate(values[:1]):
            beta_exp = beta_exp_ls[i]
            for _ in range(iters):
                X, exp, out = simulate_austin.simulate_pats_exp_out(setting, alpha0, 
                    beta0_exp, beta_exp, num_pats=num_pats)
                std_diffs.append(metrics.stand_diff_bin(exp, out))
                mean_diffs.append(metrics.diff_in_means(exp, out))
print(np.mean(mean_diffs))   
                

            