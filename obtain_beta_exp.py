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
import string
from util import simulate_austin
import pickle as pkl
from multiprocessing import Pool


with open(join(base_dir, 'data_and_params', 'alpha0_dic.pkl'), 'rb') as f:
    alpha0_dic = pkl.load(f)
with open(join(base_dir, 'data_and_params', 'beta0_exp_ls.pkl'), 'rb') as f:
    beta0_exp_ls = pkl.load(f)

def get_beta_exp_spec(input):
    """Helper function for multiprocessing"""
    alpha0, beta0_exp, setting = input
    beta_exp = simulate_austin.get_beta_exp(alpha0, beta0_exp, 
            setting, desired_gamma, num_vars, num_pats, iters)
    return beta_exp

def construct_input():
    input = []
    for i, (k, value) in enumerate(alpha0_dic.items()):
        beta0_exp = beta0_exp_ls[i]
        for v in value:
            input.append((v, beta0_exp, k))
    return input

desired_gamma = 0.02
iters=1000
num_pats = 1000
num_vars=10
settings = [string.ascii_lowercase[i] for i in range(6)]
beta_exp_dic = {}
if __name__ == '__main__':
    with Pool() as p:
        beta_exp_ls_all = p.map(get_beta_exp_spec, construct_input())
    for i, setting in enumerate(settings):
        len_prevalence = len(alpha0_dic['a'])
        beta_exp_dic[setting] =  beta_exp_ls_all[i*len_prevalence:(i+1)*len_prevalence]
    with open(join(base_dir, 'data_and_params', 'beta_exp_dic.pkl'), 'wb') as f:
        pkl.dump(beta_exp_dic, f)     
        