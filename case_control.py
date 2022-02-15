
#%%
# Import
from email.mime import base
import time
import numpy as np
import pandas as pd
from util import vis, stats
from util.basic import timing
from util import simulate as sim
import importlib
from scipy.stats import norm
importlib.reload(sim)
importlib.reload(vis)
import multiprocessing as mp
pd.options.mode.chained_assignment = None
from pathlib import Path
import os
from os.path import join
# Directories
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
results_dir = join(base_dir, 'results')
#%%
# Specify params

population_size = 10000
num_subset = 1000
num_hidden_variables = 0
true_OR = 2
random_state = None
beta = np.array([-2,1])
gamma0=-4.5
gamma_ls=[.01]
gamma = sim.get_gamma(gamma0, true_OR, gamma_ls)
num_experiments = 10

#%%
# Simulate exposure


def simulate_population(_):
    """Simulate population with the given beta and gamma"""
    df = sim.simulate_exposure(beta, num_hidden_variables,
         population_size, random_state=None)
    df = sim.simulate_disease(df, gamma)
    df = sim.add_subset_col(df, num_subset, random_state=None)
    # df = matching.match(df)
    OR_subs, OR_95CI_subs, OR_pval_subs = stats.compute_OR_CI_pval(df[df.subset==1], 
                                               print_=False)
    OR_all, OR_95CI_all, OR_pval_all = stats.compute_OR_CI_pval(df, 
                                                print_=False)                                                
    width_95CI_subs = OR_95CI_subs[1]-OR_95CI_subs[0]
    #width_95CI_all = OR_95CI_all[1]-OR_95CI_all[0]
    z_val = stats.OR_z_test(df)
    if OR_95CI_subs[0]<= 1 and OR_95CI_subs[1]>=1:
        includes_1 = 1
    else: includes_1 = 0
    return OR_all, round(z_val,5), includes_1, round(width_95CI_subs,5), OR_pval_subs


@timing
def main(num_experiments, num_procs=8):
    print("p_exposure = sigmoid(beta0 + x0*beta1+...)")
    print("t: exposure")
    print("p_disease = sigmoid(gamma0 + gamma1*t + x0*gamma2+...)")
    print("ORs from different matching methods applied to a subset of the population are compared to the true OR (whole population)")
    print("We measure bias by looking at what fraction of the experiments includes an OR of 1 in 95% CI, given that OR!=1")
    with mp.Pool(num_procs) as pool:
        metrics_ls = pool.map(simulate_population, np.arange(num_experiments))
    with open(join(results_dir, 'case_control_OR1_5.txt'), 'w') as f:
        f.write("OR_all z includes_1 width_CI_subs pval_subs\n")
        for t in metrics_ls:
            f.write(' '.join(str(s) for s in t) + '\n')

if __name__ == '__main__':
    main(100, 10)


