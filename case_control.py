
#%%
# Import
import time
import numpy as np
import pandas as pd
from util import matching as mt
from util import vis, stats
from util import simulate as sim

import importlib
from scipy.stats import norm
importlib.reload(sim)
importlib.reload(vis)
import multiprocessing as mp
pd.options.mode.chained_assignment = None
from numpy.random import default_rng

#%%
# Specify params
population_size = 10000
num_subset = 1000
num_hidden_variables = 0
true_OR = 1.5
random_state = None
beta = np.array([-2,1])
gamma0=-4.5
gamma_ls=[.01]
gamma = sim.get_gamma(gamma0, true_OR, gamma_ls)
print("p_exposure = sigmoid(beta0 + x0*beta1+...)")
print("t: exposure")
print("p_disease = sigmoid(gamma0 + gamma1*t + x0*gamma2+...)")
print("ORs from different matching methods applied to a subset of the population \
    are compared to the true OR (whole population)")
print("We measure bias by looking at what fraction of the experiments \
    includes an OR of 1 in 95% CI, given that OR!=1")
#%%
# Simulate exposure

def simulate_population():
    """Simulate population with the given beta and gamma"""
    df = sim.simulate_exposure(beta, num_hidden_variables,
         population_size, random_state=None)
    df = sim.simulate_disease(df, gamma)
    df = sim.add_subset_col(df, num_subset, random_state=None)
    OR_subs, OR_CI_subs, OR_pval_subs = stats.compute_OR_CI_pval(df[df.subset==1], 
                                                print_=False)
    logOR_subs, logORSE_subs = stats.compute_logOR_SE(df[df.subset==1])
    logOR_all, logORSE_all = stats.compute_logOR_SE(df)
    z_val = stats.z_test(logOR_all, logOR_subs, logORSE_all, logORSE_subs)
    if OR_CI_subs[0]<= 1 and OR_CI_subs[1]>=1:
        includes_1 = 1
    else: includes_1 = 0
    return z_val, includes_1
#
def main(num_procs=8):
    with mp.Pool(num_procs) as pool:
        metrics = pool.map(simulate_population)

if __name__ == '__main__':
    main(2)


