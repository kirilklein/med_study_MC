
#%%
# Import
import time
import numpy as np
import pandas as pd
from util import matching as mt
from util import vis
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
#%%
# Simulate exposure

def simulate_population():
    df = sim.simulate_exposure(beta, num_hidden_variables,
         population_size, random_state=None)
    df = sim.simulate_disease(df, gamma)
    df = sim.add_subset_col(df, num_subset, random_state=None)
    OR_nm, CI_nm, pval_nm = mt.compute_OR_CI_pval(df[df.subset==1], 
                                                print_=False)
    logOR_subs, logORSE_subs = mt.compute_logOR_SE(df[df.subset==1])
    
    return df
#
#
#z_val = mt.z_test(logOR_all, logORSE_subs, logORSE_all, logORSE_subs)
