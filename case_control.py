
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
importlib.reload(mt)
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

def simulate_exposure_disease():
    df = sim.simulate_exposure(beta, num_hidden_variables,
         population_size, random_state=None)
    df = sim.simulate_disease(df, gamma)
    #dfs = sim.get_positives_and_random_subset(df, num_subset, random_state=None)
    return df
df = simulate_exposure_disease()


def add_subset_col(df, n_subset, random_state=0):
    """Selects all the sick and a random subset of size n_subset of the rest of the population"""
    df['Subset'] = 0
    df.loc[df.disease==1, 'Subset'] = 1
    ndis_indices = df.index[df.disease==0].tolist()
    if type(random_state)==int:
        rng = default_rng(random_state)
        rng.shuffle(ndis_indices)
    else:
        np.random.shuffle(ndis_indices)
    subset_inds = ndis_indices[:n_subset]
    df.loc[subset_inds, 'Subset'] = 1
    return df


#
# OR_nm, CI_nm, pval_nm = mt.compute_OR_CI_pval(dfs, print_=False)
#logOR_subs, logORSE_subs = mt.compute_logOR_SE(dfs)
#z_val = mt.z_test(logOR_all, logORSE_subs, logORSE_all, logORSE_subs)
df = add_subset_col(df, 1000)
print(df[df.disease==1].head())