
"""Compute alpha0 similarly tp determination of beta as described in 
Austin, Peter C. (2010).
 A Data-Generation Process for Data with Specified Risk Differences 
    or Numbers Needed to Treat. 
    Communications in Statistics - Simulation and Computation, 39(3), 563–577.
Simulating 1000 datasets with 1000 patients each to get estimate of prevalence.
This is then used for alpha0 bisection.
    """
import os
from pathlib import Path
file_path = os.path.realpath(__file__)
base_dir = Path(file_path).parent
import numpy as np
import string, itertools
from util import simulate_austin
import pickle as pkl
from multiprocessing import Pool

num_vars = 10
num_pats = 1000
iters = 1000
a = -9
b = -1


def get_alpha0_spec(input):
    """Helper function for multiprocessing"""
    setting, prevalence = input
    a0 = simulate_austin.get_alpha0(setting, prevalence, 
                num_vars=num_vars, num_pats=num_pats, 
                iters=iters, a=a, b=b)
    return a0

prevalences = np.logspace(0, 3, 4, base=2)/100
settings = [string.ascii_lowercase[i] for i in range(6)]
inputs = list(itertools.product(settings, prevalences)) 
a0_dic = {}

if __name__ == '__main__':    
    with Pool() as p:
        a0_ls_all = p.map(get_alpha0_spec, inputs)
    for i, setting in enumerate(settings):
        len_prevalence = len(prevalences)
        a0_dic[setting] =  a0_ls_all[i*len_prevalence:(i+1)*len_prevalence]
    with open(os.path.join(base_dir, 'data_and_params', 'alpha0_dic.pkl'), 'wb') as f:
        pkl.dump(a0_dic, f)       
           
           
         #axs[i].boxplot(y=np.divide(diffs, prevalence), positions=prevalence, widths=.004*(1+j))
            #diffs_ls.append(diffs)
        #axs[i].format(xlim=(0,0.11), xlocator=prevalences, xformatter='{:.2f}')
        #axs[i].format(xlim=(0.005,0.09), xscale=('power', 1/2), xticks=prevalences)
        #axs[i].set_title(simulate_austin.var_set_dic[setting], fontsize=8)
    # fig.format(xlabel='p', ylabel=r'$(p-p_{\mathrm{sim}})/p$') 
    # fig.savefig(join(base_dir, 'figs', 'get_alpha_0_res_1000_pats_100_iter.png'),dpi=300)
    