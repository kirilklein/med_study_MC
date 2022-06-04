import os

from pathlib import Path
from os.path import join
file_path = os.path.realpath(__file__)
base_dir = Path(file_path).parent
# if base_dir not in sys.path:
    # sys.path.append(base_dir)
# if Path(file_path).parent not in sys.path:
    # sys.path.append(Path(file_path).parent)
# from . import util.sim import simulate_austin
import numpy as np
import proplot as pplt
import string
from util import simulate_austin
import pickle as pkl

prevalences = np.logspace(0, 3, 4, base=2)/100
variable_setting_dic = {'a'}
a0_med_ls_ls = []
#fig, axs = plt.subplots()#
fig, axs = pplt.subplots(ncols=3, nrows=2, refwidth=2,abc=True,)
#axs = axs.flatten()
alpha0_med_dic = {}
for i in range(6):
    setting = string.ascii_lowercase[i]
    X = simulate_austin.simulate_pats(setting, num_patients=1000)
    a0_med_ls = []
    #diffs_ls = []
    for j, prevalence in enumerate(prevalences):
        a0_med, diffs = simulate_austin.get_alpha0(X, prevalence=prevalence, iter=100)
        a0_med_ls.append(a0_med)
        axs[i].boxplot(y=np.divide(diffs, prevalence), positions=prevalence, widths=.004*(1+j))
        
        #diffs_ls.append(diffs)
    #axs[i].format(xlim=(0,0.11), xlocator=prevalences, xformatter='{:.2f}')
    axs[i].format(xlim=(0.005,0.09), xscale=('power', 1/2), xticks=prevalences)
    axs[i].set_title(simulate_austin.var_set_dic[setting])
    alpha0_med_dic[setting] = a0_med_ls
fig.format(xlabel='p', ylabel=r'$(p-p_{\mathrm{sim}})/p$') 
fig.savefig(join(base_dir, 'figs', 'get_alpha_0_res_1000_pats_100_iter.png'),dpi=300)
with open(join(base_dir, 'data_and_params', 'alpha0_meds.pkl'), 'wb') as f:
    pkl.dump(alpha0_med_dic, f)