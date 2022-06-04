import os
from pathlib import Path
from os.path import join
file_path = os.path.realpath(__file__)
base_dir = Path(file_path).parent
import pickle as pkl
import seaborn as sns
import proplot as pplt
import numpy as np
import pandas as pd
from util import simulate_austin
import matplotlib.pyplot as plt


with open(join(base_dir, 'data_and_params', 'alpha0_meds.pkl'), 'rb') as f:
    alpha0_med_dic = pkl.load(f)
alpha0_df = pd.DataFrame(alpha0_med_dic).transpose()
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9.5,2.5))
sns.heatmap(alpha0_df, ax=axs[0], annot=True)
axs[0].set_xticklabels(np.logspace(0, 3, 4, base=2)/100)
fig.savefig(join(base_dir, 'figs', 'alpha_0_beta0_exp_beta_0_res_1000_pats_100_iter.png'),dpi=300)