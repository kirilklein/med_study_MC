import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import norm
######################Simulate population##############
def get_rand_uniform(num_variables, random_state=0):
    if type(random_state)==int:
        rng = np.random.RandomState(random_state)
        return rng.rand(num_variables)
    else:
        return np.random.rand(num_variables)

def compute_PS(beta, X):
    return 1/(1+np.exp(-(beta[0]+beta[1:]@X.T)))

def compute_disease_proba(df, gamma):
    variables_cols = [col for col in df.keys() \
        if col.startswith('x') or col.startswith('hx')]
    disease_proba =  1/(1+np.exp(-(gamma[0]+gamma[1]*df.exposed\
        +gamma[2:]@df[variables_cols].T)))
    df['disease_proba'] = disease_proba
    return df

def simulate_variables_and_PS(beta, num_variables, population_size, random_state=0):
    X = norm.rvs(size=(population_size, num_variables), 
                random_state=random_state)
    true_PS = compute_PS(beta, X)
    return X, true_PS 

def simulate_exposure(beta, num_hidden_variables,
        population_size, random_state=0):
    """Returns dataframe with simulated, gaussian variables (mu=0, sig=1) and 
    exposure (0 or 1) for population_size patients. 
    Exposure is simulated using probabilities from a logistic model 
    (true propensity score)"""
    num_variables = len(beta)-1
    X, true_PS = simulate_variables_and_PS(
        beta, num_variables, population_size, random_state=random_state)
    exposures = np.zeros(len(X))
    if type(random_state)==int:
        rng = np.random.RandomState(random_state+1)
        exposures[true_PS>rng.rand(len(X))] = 1
    else:
        exposures[true_PS>np.random.rand(len(X))] = 1
    df_data = np.concatenate([X, exposures.reshape(-1,1)], axis=1)
    df_columns = ['x'+str(i) for i in range(num_variables-num_hidden_variables)]
    df_columns = df_columns + ['hx'+str(i) for i in range(num_hidden_variables)]
    df_columns.append('exposed')
    df = pd.DataFrame(data=df_data, columns=df_columns)
    return df

def get_gamma1(true_OR, gamma0):
    """ crude estimate: np.log(true_OR)
    Calculate gamma in gamma*t from the OR (rare disease assumption)"""
    gamma0_max = np.log(1/(true_OR-1))
    assert gamma0<=gamma0_max, f'gamma0>np.log(1/(true_OR-1))={gamma0_max:.3f}'
    return np.log(true_OR)-np.log(1-(true_OR-1)*np.exp(gamma0))

def get_gamma(gamma0, true_OR, gamma_ls):
    gamma1 = get_gamma1(true_OR, gamma0)
    gamma = np.array([gamma0, gamma1, *gamma_ls])
    return gamma


def simulate_disease(df, odds_exp, OR, random_state=0):
    """
    Parameters:
        df: Df with exposed column (0 or 1)
        odds_exp: odds of getting the disease when being exposed
        OR: odds ratio
    returns: Dataframe with disease column (0 or 1)"""
    df_exp = df[df.exposed==1]
    df_nexp = df[df.exposed==0]
    
    if type(random_state)==int:
        rng1 = np.random.RandomState(random_state+2)
        rng2 = np.random.RandomState(random_state+3)
    else:
        rng1 = np.random
        rng2 = np.random
    df_exp['disease'] = rng1.rand(len(df_exp))<odds_exp
    df_nexp['disease'] = rng2.rand(len(df_nexp))<odds_exp/OR
    df = pd.concat([df_exp, df_nexp], ignore_index=True)
    df['disease'] = df['disease']*1
    return df

def get_positives_and_random_subset(df, n_subset, random_state=0):
    """Selects all the sick and a random subset of size n_subset of the rest of the population"""
    dfd = df[df.disease==1]
    dfnd = df[df.disease==0]
    deck = np.arange(len(dfnd))
    if type(random_state)==int:
        rng = default_rng(random_state)
        rng.shuffle(deck)
    else:
        np.random.shuffle(deck)
    df_rand = dfnd.iloc[deck[:n_subset], :].reset_index()
    df_subs = pd.concat([dfd, df_rand], ignore_index=True)
    return df_subs

def get_positives_and_random_subset(df, n_subset, random_state=0):
    """Selects all the sick and a random subset of size n_subset of the rest of the population"""
    dfd = df[df.disease==1]
    dfnd = df[df.disease==0]
    deck = np.arange(n_subset)
    if type(random_state)==int:
        rng = default_rng(random_state)
        rng.shuffle(deck)
    else:
        deck = np.random.shuffle(deck)
    df_rand = dfnd.iloc[deck, :].reset_index()
    df_subs = pd.concat([dfd, df_rand], ignore_index=True)
    return df_subs