import numpy as np
import pandas as pd
from util import helper
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
####################evaluation##############################


def compute_bias(estimated_OR, true_OR):
    """estimate bias as in https://academic.oup.com/aje/article/158/3/280/70529"""
    return ((estimated_OR/true_OR)-1)*100
def estimate_precision(logOR_SE_ls):
    return np.median(logOR_SE_ls)
def crude_estimation_OR(df, RR=False):
    """Estimation of the OR or RR if RR==True, 
    from the disease probabilities. 
    only works if gamma_i, i>1, is small"""
    d1e1 = df[df.exposed==1].disease_proba.median()
    print(d1e1)
    d0e1 = 1 - d1e1
    d1e0 = df[df.exposed==0].disease_proba.median()
    print(d1e0)
    d0e0 = 1 - d1e0
    if RR:
        return d1e1/d1e0
    else:
        return d0e0*d1e1/(d1e0*d0e1)
        
def crude_estimation_exp1dis1(df):
    return df[df.exposed==1].disease_proba.mean()*(df.exposed==1).sum()
def crude_estimation_exp0dis1(df):
    return df[df.exposed==0].disease_proba.mean()*(df.exposed==0).sum()

def get_contingency_table(df, two_by_two=True):
    ct = pd.crosstab(index=df['exposed'], columns=df['disease'], margins=True)
    if two_by_two:
        return np.array(ct.iloc[:2,:2])
    else:
        return ct

def compute_OR_pval(df):
    ct = get_contingency_table(df)
    OR, pval = fisher_exact(ct) 
    return OR, pval
    
def compute_logOR_SE(df):
    logOR = np.log(compute_OR_pval(df)[0])
    ct = get_contingency_table(df)
    logOR_SE = np.sqrt(np.sum(1/ct))
    return logOR, logOR_SE 

def compute_OR_95CI(df):
    OR = compute_OR_pval(df)[0]
    ct = get_contingency_table(df)
    range_ = 1.96*np.sqrt(np.sum(1/ct))
    a = np.log(OR)-range_
    b = np.log(OR)+range_
    return (np.exp(a), np.exp(b))

def compute_OR_CI_pval(df, print_=False, start_string=''):
    ct = get_contingency_table(df)
    CI = compute_OR_95CI(df)
    OR, p_val = fisher_exact(ct) 
    if print_:
        print(start_string, "\n     OR (95% CI) =", OR, 
            f'({CI[0]:.2f},{CI[1]:.2f})', '\n    p =', p_val)
    return OR, CI, p_val

def OR_z_test(df):
    """Compute z-value for ORs between the whole 
    population (true OR) and (matched) subset of population"""
    logOR_subs, logORSE_subs = compute_logOR_SE(df[df.subset==1])
    logOR_all, logORSE_all = compute_logOR_SE(df)
    return np.abs(logOR_all-logOR_subs)/np.sqrt(logORSE_subs**2+logORSE_all**2)
def z_test_two_sided(mu1, mu2, sig1, sig2):
    return np.abs(mu1-mu2)/np.sqrt(sig1**2+sig2**2)

def run_logistic_regression(df, outcome='exposed',random_state=0):
    """Predict probability of being exposed"""
    num_variables = helper.get_num_variables(df)
    variables_columns = ['x'+str(i) for i \
        in range(num_variables)]
    X = df[variables_columns]
    y = df[outcome]
    if type(random_state)==int:
        LR = LogisticRegression(random_state=random_state).fit(X, y)
    else:
        LR = LogisticRegression().fit(X, y)
    if outcome=='exposed':
        df['PS'] = LR.predict_proba(X)[:,1]
    elif outcome=='disease':
        df['P_disease'] = LR.predict_proba(X)[:,1]
    else:
        print("Choose either exposed or disease as outcome")
    return df
