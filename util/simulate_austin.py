import numpy as np
import scipy.stats as ss
import scipy.optimize as so


def simulate_pats(setting, num_vars=10, num_patients=1000,
        correlation=0.25, p=0.5, cutoff=0):
    """Simulating patient variables as described in 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4285163/#b16."""
    if setting=='a':
        return ss.norm.rvs(size=(num_patients, num_vars))
    if setting=='b':
        cov_mat = np.eye(num_vars)
        cov_mat[cov_mat==0] = correlation
        vars = np.random.multivariate_normal(
                    mean=np.zeros(num_vars),
                    cov=cov_mat, size=num_patients)
        return vars
    if setting=='c':
        x_n = ss.norm.rvs(size=(num_patients, 5))
        x_b = ss.bernoulli.rvs(p=p, size=(num_patients,5))
        return np.concatenate((x_b, x_n), axis=1)
    if setting=='d':
        return ss.bernoulli.rvs(p=p, size=(num_patients,10))
    if setting=='e':
        x_n = simulate_pats('b', num_vars)
        return np.where(x_n<cutoff, 0, 1)
    if setting=='f':
        x_n = simulate_pats('b', num_vars=2)
        x_b = simulate_pats('e', num_vars=8)
        return np.concatenate((x_b, x_n), axis=1)


def simulate_exposure(X, a0, 
        al = np.log(1.25),
        am = np.log(1.5),
        ah = np.log(1.75),
        avh = np.log(2)):
    """Coefficients a_ are same as in austin 2014"""
    exponent = a0 + al*X[:,0] + al*X[:,1]\
        + am*X[:,3] + am*X[:,4]\
            + ah*X[:,6]+ ah*X[:,7]\
                + avh*X[:,9]
    p = 1/(1+np.exp(-exponent))
    z = ss.bernoulli.rvs(p=p)
    return z
def prevalence_diff(a0, prevalence, X):
    """Compute difference between actual and desired exposure prevalence
    a0: alpha0 
    prevalence: desired prevalence
    X: patient variables"""
    z = simulate_exposure(X, a0)
    return np.sum(z)/len(z)-prevalence

def get_alpha0(X, prevalence, iter=100, a=-10, b=0):
    """
    x: patient variables
    prevalence: desired prevalence
    iter: number of iterations
    a, b: start interval for bisection [a,b]
    Exposure is simulated iter times and the corresponding alpha0 is computed.
    For every simulated exposure and alpha0, the difference between true 
    and desired prevalence is returned.
    """
    alpha_res_ls = []
    diffs = []
    f = lambda y: prevalence_diff(y, prevalence=prevalence, X=X)
    for _ in range(iter):
        alpha_res = so.bisect(f, a=a, b=b)
        diffs.append(f(alpha_res))
        alpha_res_ls.append(alpha_res)
    return np.median(alpha_res_ls), diffs
    

def simulate_outcome(X, beta0_exp, beta_exp, exposures,  
        al = np.log(1.25),
        am = np.log(1.5),
        ah = np.log(1.75),
        avh = np.log(2)):
    """Coefficients a_ are same as in austin 2014"""
    exponent = beta0_exp + beta_exp*exposures + al*X[:,1] + al*X[:,2]\
        + am*X[:,4] + am*X[:,5]\
            + ah*X[:,7]+ ah*X[:,8]\
                + avh*X[:,9]
    p = 1/(1+np.exp(-exponent))
    z = ss.bernoulli.rvs(p=p)
    return z

def incidence_diff(beta0_exp, X, incidence):
    """Compute difference between actual and desired exposure prevalence
    a0: alpha0 
    prevalence: desired prevalence
    X: patient variables"""
    z = simulate_outcome(X, beta0_exp, 0, np.zeros(len(X)))
    return np.sum(z)/len(z)-incidence

def get_beta0_exp(X, incidence, iter=100, a=-10, b=0):
    """
    X: patient variables
    incidence: desired incidence
    iter: number of iterations
    a, b: start interval for bisection [a,b]
    Outcome is simulated iter times, assuming no patients are exposed,
    For every simulated outcome the difference between true 
    and desired incidence is returned.
    """
    beta0_exp_res_ls = []
    diffs = []
    f = lambda y: incidence_diff(y, incidence=incidence, X=X)
    for _ in range(iter):
        beta0_exp_res = so.bisect(f, a=a, b=b)
        diffs.append(f(beta0_exp_res))
        beta0_exp_res_ls.append(beta0_exp_res)
    return np.median(beta0_exp_res_ls), diffs