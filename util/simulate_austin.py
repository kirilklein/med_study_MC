import numpy as np
import scipy.stats as ss
import scipy.optimize as so


var_set_dic = {'a':'Independent Normal',
                'b':'Multivariate Normal',
                'c':'Binary',
                'd':'Binary/Independent Normal 5:5',
                'e':'Correlated Binary',
                'f':'Binary/Normal 8:2'}

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

#############################################################
def prevalence_diff(a0, setting, num_vars, num_pats, prevalence, iters):
    """Compute difference between actual and desired exposure prevalence,
    averaged over 1000 iterations
    a0: alpha0 
    prevalence: desired prevalence
    """
    diffs = []
    for _ in range(iters):
        X = simulate_pats(setting, num_vars, num_pats)
        z = simulate_exposure(X, a0)
        diffs.append(np.sum(z)/len(z)-prevalence)
    return np.mean(diffs)

def get_alpha0(setting, prevalence, num_vars=10, num_pats=1000, 
            iters=1000, a=-10, b=0):
    """
    setting: a-f
    prevalence: desired prevalence
    iters: number of iterations
    a, b: start interval for bisection [a,b]
    num_vars: number variables to simulate
    num_pats: number of patients
    Exposure is simulated iter times and the corresponding alpha0 is computed.
    For every simulated exposure and alpha0, the difference between true 
    and desired prevalence is returned.
    """
    f = lambda y: prevalence_diff(y, setting, num_vars, num_pats,
                prevalence, iters)
    alpha0_res = so.bisect(f, a=a, b=b)
    return alpha0_res
####################################################


def compute_outcome_prob(X, beta0_exp, beta_exp, exposures,  
        al = np.log(1.25),
        am = np.log(1.5),
        ah = np.log(1.75),
        avh = np.log(2)):
    exponent = beta0_exp + beta_exp*exposures + al*X[:,1] + al*X[:,2]\
        + am*X[:,4] + am*X[:,5]\
            + ah*X[:,7]+ ah*X[:,8]\
                + avh*X[:,9]
    p = 1/(1+np.exp(-exponent))
    return p

def simulate_outcome(X, beta0_exp, beta_exp, exposures,  
        al = np.log(1.25),
        am = np.log(1.5),
        ah = np.log(1.75),
        avh = np.log(2)):
    """Coefficients a_ are same as in austin 2014"""
    p = compute_outcome_prob(X, beta0_exp, beta_exp, exposures, 
        al=al, am=am, ah=ah, avh=avh)
    z = ss.bernoulli.rvs(p=p)
    return z

def incidence_diff(beta0_exp, incidence, setting, num_vars, num_pats, iters):
    """Compute difference between actual and desired exposure prevalence
    a0: alpha0 
    prevalence: desired prevalence
    """
    diffs = []
    for _ in range(iters):
        X = simulate_pats(setting, num_vars, num_pats)
        z = simulate_outcome(X, beta0_exp, 0, np.zeros(len(X)))
        diffs.append(np.sum(z)/len(z)-incidence)
    return np.mean(diffs)

def get_beta0_exp(setting, incidence, num_vars=10, num_pats=1000, 
            iters=1000, a=-10, b=0):
    """
    incidence: desired incidence
    iters: number of iterations
    a, b: start interval for bisection [a,b]
    Outcome is simulated iters times, assuming no patients are exposed,
    """
    f = lambda y: incidence_diff(y, incidence, setting, 
                num_vars, num_pats, iters)
    beta0_exp_res = so.bisect(f, a=a, b=b)
    return beta0_exp_res
#####################################################################
def simulate_risk_difference(beta_exp, setting, alpha0, beta0_exp, 
                        num_vars=10, num_patients=1000, iters=1000):
    gammas = []
    for _ in range(iters):
            X = simulate_pats(setting, num_vars, num_patients)
            exposures = simulate_exposure(X, alpha0)
            p = compute_outcome_prob(X, beta0_exp, 
                beta_exp, exposures)
            p0 = np.mean(p[exposures==0])
            if (exposures==1).sum()==0:
                p1 = 0
            else:
                p1 = np.mean(p[exposures==1])
            gammas.append(p1 - p0)  # Different than described in the paper (p0-p1)
    return np.mean(gammas)

def get_beta_exp(alpha0, beta0_exp, setting, desired_gamma, num_vars=10, 
        num_patients=1000, iters=1000,  a=-3, b=2):
    f = lambda y: simulate_risk_difference(y, setting, alpha0, beta0_exp,
                     num_vars, num_patients, iters) - desired_gamma
    bexp_res = so.bisect(f, a=a, b=b)
    return bexp_res
#####################################################################
def compute_rr(exposures, outcomes):
    """Compute the odds ratio"""
    n_exp = np.sum(exposures)
    n_nexp = len(exposures) - n_exp
    
    n_exp_pos = np.sum(outcomes[exposures==1])
    n_exposed_neg = len(n_exp) - n_exp_pos
    
    n_nexp_pos= np.sum(outcomes[exposures==0])
    n_nexp_neg = len(n_nexp) - n_nexp_pos
    # TODO: continue here
    pass
