import numpy as np
import scipy.stats as ss


def simulate_variables(setting, num_vars=10, num_patients=1000,
        correlation=0.25, p=0.5, cutoff=0):
    """Simulating variables as described in 
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
        x_n = simulate_variables('b', num_vars)
        return np.where(x_n<cutoff, 0, 1)
    if setting=='f':
        x_n = simulate_variables('b', num_vars=2)
        x_b = simulate_variables('e', num_vars=8)
        return np.concatenate((x_b, x_n), axis=1)

