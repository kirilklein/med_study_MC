import numpy as np

def get_num_variables(df):
    return len([k for k in df.keys() if k.startswith('x')])

def get_num_hidden_variables(df):
    return len([k for k in df.keys() if k.startswith('hx')])

def produce_hist_values(x_all, N_bins, x_range = None, poisson_error = False,
                        only_nonzero = False, log = False):
    r"""
    Produce histogram
    
    Parameters
    --------------
    x_all: array_like
        Input data.
    N_bins: int
        defines the number of equal-width
        bins in the given range
    x_range: (float, float), optional
        Lower and upper range for the bins,
        if not provided range is (x_all.min(), x_all.max())
    only_nonzero: bool, optional
        if True return only values that are associated with bins with y!=0
    Returns
    -------
    x: array_like
        Centres of bins.
    y: array_like
        Counts
    sy: array_like
        Error on counts assuming Poisson distributed bin counts
    binwidth: float
    """
    if x_range is None:
        x_range = (x_all.min(), x_all.max())
    if log:
        N_bins = np.logspace(np.log10(x_range[0]),
                             np.log10(x_range[1]), N_bins)
    counts, bin_edges = np.histogram(x_all, bins=N_bins, 
                                     range=x_range)
    x = (bin_edges[1:] + bin_edges[:-1])/2 
    binwidth = bin_edges[1]-bin_edges[0]
    y = counts  #assume: the bin count is Poisson distributed.
    
    if poisson_error:
        sy = np.sqrt(counts)
    
    if only_nonzero:
        mask = counts>0
        x, y = x[mask], y[mask] 
        if poisson_error:
            sy =  sy[mask]
            
    if poisson_error:
        return x, y, sy, binwidth
    else:
        return x, y, binwidth

