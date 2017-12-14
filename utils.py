# Matplot ticks
import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 1.
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 1.
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal

import numpy as np
from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def plot_cov_ellipse(ax, mus, covs, var_num1, var_num2, MoG_color="Blue"):
    N_ellip = len(mus)
    for i in range(N_ellip):
        cov = covs[i]
        cov = [[cov[var_num1, var_num1], cov[var_num1, var_num2]], [cov[var_num2, var_num1], cov[var_num2, var_num2]]]
        mu = mus[i]
        mu = [mu[var_num1], mu[var_num2]]
        for j in [1, 2]:
            width, height, theta = cov_ellipse(cov, q=None, nsig=j)
            e = Ellipse(xy=mu, width=width, height=height, angle=theta, lw=1.25)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor("none")
            e.set_edgecolor(MoG_color)

    return



# ----- Convergence statistics
def split_R(q_chain, thin_rate = 5, warm_up_frac = 0.5, warm_up_num = None):
    """
    Given MCMC chain with dimension (Nchain, Niter, D) return Gelman-Rubin statistics corresponding to each.
    """
    Nchain, Niter, D = q_chain.shape
    
    assert Nchain >= 1
    
    if warm_up_num is None: # Use the warm-up fraction provided
        warm_up_num = int(Niter * warm_up_frac)
    
    chains = [] # chains to be used to calculate the statistics.
    for m in xrange(Nchain):
        # For each chain dicard warm-up samples.
        q_chain_warmed = q_chain[m, warm_up_num:, :]
            
        # Thin the resulting chain.
        q_chain_warmed_thinned = q_chain_warmed[::thin_rate, :]
        L_chain = q_chain_warmed_thinned.shape[0]
        assert (L_chain % 2) == 0

        # Split the remaining chain in two parts and save.
        n = L_chain/2        
        chains.append(q_chain_warmed_thinned[:n])
        chains.append(q_chain_warmed_thinned[n:])
    
    m = len(chains)
    # Compute within chain variance, W
    # This an under-estimate of true variance.
    var_within = np.empty((m, D))
    for j in range(m):
        var_within[j, :] = np.std(chains[j], ddof=1, axis=0)
    W = np.mean(var_within, axis=0)
    
    # Compute between chain variance, B
    # This in general is an over-estimate because of the overdispersion of the starting distribution.
    mean_within = np.empty((m, D))
    for j in range(m):
        mean_within[j, :] = np.mean(chains[j], axis=0)
    B = np.std(mean_within, ddof=1, axis=0) * n
    
    # Unbiased posterior variance estimate
    var = W * (n-1)/float(n) + B / float(n)
    
    # Compute Gelman-Rubin statistics
    R = np.sqrt(var/W)
    
    return R

def acceptance_rate(decision_chain, start=None, end=None):
    """
    Return acceptance given the record of decisions made
    1: Accepted
    0: Rejected
    
    start, end: Used if average should be taken in a range
    rather than full.
    """
    _, Niter, _ = decision_chain.shape            
    if start is None and end is None:
        return np.sum(decision_chain, axis=(1, 2))/Niter
    else:
        if end > 0:
            Niter = end - start 
        else:
            Niter = Niter - start
        return np.sum(decision_chain[:, start:end, :], axis=(1, 2))/Niter  



def start_pts(q0, cov0, size):
    """
    Returns one starting point from a normal distribution
    with mean "q0" and diagonal covariance "cov".
    """
    return np.random.multivariate_normal(q0, cov0, size=size)



def normal_lnL(q, q0, cov0):
    """
    Multivarite-normal lnL.
    """
    
    return multivariate_normal.logpdf(q, mean=q0, cov=cov0)
