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


def plot_cov_ellipse(ax, mus, covs, var_num1, var_num2, MoG_color="Blue", lw=2):
    N_ellip = len(mus)
    for i in range(N_ellip):
        cov = covs[i]
        cov = [[cov[var_num1, var_num1], cov[var_num1, var_num2]], [cov[var_num2, var_num1], cov[var_num2, var_num2]]]
        mu = mus[i]
        mu = [mu[var_num1], mu[var_num2]]
        for j in [1, 2]:
            width, height, theta = cov_ellipse(cov, q=None, nsig=j)
            e = Ellipse(xy=mu, width=width, height=height, angle=theta, lw=lw)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor("none")
            e.set_edgecolor(MoG_color)

    return




# ----- Convergence statistics
def convergence_stats(q_chain, thin_rate = 5, warm_up_num = 0):
    """
    Given MCMC chain with dimension (Nchain, Niter, D) return 
    - Gelman-Rubin statistics corresponding to each variable.
    - Effective sample number for each varaible.
    """
    Nchain, Niter, D = q_chain.shape
    
    assert Nchain > 1 # There should be at least two chains.
        
    chains = [] # chains to be used to calculate the statistics.
    for m in xrange(Nchain):
        # For each chain dicard warm-up samples.
        q_chain_warmed = q_chain[m, warm_up_num:, :]
            
        # Thin the resulting chain.
        q_chain_warmed_thinned = q_chain_warmed[::thin_rate, :]
        L_chain = q_chain_warmed_thinned.shape[0]
        if (L_chain % 2) == 0:
            pass
        else:
            q_chain_warmed_thinned = q_chain_warmed_thinned[:L_chain-1]

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

    #---- n_eff computation
    n_eff = np.zeros(D, dtype=float) # There is an effective number for each.
    for i in range(D): # For each variable
        # First two base cases rho_t
        V_t1 = variogram(chains, i, 1)
        V_t2 = variogram(chains, i, 2)
        rho_t1 = 1 - V_t1/(2*var[i])
        rho_t2 = 1 - V_t2/(2*var[i])
        rho_t = [rho_t1, rho_t2]# List of autocorrelation numbers: Unknown termination number.
        t = 1 # Current t
        while (t < n-2): # While t is less than the length of the chain
            # Compute V_t and rho_t
            V_t = variogram(chains, i, t+2)
            rho_t.append(1 - V_t/(2*var[i]))

            # Check for termination condition
            if ((t%2)==1) & ((rho_t[-1]+rho_t[-2]) < 0): # If t is odd and t
                break
            
            # Otherwise just update t
            t += 1
        
        # Sum all rho upto maximum T
        sum_rho = np.sum(rho_t[:-2])
        n_eff[i] = m*n/(1+2*sum_rho)# Computed n_eff
        # print i, sum_rho, n_eff[i]


    return R, n_eff

def variogram(chains, var_num, t_lag):
    """
    Variogram as defined in BDA above (11.7).
    
    Args:
    - chains: List with dimension(m chains, (n samples, number of variables)).
    - var_num: Variable of interest
    - t_lag: Time lag
    """
    m = len(chains)
    n = chains[0].shape[0]
    V_t = 0
    for i in range(m): # For each chain
        chain_tmp = chains[i] # Grab the chain
        # Compute the inner sum
        inner_sum = np.sum(np.square(chain_tmp[t_lag:, var_num]-chain_tmp[:-t_lag, var_num]))
        # Add to the cumulative sum
        V_t += inner_sum
    V_t /= float(m*(n-t_lag))

    return V_t    


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


# /--- Used for NUTS termination criteria conditions
def find_next(table):
    """
    Given the save index table, return the first empty slot.
    """
    for i, e in enumerate(table):
        if e == -1:
            return i

def retrieve_save_index(table, l):
    """
    Given the save index table and the point number,
    return the save indexe corresponding to the point.s
    """
    for i, m in enumerate(table):
        if m == l:
            return i

def power_of_two(r):
    """
    Return True if r is power of two, False otherwise.
    """
    assert type(r) == int
    return np.bitwise_and(r, r-1) == 0
    
def check_points(m):
    """
    Given the current point m, return all points against which to check
    the terminiation criteria. Assumes m is even.
    """
    assert (m % 2) ==0
    r = int(m)
    # As long as r is not a power of two, keep subtracting the last possible power of two.
    d_last = np.floor(np.log2(r))
    
#     #---- Debug lines
#     counter = 0
#     print counter
#     print "r", r
#     print "d_last", d_last
#     print "\n"
    while ~power_of_two(r) and r>2:
        d_last = np.floor(np.log2(r))
        r -= int(2**d_last)
        d_last -=1
#         #---- Debug lines
#         counter +=1
#         print counter
#         print "r", r
#         print "d_last", d_last
#         print "\n"
        
    pow_tmp = np.log2(r)
    start = m-r+1
    pts = [start]
    
    tmp = start
    while pow_tmp > 1:
        pow_tmp-=1
        tmp += int(2**(pow_tmp))
        pts.append(tmp)
    
    return np.asarray(pts)


def release(m, l):
    """
    Given the current point m and that the termination condition was
    checked against l, return True if the point should no longer be saved.
    Return False, otherwise.
    """
    assert (l != 1) and (m %2) ==0
    r_m, r_l = int(m), int(l)
    d_last = np.floor(np.log2(r_m))
    while ~power_of_two(r_m) and r_m>4:
        tmp = int(2**d_last)
        r_m -= tmp
        r_l -= tmp
        d_last = np.floor(np.log2(r_m))        
    
    if (r_m >= 4) and (r_l>1):
        return True
    else:
        return False

def test_NUTS_binary_tree_flatten():
    """
    Code used to test whether the auxilary functions are working well.
    """
    d = 5
    d_max = 10
    D = 1

    # Arrays in which to save intermediate points
    q_save = np.zeros((d_max+1, D), dtype=float)
    p_save = np.zeros((d_max+1, D), dtype=float)

    # Index table
    # Initially all -1. If not -1, then holds the point number m. The corresponding
    # array index is the save index. This scheme requires manual release.
    save_index_table = np.ones(d_max+1, dtype=int) * -1

    trajectory_reject = False

    def print_line(m, save_index_table):
        print_line = "%2d: " % m
        for i in range(1, m+1):
            if (i in save_index_table) or (i == 1) or (i==m):
                print_line = print_line + "x "
            else:
                print_line = print_line + "o "
        print print_line
        return None

    for m in range(2, 2**d+1):
        # Update the trajectory
    #     q_tmp, p_tmp = update(q_tmp, p_tmp)

        # Decide whether to save the point for future comparison.
        if (m % 2) == 1: # Only odd numbered points are saved.
            save_index = find_next(save_index_table)
    #         q_save[save_index, :] = q_tmp # Current point
    #         p_save[save_index, :] = p_tmp 
            save_index_table[save_index] = m    
            print_line(m, save_index_table)
        else:
            print_line(m, save_index_table)        
            # Check termination conditions against each point.
            check_pts = check_points(m)
            for l in check_pts:
                # Retrieve a previous point 
                save_index = retrieve_save_index(save_index_table, l)
                q_check = q_save[save_index, :] 
                p_check = p_save[save_index, :] 
    #             # Check termination condition
    #             left_terminate, right_terminate = check_termination(q_check, p_check, q_tmp, p_tmp)
    #             if left_terminate and right_terminate:
    #                 # If the termination condition is satisfied by any subtree
    #                 # reject the whole trajector expansion.
    #                 trajectory_reject = True
    #                 break 

                # If the point is no longer needed, then release the space.     
                if (l > 1) and release(m, l):
                    save_index_table[save_index] = -1
    
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

