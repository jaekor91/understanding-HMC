from utils import *
from samplers import *

#---- Parameters
title_str = "case3"
Niter = 2000
Nchain = 10
N_warm_up = 1000
R_thin = 1
N_save_chain0 = 50

# Time step
global_dt = True # If True, same time step for all variables.
dt = 1e-1

# Random trajectory length limits
L_low = 50
L_high = 200




print "#---- Case 3d. ----#"
title_str = "./case3/case3d"
D = 100

#---- True parameters
# True mean
q0 = np.zeros(D, dtype=np.float) 

# True covariance
rho = 0.95
cov0 = np.diag(np.ones(D)) * (1-rho)
cov0 += rho

# Inverse covariance = percision matrix
inv_cov0 = np.linalg.inv(cov0)

#---- Defining the input potential
def V(q):
    """
    Potential energy -lnL
    """
    return -normal_lnL(q, q0, cov0)

def dVdq(q):
    """
    Gradient of potential energy.
    """
    return np.dot(inv_cov0, (q-q0))

cov0_diag = []
for i in range(D):
    cov0_diag.append(cov0[i, i])

print "Min/Max of marginal variances: %.3f, %.3f" % (np.min(cov0_diag), np.max(cov0_diag))

# Starting points
cov_start = np.diag(np.ones(D)) * 2
q_start = start_pts(q0, cov_start, Nchain)

# --- Random trajectory length ---- #
HMC1 = HMC_sampler(D, V, dVdq, Niter=Niter, Nchain=Nchain, sampler_type="Random", L_low=L_low, \
                  L_high=L_high, dt=dt, thin_rate=R_thin, warm_up_num = N_warm_up)
HMC1.gen_sample(q_start, N_save_chain0 = N_save_chain0)
HMC1.compute_convergence_stats()
HMC1.plot_samples(title_prefix=title_str, savefig=True, show=False, plot_normal=True, q0=q0, cov0=cov0)
HMC1.make_movie(title_prefix=title_str, q0=q0, cov0=cov0, plot_cov=True, qmin=-4, qmax=4)

print "Random"
print "Total number of samples: %d" % ((HMC1.L_chain-1) * HMC1.Nchain)
print "Effective number per param: ", HMC1.n_eff_q
print "Ratio", HMC1.n_eff_q/((HMC1.L_chain-1) * HMC1.Nchain)
print "\n"
