from utils import *

# print "\----- Un-correlated multivariate normal (posterior)"
print "D: Mean std"
for D in [2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 500]:
    q0 = np.zeros(D, dtype=np.float) # True mean
#     cov = np.diag(np.ones(D)) * 0.01
#     cov += 0.99
    cov = np.diag(np.ones(D))
    Nsample = int(100)

    samples = np.random.multivariate_normal(q0, cov, size=Nsample)
    mus = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1)
    mu_std = std/np.sqrt(Nsample)
    mu_std_abs_mean = np.mean(np.abs(mu_std))
    # print mus, mu_std, mu_std_abs_mean
    print "%d: %.2e" % (D, mu_std_abs_mean)
    
print "See README for some thoughts."