from utils import *


class sampler(object):
    """
    Parent class to MH_sampler and HMC_sampler. Contains common functions such as the plot function.
    """
    
    def __init__(self, D, target_lnL, Nchain=2, Niter=1000, thin_rate=1, warm_up_num=0):
        """
        Args:
        - D: Number of paramters to be inferred on.
        - target_lnL: Function that takes in a D-dimensional input q and outputs the log-like of the target distribution.
        - Nchain: Number of chains.
        - Niter: Number of iterations.
        - Thin rate
        - warm_up_num: Sample index at which to start computing statistics. 
        
        None of these variable can be changed. In other words, if you want to change the variables, 
        then one must create a new sampler object as only the appropriate number of samples are retained.        
        """
    
        self.D = D 
        self.target_lnL = target_lnL        
        self.Nchain = Nchain
        self.Niter = Niter
        self.thin_rate = thin_rate
        self.warm_up_num = warm_up_num
                    
        # Allocate memory for samples
        self.L_chain = 1+ ((self.Niter-self.warm_up_num)//self.thin_rate) # Length of individual chain
        # Simple example: Niter = 3, N_warm_up = 0 --> N_total = 4.
        self.q_chain = np.zeros((self.Nchain, self.L_chain, self.D), dtype=np.float) # Samples
        self.lnL_chain = np.zeros((self.Nchain, self.L_chain, 1)) # loglikelihood
        
        # Stats 
        self.R_q = None # R statistics for each parameter
        self.R_lnL = None # R stats for the loglikelihood.
        self.n_eff_q = None # Effective number of samples
        self.accept_R_warm_up = None # Acceptance rate during warm up
        self.accept_R = None # Acceptance rate after warm up

        # Time for measuring total time taken for inference
        self.dt_total = 0 # Only for chain inference.

        # Total number of computations. All of the following computations has a unit cost.
        # None other computations figure in this accounting.
        # - Gradient computation per variable: 1.
        # - Likelihood evaluation: 1
        self.N_total_steps = 0

        
    def compute_convergence_stats(self):
        """
        Compute stats on the chain and return.
        
        Remember, the chain has already been warmed up and thinned.
        """

        # Note that we should not include the first point.
        self.R_q, self.n_eff_q = convergence_stats(self.q_chain[:, 1:, :], warm_up_num=0, thin_rate=1)
        # self.R_lnL, _ = convergence_stats(self.lnL_chain, warm_up_num=0, thin_rate=1)

        return
    
    
    def plot_samples(self, title_prefix, show=False, savefig=False, xmax = None, dx = None, plot_normal=True, plot_cov=True, q0=None, cov0=None):
        """
        Plot the samples after warm up and thinning.
        Args:
        - show: If False, don't show the plot, whic wouldn't make sense unless savefig is True.
        - xmax: Unless specified by the user, the plot limit is set automatically,
         by first computing inner 95% tile and expanding it by 200 percent keeping the center the same. Assumes zero centering.
        - dx: Unless specified by the user, Bin size is set as 1% of the 95% tile range. This means with 1000 samples there should be on average
        10 samples in each bin.
        - savefig: Saves figure with the name fname. 
        - title_prefix: Title prefix must be provided by the user
        - plot_normal: If true, the user specified normal marginal for q1 and q2 are plotted with proper normalizatin.
        - plot_cov: If true, the the user provided normal models are plotted with proper normalizatin.
        - q0, cov0: Are parameters of the normal models to be overlayed.
        """
        
        #---- Extract samples from all chains
        q_chain_tmp_1 = self.q_chain[:, :, 0].flatten()
        q_chain_tmp_2 = self.q_chain[:, :, 1].flatten()
        E_chain_tmp = self.E_chain[:, 1:, :].flatten() # Only HMC samplers, so these are always defined.
        E_chain_tmp -= np.mean(E_chain_tmp)# Center the energies
        dE_chain_tmp = self.dE_chain[:, 1: , :].flatten()
        
        #---- Setting the boundary and binwidth
        # Boundary
        if xmax is None: # If the user didn't specify the range
            # q1 range
            # Compute 95 percent tile
            q1_max = np.percentile(q_chain_tmp_1, 97.5)
            q1_min = np.percentile(q_chain_tmp_1, 2.5)
            q1_range = q1_max - q1_min
            q1_center = (q1_max + q1_min)/2.
            # Adjust the range
            q1_range *= 2.5
            q1_max = q1_center + (q1_range/2.)
            q1_min = q1_center - (q1_range/2.)

            # q2 range
            # Compute 95 percent tile
            q2_max = np.percentile(q_chain_tmp_2, 97.5)
            q2_min = np.percentile(q_chain_tmp_2, 2.5)
            q2_range = q2_max - q2_min
            q2_center = (q2_max + q2_min)/2.
            # Adjust the range
            q2_range *= 2.5
            q2_max = q2_center + (q2_range/2.)
            q2_min = q2_center - (q2_range/2.)
        else:
            xmin = -xmax
            q1_min, q1_max = xmin, xmax
            q2_min, q2_max = xmin, xmax
            q1_range = q2_range = q1_max - q1_min            

        # Bin width
        if dx is None:
            dq1 = q1_range/100.
            dq2 = q2_range/100.
        else:
            dq1 = dq2 = dx 

        
        #---- Plot normal models, properly normalized.
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)
            assert q0.size == self.D
            xgrid1 = np.arange(q1_min, q1_max, dq1/10.)
            q1_marginal = multivariate_normal.pdf(xgrid1, mean=q0[0], cov=cov0[0, 0]) * self.L_chain * dq1 * self.Nchain
            xgrid2 = np.arange(q2_min, q2_max, dq2/10.)            
            q2_marginal = multivariate_normal.pdf(xgrid2, mean=q0[1], cov=cov0[1, 1]) * self.L_chain * dq2 * self.Nchain

        #---- Start of the figure generation ----#
        plt.close() # Clear any open panels.
        fig, ax_list = plt.subplots(3, 3, figsize = (20, 20))

        # Font sizes
        ft_size = 25 # axes labels, 
        ft_size2 = 20 # Legend
        ft_size_title = 30

        #---- Scatter plot
        ax_list[0, 0].scatter(q_chain_tmp_1, q_chain_tmp_2, s=2, c="black")
        if plot_cov:
            plot_cov_ellipse(ax_list[0, 0], [q0], [cov0], 0, 1, MoG_color="Blue", lw=2)
        ax_list[0, 0].set_xlabel("q1", fontsize=ft_size)
        ax_list[0, 0].set_ylabel("q2", fontsize=ft_size)
        ax_list[0, 0].axis("equal")
        ax_list[0, 0].set_xlim([q1_min, q1_max])
        ax_list[0, 0].set_ylim([q2_min, q2_max])
        
        #---- q2 histogram
        ax_list[0, 1].hist(q_chain_tmp_2, bins=np.arange(q2_min, q2_max, dq2), histtype="step", \
            color="black", orientation="horizontal", lw=2, label=(r"R = %.3f" % self.R_q[1]))
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)
            ax_list[0, 1].plot(q2_marginal, xgrid2, c="green", lw=3)
        ax_list[0, 1].set_ylim([q2_min, q2_max])
        ax_list[0, 1].set_ylabel("q2", fontsize=ft_size)
        ax_list[0, 1].legend(loc="upper right", fontsize=ft_size2)
        
        #---- q1 histogram
        ax_list[1, 0].hist(q_chain_tmp_1, bins=np.arange(q1_min, q1_max, dq1), histtype="step", \
            color="black", lw=2, label=(r"R = %.3f" % self.R_q[0]))
        if plot_normal:        
            ax_list[1, 0].plot(xgrid1, q1_marginal, c="green", lw=3)
        ax_list[1, 0].set_xlim([q1_min, q1_max])
        ax_list[1, 0].set_xlabel("q1", fontsize=ft_size)
        ax_list[1, 0].legend(loc="upper right", fontsize=ft_size2)

        #---- E and dE histograms
        # Compute the proper range
        E_min = np.percentile(E_chain_tmp, 2.5)
        E_max = np.percentile(E_chain_tmp, 97.5)
        E_range = (E_max - E_min) * 2.5
        E_center = (E_min+E_max)/2.
        E_min = E_center - (E_range/2.)
        E_max = E_center + (E_range/2.)
        bin_E = E_range/100.
        Egrid = np.arange(E_min, E_max, bin_E)
        ax_list[0, 2].hist(E_chain_tmp, bins=Egrid, histtype="step", color="black", label="E", lw=2)
        ax_list[0, 2].hist(dE_chain_tmp, bins=Egrid, histtype="step", color="red", label="dE", lw=2)        
        ax_list[0, 2].set_xlim([E_min, E_max])
        ax_list[0, 2].set_xlabel("Energy", fontsize=ft_size)
        ax_list[0, 2].legend(loc="upper right", fontsize=ft_size2)

        #---- Rhat distribution
        # Compute the proper range
        R_min = np.percentile(self.R_q, 2.5)
        R_max = np.percentile(self.R_q, 97.5)
        R_range = (R_max - R_min) * 2.5
        R_center = (R_min+R_max)/2.
        R_min = R_center - (R_range/2.)
        R_max = R_center + (R_range/2.)
        bin_R = R_range/50.
        Rgrid = np.arange(R_min, R_max, bin_R)
        ax_list[1, 2].hist(self.R_q, bins=Rgrid, histtype="step", color="black", lw=2, \
            label = ("R med/std: %.3f/ %.3f" % (np.median(self.R_q), np.std(self.R_q))))
        ax_list[1, 2].set_xlim([R_min, R_max])
        ax_list[1, 2].set_xlabel("Rhat", fontsize=ft_size)
        ax_list[1, 2].legend(loc="upper right", fontsize=ft_size2)           

        #---- Inferred standard deviations
        # Extracting true diagonal covariances
        cov0_diag = []
        cov_diag = [] # Inferred covariances
        for i in range(self.D):
            cov0_diag.append(cov0[i, i])
            cov_diag.append(np.std(self.q_chain[:, 1:, i])**2)
        # Converting
        cov0_diag = np.asarray(cov0_diag)
        cov_diag = np.asarray(cov_diag)        

        # Setting x-ranges for both plots below
        xmax = np.max(cov0_diag) * 1.1
        xmin = np.min(cov0_diag) * 0.9

        # Plotting true vs. inferred
        ymin = 0.5 * np.min(cov_diag)
        ymax = 1.5 * np.max(cov_diag)                
        ax_list[2, 1].scatter(cov0_diag, cov_diag, s=50, c="black", edgecolor="none")
        ax_list[2, 1].plot([xmin, xmax], [xmin, xmax], c="black", lw=2, ls="--")
        ax_list[2, 1].set_xlim([xmin, xmax])
        ax_list[2, 1].set_ylim([ymin, ymax])
        ax_list[2, 1].set_xlabel("True cov", fontsize=ft_size)
        ax_list[2, 1].set_ylabel("Estimated cov", fontsize=ft_size)

        # Plotting the ratio
        cov_ratio = cov_diag/cov0_diag
        ymin = 0.5 * np.min(cov_ratio)
        ymax = 1.5 * np.max(cov_ratio)        
        ax_list[2, 2].scatter(cov0_diag, cov_ratio, s=50, c="black", edgecolor="none")
        ax_list[2, 2].axhline(y=1, lw=2, c="black", ls="--")
        ax_list[2, 2].set_xlim([xmin, xmax])
        ax_list[2, 2].set_ylim([ymin, ymax])
        ax_list[2, 2].set_xlabel("True cov", fontsize=ft_size)
        ax_list[2, 2].set_ylabel("Ratio cov", fontsize=ft_size)

        #---- Inferred means
        q_mean = []
        for i in range(self.D):
            q_mean.append(np.mean(self.q_chain[:, 1:, i]))
        q_mean = np.asarray(q_mean)

        # Calculate the bias
        bias = q_mean-q0

        # Setting x-ranges for both plots below
        xmax = np.max(cov0_diag) * 1.1
        xmin = np.min(cov0_diag) * 0.9

        # Plotting histogram of bias
        ymax = np.max(bias)
        ymin = np.min(bias)
        y_range = ymax-ymin
        y_range *= 2.5
        y_center = (ymax+ymin)/2.
        ymax = y_center + (y_range/2.)
        ymin = y_center - (y_range/2.)       
        ax_list[2, 0].scatter(cov0_diag, bias, s=50, c="black", edgecolor="none")
        ax_list[2, 0].axhline(y=0, c="black", ls="--", lw=2)
        ax_list[2, 0].set_xlim([xmin, xmax])
        ax_list[2, 0].set_ylim([ymin, ymax])
        ax_list[2, 0].set_xlabel("True cov", fontsize=ft_size)
        ax_list[2, 0].set_ylabel("bias(mean)", fontsize=ft_size)             

        #----- Stats box
        ax_list[1, 1].scatter([0.0, 1.], [0.0, 1.], c="none")
        if self.warm_up_num > 0:
            ax_list[1, 1].text(0.1, 0.8, "RA before warm-up: %.3f" % (self.accept_R_warm_up), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.7, "RA after warm-up: %.3f" % (self.accept_R), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.6, "Total time: %.1f s" % self.dt_total, fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.5, "Total steps: %.2E" % self.N_total_steps, fontsize=ft_size2)        
        ax_list[1, 1].text(0.1, 0.4, "Ntot/eff med: %.1E/%.1E" % (self.L_chain * self.Nchain, np.median(self.n_eff_q)), fontsize=ft_size2)                
        ax_list[1, 1].text(0.1, 0.3, "#steps/ES med: %.2E" % (self.N_total_steps/np.median(self.n_eff_q)), fontsize=ft_size2)                
        ax_list[1, 1].text(0.1, 0.2, "#steps/ES best: %.2E" % (self.N_total_steps/np.max(self.n_eff_q)), fontsize=ft_size2)                
        ax_list[1, 1].text(0.1, 0.1, "#steps/ES worst: %.2E" % (self.N_total_steps/np.min(self.n_eff_q)), fontsize=ft_size2)                
        ax_list[1, 1].set_xlim([0, 1])
        ax_list[1, 1].set_ylim([0, 1])

        plt.suptitle("D/Nchain/Niter/Warm-up/Thin = %d\%d\%d\%d\%d" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate), fontsize=ft_size_title)
        if savefig:
            fname = title_prefix+"samples-D%d-Nchain%d-Niter%d-Warm%d-Thin%d.png" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate)
            plt.savefig(fname, dpi=100, bbox_inches = "tight")
        if show:
            plt.show()
        plt.close()

        



class HMC_sampler(sampler):
    """
    HMC sampler for a general distribution.
    
    The main user functions are the constructor and the gen_sample.
    """
    
    def __init__(self, D, V, dVdq, Nchain=2, Niter=1000, thin_rate=1, warm_up_num=0, \
                 cov_p=None, sampler_type="Fixed", L=None, global_dt = True, dt=None, \
                 L_low=None, L_high=None, log2L=None, d_max=10):
        """
        Args: See Sampler class constructor for other variables.
        - D: Dimensin of inference.
        - V: The potential function which is the negative lnL.
        - dVdq: The gradient of the potential function to be supplied by the user. Currently, numerical gradient is not supported.
        - cov_p: Covariance matrix of the momentum distribution assuming Gaussian momentum distribution.
        - global_dt: If True, then a single uniform time step is used for all variables. Otherwise, one dt for each variable is used.
        - dt: Time step(s).        
        - L: Number of steps to be taken for each sample if sampler type is "Fixed". 
        - L_low, L_high: If "Random" sampler is chosen, then vary the trajectory length as a random sample from [L_low, L_high]. 
        - log2L: log base 2 of trajectory length for the static scheme        
        - sampler_type = "Fixed", "Random", "Static", or "NUTS"; 
        - d_max: Maximum number of doubling allowed by the user.

        Note: "Fixed" and "Static" is no longer supported.
        """
        # parent constructor
        sampler.__init__(self, D=D, target_lnL=None, Nchain=Nchain, Niter=Niter, thin_rate=thin_rate, warm_up_num=warm_up_num)
        
        # Save the potential and its gradient functions
        self.V = V
        self.dVdq = dVdq
        
        # Which sampler to use?
        assert (sampler_type=="Fixed") or (sampler_type=="Random") or (sampler_type=="NUTS") or (sampler_type=="Static")
        assert (dt is not None)
        self.dt = dt
        self.global_dt = global_dt 
        self.sampler_type = sampler_type
        if self.sampler_type == "Fixed":
            assert (L is not None)
            self.L = L
        elif self.sampler_type == "Random":
            assert (L_low is not None) and (L_high is not None)
            self.L_low = L_low
            self.L_high = L_high
        elif self.sampler_type == "Static":
            assert (log2L is not None)
            self.log2L =log2L
        elif self.sampler_type == "NUTS":
            assert d_max is not None
            self.d_max = d_max


        # Momentum covariance matrix and its inverse
        if cov_p is None: 
            self.cov_p = np.diag(np.ones(self.D))
        else:
            self.cov_p = cov_p                    
        self.inv_cov_p = np.linalg.inv(self.cov_p)             
        
        # Save marginal energy and energy difference.
        self.E_chain = np.zeros((self.Nchain, self.L_chain, 1), dtype=np.float)
        self.dE_chain = np.zeros((self.Nchain, self.L_chain, 1), dtype=np.float)
        
        
    def gen_sample(self, q_start, N_save_chain0=0, verbose=True):
        """
        Save Nchain of (Niter-warm_up_num+1)//thin_rate samples.
        Also, record acceptance/rejection rate before and after warm-up.
        
        Appropriate samplers "Random" and "NUTS". Others not supported.
        
        Args:
        - q_start: The function takes in the starting point as an input.
        This requirement gives the user a fine degree of choice on how to
        choose the starting point. Dimensions (self.Nchain, D)
        - N_save_chain0: (# of samples - 1) to save from chain0 from the beginning in order to produce a video.
        - verbose: If true, then print how long each chain takes.
        """
        
        if (self.sampler_type == "Random"):
            self.gen_sample_random(q_start, N_save_chain0, verbose)
        elif (self.sampler_type == "NUTS"):
            self.gen_sample_NUTS(q_start, N_save_chain0, verbose)
            
        return



    def gen_sample_random(self, q_start, N_save_chain0, verbose):
        """
        Random trajectory length sampler.

        Same arguments as gen_sample.
        """
    
        #---- Param checking/variable construction before run
        # Check if the correct number of starting points have been provided by the user.
        assert q_start.shape[0] == self.Nchain
        if (N_save_chain0 > 0):
            save_chain = True
            self.decision_chain = np.zeros((self.N_save_chain0+1, 1), dtype=np.int)
            self.phi_q = [] # List since the exact dimension is not known before the run.
        else:
            save_chain = False
            
        # Variables for computing acceptance rate
        accept_counter_warm_up = 0
        accept_counter = 0
        
        #---- Executing HMC ---- #
        # Report time for computing each chain.
        for m in xrange(self.Nchain): # For each chain            
            # ---- Initializing the chain
            # Take the initial value: We treat the first point to be accepted without movement.
            self.q_chain[m, 0, :] = q_start[m]
            q_tmp = q_start[m]
            p_tmp = self.p_sample()[0] # Sample momentun
            E_initial = self.E(q_tmp, p_tmp)
            self.N_total_steps += 1 # Energy calculation has likelihood evaluation.
            self.E_chain[m, 0, 0] = E_initial
            self.dE_chain[m, 0, 0] = 0 # There is no previous momentum so this is zero.
            E_previous = E_initial # Initial energy            

            #---- Start measuring time
            if verbose:
                print "Running chain %d" % m                
                start = time.time()                   

            #---- Execution starts here
            for i in xrange(1, self.Niter+1): # According to the convention we are using.
                # Initial position/momenutm
                q_initial = q_tmp # Saving the initial point
                p_tmp = self.p_sample()[0] # Sample momentun

                # Compute initial energy and save
                E_initial = self.E(q_tmp, p_tmp)
                self.N_total_steps += 1
                if i >= self.warm_up_num: # Save the right cadence of samples.
                    self.E_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_initial
                    self.dE_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_initial - E_previous                    
                    
                # Draw the length of the trajectory
                L_random = np.random.randint(low=self.L_low, high=self.L_high, size=1)[0]
                if save_chain and (m==0) and (i<(N_save_chain0+1)):
                    # Construct an array of length L_random+1 and save the initial point at 0.
                    phi_q_tmp = np.zeros((L_random+1, self.D))
                    phi_q_tmp[0, :] = q_tmp

                # Take leap frog steps
                for l in xrange(1, L_random+1):
                    p_tmp, q_tmp = self.leap_frog(p_tmp, q_tmp)
                    self.N_total_steps += L_random * self.D                    
                    if save_chain and (m==0) and (i<(N_save_chain0+1)):
                        phi_q_tmp[l, :] = q_tmp

                # Compute final energy and save.
                E_final = self.E(q_tmp, p_tmp)
                self.N_total_steps += 1                               
                    
                # With correct probability, accept or reject the last proposal.
                dE = E_final - E_initial
                E_previous = E_initial # Save the energy so that the energy differential can be computed during the next run.
                lnu = np.log(np.random.random(1))        
                if (dE < 0) or (lnu < -dE): # If accepted.
                    if save_chain and (m==0) and (i<(N_save_chain0+1)):
                        self.decision_chain[i, 0] = 1
                        self.phi_q.append(phi_q_tmp)
                    if i >= self.warm_up_num: # Save the right cadence of samples.
                        self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_tmp # save the new point
                        accept_counter +=1                            
                    else:
                        accept_counter_warm_up += 1                        
                else: # Otherwise, proposal rejected.
                    self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_initial # save the old point
                    q_tmp = q_initial
            
            #---- Finish measuring time
            if verbose:
                dt = time.time() - start
                self.dt_total += dt
                print "Time taken: %.2f\n" % dt 

        print "Compute acceptance rate"
        if self.warm_up_num > 0:
            self.accept_R_warm_up = accept_counter_warm_up / float(self.Nchain * self.warm_up_num)
            print "During warm up: %.3f" % self.accept_R_warm_up            
        self.accept_R = accept_counter / float(self.Nchain * (self.Niter - self.warm_up_num + 1))
        print "After warm up: %.3f" % self.accept_R
        print "Completed."            

        return 
  


    def gen_sample_NUTS(self, q_start, N_save_chain0, verbose):
        """
        The same as *_static (no longer supported) except trajectory length is determined dynamically
        and pathological sub-trajectories are rejected (not included in the sampling).
        That is, stop expansion process if any sub-tree of the new sub-trajectory meets the termination criteria. 

        Notes on saving: Since NUTS scheme expands the trajectory forward and backward randomly,
        for each iteration, we create a list and keep appending sub-trajetories that are 
        represented as a list.
        
        See gen_sample "args"
        """
    
        #---- Param checking/variable construction before run
        # Check if the correct number of starting points have been provided by the user.
        assert q_start.shape[0] == self.Nchain
        if (N_save_chain0 > 0):
            save_chain = True
            # self.decision_chain = np.zeros((self.N_save_chain0+1, 1), dtype=np.int) # Unused feature in NUTS sampler.
            self.phi_q = [] # List since the exact dimension is not known before the run.
        else:
            save_chain = False

        #---- Arrays in which to save intermediate points
        q_save = np.zeros((self.d_max+1, self.D), dtype=float)
        p_save = np.zeros((self.d_max+1, self.D), dtype=float)

        #---- Energies and canonical sums used to perform sampling
        E_max_old = 0
        E_max_new_now = 0
        E_max_new_previous = 0
        pi_old = 0 # Sum of exponentials in the old trajectory divided by e^-E_max_old
        pi_new = 0 # Sum of exponentials in the new trajectory divided by e^-E_max_now (after update)
        
        #---- Executing HMC
        # Report time for computing each chain.
        for m in xrange(self.Nchain): # For each chain            
            # ---- Initializing the chain
            # Take the initial value: We treat the first point to be accepted without movement.
            self.q_chain[m, 0, :] = q_start[m]
            q_tmp = q_start[m]
            p_tmp = self.p_sample()[0] # Sample momentun
            E_initial = self.E(q_tmp, p_tmp)
            self.N_total_steps += 1 # Energy calculation has likelihood evaluation.
            self.E_chain[m, 0, 0] = E_initial
            self.dE_chain[m, 0, 0] = 0 # There is no previous momentum so this is zero.
            E_previous = E_initial # Initial energy            

            #---- Start measuring time
            if verbose:
                print "Running chain %d" % m                
                start = time.time()                   

            #---- Execution starts here ----#
            for i in xrange(1, self.Niter+1): # For each step
                #---- Sample momentun
                p_tmp = self.p_sample()[0] 
                # q_tmp is the initial point set from the last run.

                #---- Compute initial energy and save
                E_initial = self.E(q_tmp, p_tmp)
                self.N_total_steps += 1
                if i >= self.warm_up_num: # Save the right cadence of samples.
                    self.E_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_initial
                    self.dE_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_initial - E_previous                    
                    
                #---- Set up necessary variables
                # Live points from the old trajectory
                live_point_q_old = q_tmp
                live_point_p_old = p_tmp
                # Live points from the new trajectory
                live_point_q_new = None
                live_point_p_new = None
                # Left and right boundary points -- Initial point is both left and right boundary points initially.
                left_q = q_tmp
                left_p = -p_tmp
                right_q = q_tmp
                right_p = p_tmp
                # The initial point
                E_max_old = E_initial
                pi_old = 1

                # Continue doubling the trajectory until termination conditions are reached.
                left_terminate = False
                right_terminate = False

                #---- Index table
                # Used to indicate where intermediate boundary points are saved q_save, p_save.
                # Note that leftmost and rightmost boundary points are saved separately.
                # The index table is used purely for checking sub-trajectory termination.
                # Initially all -1. If not -1, then holds the point number m. The corresponding
                # array index is the save index. This scheme requires manual release.
                save_index_table = np.ones(self.d_max+1, dtype=int) * -1

                #---- Main sampling occurs here.
                d = 0
                while (~left_terminate) or (~right_terminate): # While the termination condition hasn't been met.
                    if d > self.d_max-1:
                        print "Doubling number d exceeds d_max = %d" % self.d_max
                        assert False

                    # Refresh the index table
                    save_index_table[:] = -1

                    # Compute the length of the new sub trajectory
                    L_new_sub = 2**d

                    # Random draw of the expansion direction.
                    # If 0, integrate forward. Else integrate backward.                    
                    u_dir = np.random.randint(low=0, high=2, size=1)[0]

                    # Initial point of the new sub-trajectory
                    if u_dir == 0: # Integrate forward
                        p_tmp, q_tmp = self.leap_frog(right_p, right_q)
                    else: # Integrate backward
                        p_tmp, q_tmp = self.leap_frog(left_p, left_q)
                    self.N_total_steps += self.D

                    live_point_q_new, live_point_p_new = q_tmp, p_tmp
                    E_max_new_now = self.E(q_tmp, p_tmp) # New sub-trajectory initial and maximum energy
                    pi_new = 1
                    self.N_total_steps += 1

                    # Saving the initial point
                    save_index = find_next(save_index_table)
                    q_save[save_index, :] = live_point_q_new
                    p_save[save_index, :] = live_point_p_new
                    save_index_table[save_index] = 1 # Note 1-indexing convention.
                    
                    # For rejecting the whole trajectory                    
                    trajectory_reject = False 

                    # Constructing the new trajectory with progressive updating.
                    # Only if the new trajectory length is greater than 1
                    if L_new_sub > 1:
                        # Uniform progressive sampling: In each step, sample a uniform number u ~[0, 1] and compare
                        # to r = sum_i=1^k-1 pi(z_i) / sum_i=1^k pi(z_i). If u > r take the new point as the live point.
                        # Else retain the old point.                    
                        for k in xrange(1, L_new_sub):
                            # Step forward                            
                            p_tmp, q_tmp = self.leap_frog(p_tmp, q_tmp) 
                            self.N_total_steps += self.D                            

                            # Compute new energy 
                            E_tmp = self.E(q_tmp, p_tmp)
                            self.N_total_steps += 1


                            # # If the energy difference is too large then reject the trajectory.                            
                            # if np.abs(E_tmp - E_initial) > 1000: 
                            #     trajectory_reject = True
                            #     q_tmp = live_point_q_old
                            #     break

                            # Check the termination criteria within the sub-trajectory so far.
                            if ((k+1) % 2) == 1: # If odd point, then save.
                                save_index = find_next(save_index_table)
                                q_save[save_index, :] = q_tmp # Current point
                                p_save[save_index, :] = p_tmp 
                                save_index_table[save_index] = k+1                                
                                # ---- Debug line
                                # print "save", k+1, save_index
                                # assert save_index >= 0                                 
                            else: 
                                # Check termination conditions against each point calculated
                                # according to the rule.
                                check_pts = check_points(k+1)
                                for l in check_pts:
                                    # Retrieve a previous points
                                    save_index = retrieve_save_index(save_index_table, l)

                                    # ---- Debug line
                                    # print "check", k+1, l, save_index
                                    # assert save_index >= 0                                 

                                    q_check = q_save[save_index, :] 
                                    p_check = p_save[save_index, :]

                                    # Check termination condition
                                    if u_dir == 0: # Forward
                                        left_q_tmp, left_p_tmp = q_check, -p_check
                                        right_q_tmp, right_p_tmp = q_tmp, p_tmp
                                    else:                                        
                                        left_q_tmp, left_p_tmp = q_tmp, p_tmp
                                        right_q_tmp, right_p_tmp = q_check, -p_check
                                    Dq_tmp = right_q_tmp - left_q_tmp
                                    
                                    right_terminate_tmp = np.dot(Dq_tmp, right_p_tmp) < 0
                                    left_terminate_tmp = np.dot(-Dq_tmp, left_p_tmp) < 0

                                    if left_terminate_tmp and right_terminate_tmp:
                                        # If the termination condition is satisfied by any subtree
                                        # reject the whole trajectory expansion.
                                        trajectory_reject = True
                                        q_tmp = live_point_q_old
                                        break 

                                    # If the point is no longer needed, then release the space.     
                                    if (l>1) and release(k+1, l):
                                        save_index_table[save_index] = -1                                

                            #---- If the sub-trajectory is rejected, then break out of the sub-trajectory expansion.
                                if trajectory_reject: # This is necessary.
                                    break
                            if trajectory_reject:
                                break

                            # If the sub-trajectory expansion is approved, then save the energy.
                            E_max_new_previous = E_max_new_now
                            E_max_new_now = max(E_max_new_previous, E_tmp)
                            numerator = np.exp(-(E_tmp-E_max_new_now))
                            pi_new = numerator + np.exp(E_max_new_now-E_max_new_previous) * pi_new
                            r = numerator/pi_new
                            u = np.random.random() # Draw random uniform [0, 1]
                            if u < r:
                                # Update the live point.
                                live_point_q_new, live_point_p_new = q_tmp, p_tmp
                    
                    # If the sub-trajectory is rejected, stop NUTS
                    if trajectory_reject: 
                        break

                    # Update the boundary point if last                        
                    if u_dir == 0: 
                        right_q, right_p = q_tmp, p_tmp
                    else: # Integrate backward
                        left_q, left_p = q_tmp, p_tmp

                    # Biased trajectory sampling    
                    # - Perform a biased trajectory sampling and keep one live point. 
                    # - Bernouli sampling with min(1, w_new/w_old) for the new trajectory. 
                    r = np.exp(-(E_max_new_now-E_max_old)) * pi_old/pi_new
                    # Update pi_old
                    E_max_old_previous = E_max_old
                    E_max_old = max(E_max_old_previous, E_max_new_now)

                    pi_old = np.exp(-(E_max_new_now-E_max_old)) * pi_new + np.exp(-(E_max_old_previous-E_max_old)) * pi_old
                    A = min(1, r)
                    u = np.random.random() # Draw random uniform [0, 1]                
                    if u < A:
                        live_point_q_old = live_point_q_new
                    q_tmp = live_point_q_old

                    # Check for termination condition
                    Dq = right_q - left_q
                    right_terminate = np.dot(Dq, right_p) < 0
                    left_terminate = np.dot(-Dq, left_p) < 0

                    # Next doubling length 2**d
                    d +=1

                # Saving the energy for dE calculation.
                E_previous = E_initial

                # Dynamic loop was exited. Save the right cadence of samples.
                if i >= self.warm_up_num:
                    self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_tmp 
            
            #---- Finish measuring time
            if verbose:
                dt = time.time() - start
                self.dt_total += dt
                print "Time taken: %.2f\n" % dt 

        #---- Finally tally before completion.
        print "Compute acceptance rate: By default equal to 1."
        if self.warm_up_num > 0:
            self.accept_R_warm_up = 1.
            print "During warm up: %.3f" % self.accept_R_warm_up            
        self.accept_R = 1. 
        print "After warm up: %.3f" % self.accept_R
        print "Completed."            

        return         


    def K(self, p):
        """
        The user supplies potential energy and its gradient.
        User also sets the form of kinetic distribution.
        Kinetic energy -ln P(p)
        """
        return np.dot(p, np.dot(self.inv_cov_p, p)) / 2.

    def E(self, q, p):
        """
        Kinetic plus potential energy    
        """
        return self.V(q) + self.K(p)

    def p_sample(self):
        """
        Return a random sample from a unit multi-variate normal of dimension D
        """
        return np.random.multivariate_normal(np.zeros(self.D), self.cov_p, size=1)
        
    def leap_frog(self, p_old, q_old):
        """
        dVdq is the gradient of poential function supplied by the user.
        """
        p_half = p_old - self.dt * np.dot(self.inv_cov_p,  self.dVdq(q_old)) / 2.
        q_new = q_old + self.dt * p_half 
        p_new = p_half - self.dt * np.dot(self.inv_cov_p,  self.dVdq(q_new)) / 2. 

        return p_new, q_new
    

    
    def make_movie(self, T_last=-1, warm_up=False, fname_prefix="test-HMC", dx =0.1, xmax=4, plot_normal=True, q0=None, cov0=False, plot_cov=True):
        """
        Creates a deck of png files that can be turned into a movie.
        
        Note each slide is indexed by the iteration number, i, and
        the time step number l = 0, ..., L. 
        
        Args:
        - T_last: The duration of the movie in units of MH step. 
        If T_last exceeds the total number of steps, then maximum is used.
        - warm_up: If True, start making the movie after the warm-up fraction.
        """
        
        # Setting the beginning and the final index.
        if warm_up:
            idx_start = self.warm_up_num
        else:
            idx_start = 0
            
        if T_last > self.Niter:
            idx_last = self.Niter
        else:
            idx_last = T_last
            
        # Make a movie
        counter = 0
        for i in xrange(idx_start, idx_last):
            for l in xrange(0, self.L+1): # For each iteration, plot the whole trajectory.
                idx_slide = (i-idx_start) * (self.L+1) + l                
                self.make_slide(fname_prefix, idx_start, i, l, idx_slide, plot_normal=plot_normal, q0=q0, cov0=cov0, plot_cov=plot_cov, Ntotal=(idx_last-idx_start+1), xmax=xmax, dx=dx)
                if (counter % 10) == 0:
                    print "%d: Making slide (%d, %d)" % (counter, i, l)
                counter +=1
        # Command for creating and movies and deleting slides.
        print "Use the following command to make a movie:\nffmpeg -r 1 -start_number 0 -i %s-%%d.png -vcodec mpeg4 -y %s-movie.mp4"  % (fname_prefix, fname_prefix)
        
        return 
    
    def make_slide(self, fname_prefix, idx_chain_start, idx_chain, l, idx_slide, dx =0.1, xmax=4, plot_normal=True, q0=None, cov0=False, plot_cov=True, Ntotal=100):
        """
        Make a slide for the movie.
        Args:
        - fname_prefix: 
        - idx_chain: Slide index.
        - idx_chain_start: Start index.
        - l: The trajectory time stem number.
        - idx_slide: Index to be appended to the slide.
        - Ntotal: Total number of samples to be plotted
        
        Trajectory that gets accepted is colored red, otherwise black.
        """
        pt_size_previous = 20 # Accepted points between idx_start and now.
        pt_size_current = pt_size_proposed = 50
        pt_size_phi = 30 # Size of the trajectory bubble.
        
        
        # Take q1 and q2 that have been accepted so far.
        q1 = self.phi_q[idx_chain_start:idx_chain, 0, 0]
        q2 = self.phi_q[idx_chain_start:idx_chain, 0, 1]
        
        # Setting boundary and bin size
        xmin = -xmax
        
        # If plotting normal model is requested
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)
            assert q0.size == self.D
            xgrid = np.arange(xmin, xmax, 1e-2)
            q1_marginal = multivariate_normal.pdf(xgrid, mean=q0[0], cov=cov0[0, 0]) * (Ntotal) * dx 
            q2_marginal = multivariate_normal.pdf(xgrid, mean=q0[1], cov=cov0[1, 1]) * (Ntotal) * dx

        fig, ax_list = plt.subplots(2, 2, figsize = (12, 12))
        ft_size = 20
        # ---- Scatter plot
        # All previous points so far
        ax_list[0, 0].scatter(q1, q2, s=pt_size_previous, c="black", edgecolors="none")
        
        # Current point 
        ax_list[0, 0].scatter([self.phi_q[idx_chain, 0, 0]], [self.phi_q[idx_chain, 0, 1]], c="red", edgecolors="none", s=pt_size_current)
        
        # Current trajectory
        if self.decision_chain[idx_chain, 0] == 1: # If the proposal is accepted
            # Empty circle -- red
            ax_list[0, 0].scatter(self.phi_q[idx_chain, :l, 0], self.phi_q[idx_chain, :l, 1], c="none", edgecolors="red", s=pt_size_phi)
            ax_list[0, 0].plot(self.phi_q[idx_chain, :l, 0], self.phi_q[idx_chain, :l, 1], c="red", ls="--", lw=1) 
        else:
            # Empty circle -- black
            ax_list[0, 0].scatter(self.phi_q[idx_chain, :l, 0], self.phi_q[idx_chain, :l, 1],  c="none", edgecolors="black", s=pt_size_phi)
            ax_list[0, 0].plot(self.phi_q[idx_chain, :l, 0], self.phi_q[idx_chain, :l, 1], c="black", ls="--", lw=1) 
                
        
        
        if plot_cov:
            plot_cov_ellipse(ax_list[0, 0], [q0], [cov0], 0, 1, MoG_color="Blue")
        ax_list[0, 0].set_xlabel("q1", fontsize=ft_size)
        ax_list[0, 0].set_ylabel("q2", fontsize=ft_size)
        ax_list[0, 0].axis("equal")
        ax_list[0, 0].set_xlim([-xmax, xmax])
        ax_list[0, 0].set_ylim([-xmax, xmax])

        # q2 histogram
#         if idx_chain_start < idx_chain:
        ax_list[0, 1].hist(q2, bins=np.arange(-xmax, xmax, dx), histtype="step", color="black", orientation="horizontal")
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)
            ax_list[0, 1].plot(q2_marginal, xgrid, c="green", lw=2)
        ax_list[0, 1].set_ylim([-xmax, xmax])
        ax_list[0, 1].set_xlim([0, q1_marginal.max() * 1.5])        
        ax_list[0, 1].set_ylabel("q2", fontsize=ft_size)
        # q1 histogram
#         if idx_chain_start < idx_chain:        
        ax_list[1, 0].hist(q1, bins=np.arange(-xmax, xmax, dx), histtype="step", color="black")
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)            
            ax_list[1, 0].plot(xgrid, q1_marginal, c="green", lw=2)
        ax_list[1, 0].set_xlim([-xmax, xmax])
        ax_list[1, 0].set_ylim([0, q1_marginal.max() * 1.5])                
        ax_list[1, 0].set_xlabel("q1", fontsize=ft_size)

        # Show stats
        ft_size2 = 15
        ax_list[1, 1].scatter([0.0, 1.], [0.0, 1.], c="none")
        ax_list[1, 1].text(0.1, 0.8, "D/Nchain/Niter/Warm-up/Thin", fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.7, "%d\%d\%d\%d\%d" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate), fontsize=ft_size2)        
        ax_list[1, 1].text(0.1, 0.6, r"R q1/q2: %.3f/%.3f" % (self.R_q[0], self.R_q[1]), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.5, "R median, std: %.3f/ %.3f" % (np.median(self.R_q), np.std(self.R_q)), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.4, "Accept rate before warm up: %.3f" % (self.accept_R_warm_up), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.3, "Accept rate after warm up: %.3f" % (self.accept_R), fontsize=ft_size2)        
        ax_list[1, 1].set_xlim([0, 1])
        ax_list[1, 1].set_ylim([0, 1])
        
        fname = fname_prefix + ("-%d" % idx_slide)
        plt.suptitle("T_MH = %d (T_HMC = %d)" % ((idx_chain * (self.L) + l) * self.D, idx_chain), fontsize=25)
        plt.savefig(fname, dpi=100, bbox_inches = "tight")
        plt.close()       


    # def gen_sample_fixed(self, q_start, save_chain, verbose):
    #     """
    #     - save_chain: If True, in addition to saving all warmed-up and thinned samples,
    #     save in an array the following (only the first chain):
    #     1) phi_q (Niter, L, D): The full trajectory starting with the initial.
    #     2) decision_chain (Niter, 1): Whether the proposal was accepted or not.
    #     """
    
    #     # Check if the correct number of starting points have been        
    #     assert q_start.shape[0] == self.Nchain
    
    #     if (save_chain):
    #         self.decision_chain = np.zeros((self.Niter, 1), dtype=np.int)
    #         self.phi_q = np.zeros((self.Niter, self.L+1, self.D), dtype=np.float)
    #         # +1 because we want to count the initial AND final.
            
    #     # Variables for computing acceptance rate
    #     accept_counter_warm_up = 0
    #     accept_counter = 0
        
    #     # Executing HMC
    #     for m in xrange(self.Nchain): # For each chain
    #         start = time.time()   
            
    #         # Initializing the chain
    #         if verbose:
    #             print "Running chain %d" % m                
            
    #         # Take the initial value: We treat the first point to be accepted without movement.
    #         self.q_chain[m, 0, :] = q_start[m]
    #         q_initial = q_start[m]
    #         p_tmp = self.p_sample()[0] # Sample momentun
    #         self.E_chain[m, 0, 0] = self.E(q_initial, p_tmp)
    #         self.Eq_chain[m, 0, 0] = self.K(p_tmp)
    #         # Save the initial point of the trajectory if asked for.
    #         if save_chain and (m==0):
    #             self.phi_q[0, 0, :] = q_initial            

    #         for i in xrange(self.Niter): # For each step
    #             # Initial position/momenutm
    #             q_tmp = q_initial
    #             p_tmp = self.p_sample()[0] # Sample momentun

    #             # Compute kinetic initial energy and save
    #             K_initial = self.K(p_tmp)
                
    #             # Compute the initial Energy
    #             E_initial = self.E(q_tmp, p_tmp)
                
    #             # Save the initial point of the trajectory if asked for.
    #             if save_chain and (m==0):
    #                 self.phi_q[i, 0, :] = q_initial
                    
    #             # Perform HMC integration and save the trajectory if requested.
    #             for l in xrange(1, self.L+1):
    #                 p_tmp, q_tmp = self.leap_frog(p_tmp, q_tmp)
    #                 if save_chain and (m == 0):
    #                     self.phi_q[i, l, :] = q_tmp

    #             # Compute final energy and save.
    #             E_final = self.E(q_tmp, p_tmp)
                
    #             # Save Kinetic energy
    #             if i >= self.warm_up_num: # Save the right cadence of samples.
    #                 self.Eq_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = K_initial
    #                 self.E_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_final
                    
    #             # With correct probability, accept or reject the last proposal.
    #             dE = E_final - E_initial
    #             lnu = np.log(np.random.random(1))        
    #             if (dE < 0) or (lnu < -dE): # If accepted.
    #                 q_initial = q_tmp
    #                 if (save_chain) and (m==0):
    #                     self.decision_chain[i, 0] = 1
    #                 if i >= self.warm_up_num: # Save the right cadence of samples.
    #                     self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_tmp
    #                     accept_counter +=1                            
    #                 else:
    #                     accept_counter_warm_up += 1                        
    #             else: # Otherwise proposal rejected.
    #                 self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_initial
            
    #         dt = time.time() - start
    #         print "Time taken: %.2f\n" % dt 

    #     print "Compute acceptance rate"
    #     self.accept_R_warm_up = accept_counter_warm_up / float(self.Nchain * self.warm_up_num)
    #     self.accept_R = accept_counter / float(self.Nchain * (self.Niter - self.warm_up_num))
    #     print "During warm up: %.3f" % self.accept_R_warm_up
    #     print "After warm up: %.3f" % self.accept_R
    #     print "Complete"            

    #     return 

    # def gen_sample_static(self, q_start, save_chain, verbose):
    #     """

    #     Efficient sampling approach with
    #     - Doubling of trajectory until the total length becomes L;
    #     - From each new segment of the trajectory, perform uniform progressive sampling; and
    #     - Perform biased trajectory sampling.

    #     An example iteration:
    #     - Start with the initial point. Trajectory length is 1. The initial point is the live point.
    #     Compute the energy H(z) and pi(z) = e^-H(z).
    #     - Flip a coin and double the trajectory backward or forward. As you compute the trajectory
    #     keep a cumulative sum of pi(z) for the whole trajectory, progressively updating the live point
    #     for the new trajectory. Save the last point of the trajectory as well. 
    #         - Uniform progressive sampling: In each step, sample a uniform number u ~[0, 1] and compare
    #         to r = sum_i=1^k-1 pi(z_i) / sum_i=1^k pi(z_i). If u > r take the new point as the live point.
    #         Else retain the old point.
    #     - Perform a biased trajectory sampling and keep one live point:
    #         - Bernouli sampling with min(1, w_new/w_old) for the new trajectory. 
    #     - Repeat above two steps until the desired trajectory length is reached.
    #     - In each step, we keep track of the live point from old trajectory, the live point from the 
    #     new trajectory, the most backward or forward point. Hence one keeps at most three points.

    #     - save_chain (currently not supported): If True, in addition to saving all warmed-up and thinned samples,
    #     save in an array the following (only the first chain):
    #     1) phi_q (Niter, L, D): The full trajectory starting with the initial.
    #     2) decision_chain (Niter, 1): Whether the proposal was accepted or not.
    #     """
    
    #     # Check if the correct number of starting points have been        
    #     assert q_start.shape[0] == self.Nchain
    
    #     # if (save_chain):
    #     #     assert False
    #     #     self.decision_chain = np.zeros((self.Niter, 1), dtype=np.int)
    #     #     self.phi_q = np.zeros((self.Niter, self.L+1, self.D), dtype=np.float)
    #         # +1 because we want to count the initial AND final.
            
    #     # Variables for computing acceptance rate
    #     accept_counter_warm_up = 0
    #     accept_counter = 0
        
    #     # Executing HMC
    #     for m in xrange(self.Nchain): # For each chain
    #         start = time.time()   
            
    #         # Initializing the chain
    #         if verbose:
    #             print "Running chain %d" % m                
            
    #         # Initial points
    #         q_tmp = q_start[m]

    #         # self.E_chain[m, 0, 0] = self.E(q_initial, p_tmp)
    #         # self.Eq_chain[m, 0, 0] = self.K(p_tmp)
    #         # Save the initial point of the trajectory if asked for.
    #         # if save_chain and (m==0):
    #         #     assert False
    #         #     self.phi_q[0, 0, :] = q_initial            

    #         for i in xrange(self.Niter): # For each step
    #             # Momentum resampling
    #             p_tmp = self.p_sample()[0] # Sample momentun

    #             # Compute kinetic initial energy and save
    #             K_initial = self.K(p_tmp)
                
    #             # Compute the initial Energy
    #             E_initial = self.E(q_tmp, p_tmp)
                
    #             # # Save the initial point of the trajectory if asked for.
    #             # if save_chain and (m==0):
    #             #     assert False
    #             #     self.phi_q[i, 0, :] = q_initial
                    
    #             # Perform HMC integration and save the trajectory if requested.
    #             # Live points from the old trajectory
    #             live_point_q_old = q_tmp
    #             live_point_p_old = p_tmp
    #             # Live points from the new trajectory
    #             live_point_q_new = None
    #             live_point_p_new = None
    #             # Left and right boundary point -- Initial point is both left and right boundary points initially.
    #             left_q = q_tmp
    #             left_p = -p_tmp
    #             right_q = q_tmp
    #             right_p = p_tmp
    #             # w = sum_z pi_z
    #             Es_old = np.array([E_initial]) # We save all the energies
    #             Es_new = None # For the new trajectory, we save all of the energies.

    #             # Main sampling occurs here.
    #             # First doubling d == 0 gives new sub-trajectory of length 1.
    #             # Second doubling d == 1 gives new sub-trajectory of length 2. 
    #             # Third, length 4, etc.
    #             # self.single_traj = [q_tmp]
    #             # self.single_traj_live = [live_point_q_old]
    #             for d in xrange(0, self.log2L):
    #                 L_new_sub = 2**d # Length of new sub trajectory
    #                 u_dir = np.random.randint(low=0, high=2, size=1) # If 0, integrate forward. Else integrate backward.

    #                 # Array for saving the energies corresponding to the new sub-trajectory
    #                 Es_new = np.zeros(L_new_sub, dtype=float)

    #                 # Initial point of the trajectory
    #                 if u_dir == 0: # Integrate forward
    #                     p_tmp, q_tmp = self.leap_frog(right_p, right_q)
    #                 else: # Integrate backward
    #                     p_tmp, q_tmp = self.leap_frog(left_p, left_q)
    #                 live_point_q_new, live_point_p_new = q_tmp, p_tmp
    #                 Es_new[0] = self.E(q_tmp, p_tmp)

    #                 # self.single_traj.append(q_tmp)
    #                 # self.single_traj_live.append(live_point_q_new)
    #                 # Constructing the new trajectory with progressive updating.
    #                 # Only if the new trajectory length is greater than 1
    #                 if L_new_sub > 1:
    #                     # Uniform progressive sampling: In each step, sample a uniform number u ~[0, 1] and compare
    #                     # to r = sum_i=1^k-1 pi(z_i) / sum_i=1^k pi(z_i). If u > r take the new point as the live point.
    #                     # Else retain the old point.                    
    #                     for k in xrange(1, L_new_sub):
    #                         p_tmp, q_tmp = self.leap_frog(p_tmp, q_tmp) # Step forward
    #                         Es_new[k] = self.E(q_tmp, p_tmp) # Compute new energy
    #                         u = np.random.random() # Draw random uniform [0, 1]
    #                         E_max = np.max(Es_new[:k+1])# Get the maximum energy value
    #                         numerator = np.sum(np.exp(-(Es_new[:k]-E_max)))
    #                         denominator = np.sum(np.exp(-(Es_new[:k+1]-E_max)))                            
    #                         r = numerator/denominator# Compute the desired ratio.
    #                         if u > r:
    #                             # Update the live point.
    #                             live_point_q_new, live_point_p_new = q_tmp, p_tmp
    #                         # self.single_traj.append(q_tmp)
    #                         # self.single_traj_live.append(live_point_q_new)
    #                 # Update the boundary point if last                        
    #                 if u_dir == 0: 
    #                     right_q, right_p = q_tmp, p_tmp
    #                 else: # Integrate backward
    #                     left_q, left_p = q_tmp, p_tmp

    #             # Biased trajectory sampling
    #             # - Perform a biased trajectory sampling and keep one live point. 
    #             # - Bernouli sampling with min(1, w_new/w_old) for the new trajectory. 
    #             E_max = max(np.max(Es_new), np.max(Es_old))
    #             # print np.max(Es_new),  np.max(Es_old)
    #             r = np.sum(np.exp(-(Es_new-E_max)))/np.sum(np.exp(-(Es_old-E_max)))
    #             A= min(1, r)
    #             u = np.random.random() # Draw random uniform [0, 1]                
    #             if u < A:
    #                 live_point_q_old = live_point_q_new
    #             q_tmp = live_point_q_old                    
    #             Es_old = np.concatenate((Es_old, Es_new))
    #             Es_new = None

    #             # assert False

    #             # # Compute final energy and save.
    #             # E_final = self.E(q_tmp, p_tmp)
                
    #             # # Save Kinetic energy
    #             # if i >= self.warm_up_num: # Save the right cadence of samples.
    #             #     self.Eq_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = K_initial
    #             #     self.E_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_final
                    
    #             # if (save_chain) and (m==0):
    #             #     self.decision_chain[i, 0] = 1
    #             if i >= self.warm_up_num: # Save the right cadence of samples.
    #                 self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_tmp
    #                 accept_counter +=1                            
    #             else:
    #                 accept_counter_warm_up += 1                        
            
    #         dt = time.time() - start
    #         print "Time taken: %.2f\n" % dt 

    #     print "Compute acceptance rate"
    #     self.accept_R_warm_up = accept_counter_warm_up / float(self.Nchain * self.warm_up_num)
    #     self.accept_R = accept_counter / float(self.Nchain * (self.Niter - self.warm_up_num))
    #     print "During warm up: %.3f" % self.accept_R_warm_up
    #     print "After warm up: %.3f" % self.accept_R
    #     print "Complete"            

    #     return       



# class MH_sampler(sampler):
#     """
#     Metropolis-Hastings sampler for a general distribution.
    
#     The main user functions are the constructor and the gen_sample.
#     """
    
#     def __init__(self, D, target_lnL, Nchain=2, Niter=1000, thin_rate=5, warm_up_frac=0.5, warm_up_num=None, cov_jump=None):
#         """
#         Args: See Sampler class constructor for other variables.
#         - cov_jump: covariance of the jumping proposal. If not, specified then use the "optimal" proposal.
#         """
#         # parent constructor
#         sampler.__init__(self, D=D, target_lnL=target_lnL, Nchain=Nchain, Niter=Niter, thin_rate=thin_rate, warm_up_frac=warm_up_frac, warm_up_num=warm_up_num)
        
#         # Jumping proprosal
#         self.cov_jump = cov_jump        
#         if cov_jump is None: # Proposal covariance
#             self.cov_jump = np.diag(np.ones(D)) * (2.4**2/float(D))
#         else: 
#             self.cov_jump = cov_jump            
        
        
#     def gen_sample(self, q_start, save_chain=False, verbose=True):
#         """
#         Save Nchain of (Niter-warm_up_num)//thin_rate samples.
#         Also, record acceptance/rejection rate before and after warm-up.
        
#         Args:
#         - q_start: The function takes in the starting point as an input.
#         This requirement gives the user a fine degree of choice on how to
#         choose the starting point. Dimensions (self.Nchain, D)
#         - save_chain: If True, then save the entire *first* chain including
#         decision, proposed, and actual.
#         - verbose: If true, then print how long each chain takes.
#         """
        
#         # Check if the correct number of starting points have been
#         assert q_start.shape[0] == self.Nchain
        
#         if save_chain:
#             self.decision_chain = np.zeros((self.Niter, 1), dtype=np.int)
#             self.q_proposed_chain = np.zeros((self.Niter, self.D), dtype=np.float)
#             self.q_initial_chain = np.zeros((self.Niter, self.D), dtype=np.float)
#             self.q_initial_chain[0, :] = q_start[0]# Initializing the chain
            
#         # Variables for computing acceptance rate
#         accept_counter_warm_up = 0
#         accept_counter = 0
        
#         # Executing MHMC            
#         for m in xrange(self.Nchain): # For each chain
#             start = time.time()   
            
#             # Initializing the chain
#             self.q_chain[m, 0, :] = q_start[m]
#             if save_chain and (m==0):
#                 self.q_initial_chain[0, :] = self.q_chain[m, 0, :]
#                 self.q_proposed_chain[0, :] = self.q_chain[m, 0, :]
            
#             if verbose:
#                 print "Running chain %d" % m                
            
#             # Take the initial value
#             q_initial = self.q_chain[m, 0, :]
#             lnL_initial = self.target_lnL(q_initial) 

#             for i in xrange(1, self.Niter): # For each step
#                 # Get the proposal
#                 q_proposed = q_initial + self.proposal()

#                 # Compute log-likelihood differece
#                 lnL_proposed = self.target_lnL(q_proposed)
#                 dlnL = lnL_proposed - lnL_initial

#                 if save_chain and (m==0):
#                     self.q_initial_chain[i, :] = q_initial
#                     self.q_proposed_chain[i, :] = q_proposed
                

#                 # Take the MHMC step according to dlnL
#                 lnu = np.log(np.random.random(1))        
#                 if (dlnL > 0) or (lnu < dlnL): # If the loglikelihood improves, or the dice roll works out
#                     q_initial = q_proposed            
#                     lnL_initial = lnL_proposed
#                     if save_chain and (m==0):
#                         self.decision_chain[i, 0] = 1
                        
#                     if i >= self.warm_up_num: # Save the right cadence of samples.
#                         self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_proposed                        
#                         self.lnL_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = lnL_proposed
#                         accept_counter +=1                            
#                     else:
#                         accept_counter_warm_up += 1
#                 else: # Otherwise proposal rejected.
#                     if i >= self.warm_up_num: # Save the right cadence of samples.                        
#                         self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_initial      
#                         self.lnL_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = lnL_initial
#             dt = time.time() - start
#             print "Time taken: %.2f\n" % dt 
            
#         print "Compute acceptance rate"
#         self.accept_R_warm_up = accept_counter_warm_up / float(self.Nchain * self.warm_up_num)
#         self.accept_R = accept_counter / float(self.Nchain * (self.Niter - self.warm_up_num))
#         print "During warm up: %.3f" % self.accept_R_warm_up
#         print "After warm up: %.3f" % self.accept_R

#         return 

        
    
#     def proposal(self):
#         """
#         Additive proposal. Add returned value to the current point to get
#         the proposal.
#         """
        
#         return np.random.multivariate_normal(np.zeros(self.cov_jump.shape[0]), self.cov_jump, size=1)

    
#     def make_movie(self, T_last=-1, warm_up=False, fname_prefix="test", dx =0.1, xmax=4, plot_normal=True, q0=None, cov0=False, plot_cov=True):
#         """
#         Creates a deck of png files that can be turned into a movie.
        
#         Args:
#         - T_last: The duration of the movie in units of MH step. 
#         If T_last exceeds the total number of steps, then maximum is used.
#         - warm_up: If True, start making the movie after the warm-up fraction.
#         - See others
#         """
        
#         # Setting the beginning and the final index.
#         if warm_up:
#             idx_start = self.warm_up_num
#         else:
#             idx_start = 0
            
#         if T_last > self.Niter:
#             idx_last = self.Niter
#         else:
#             idx_last = T_last
            
#         # Make a movie
#         counter = 0
#         for i in xrange(idx_start, idx_last):
#             if (counter % 10) == 0:
#                 print "%d: Making slide %d" % (counter, i)
#             self.make_slide(fname_prefix, idx_start, i, i - idx_start, plot_normal=plot_normal, q0=q0, cov0=cov0, plot_cov=plot_cov, Ntotal=(idx_last-idx_start+1), xmax=xmax, dx=dx)
#             counter +=1
        
#         # Command for creating and movies and deleting slides.
#         print "Use the following command to make a movie:\nffmpeg -r 1 -start_number 0 -i %s-%%d.png -vcodec mpeg4 -y %s-movie.mp4" % (fname_prefix, fname_prefix)
        
#         return 
    
#     def make_slide(self, fname_prefix, idx_chain_start, idx_chain, idx_slide, dx =0.1, xmax=4, plot_normal=True, q0=None, cov0=False, plot_cov=True, Ntotal=100):
#         """
#         Make a slide for the movie.
#         Args:
#         - fname_prefix: 
#         - idx_chain: Slide index.
#         - idx: Index to be appended to the slide.
#         - Ntotal: Total number of samples to be plotted
#         """
#         pt_size_previous = 20
#         pt_size_current = pt_size_proposed = 50
        
#         # Take samples of the first and second variables
#         q1 = self.q_initial_chain[:, 0]
#         q2 = self.q_initial_chain[:, 1]
#         q1_proposed = self.q_proposed_chain[:, 0]
#         q2_proposed = self.q_proposed_chain[:, 1]
        
        
#         # Setting boundary and bin size
#         xmin = -xmax
        
#         # If plotting normal model is requested
#         if plot_normal:
#             assert (q0 is not None) and (cov0 is not None)
#             assert q0.size == self.D
#             xgrid = np.arange(xmin, xmax, 1e-2)
#             q1_marginal = multivariate_normal.pdf(xgrid, mean=q0[0], cov=cov0[0, 0]) * (Ntotal) * dx 
#             q2_marginal = multivariate_normal.pdf(xgrid, mean=q0[1], cov=cov0[1, 1]) * (Ntotal) * dx

#         fig, ax_list = plt.subplots(2, 2, figsize = (12, 12))
#         ft_size = 20
#         # ---- Scatter plot
#         # All previous points so far
#         ax_list[0, 0].scatter(q1[idx_chain_start:idx_chain], q2[idx_chain_start:idx_chain], s=pt_size_previous, c="black", edgecolors="none")
        
#         # Current proposal
#         if self.decision_chain[idx_chain, 0] == 1: # If the proposal is accepted
#             # Big black dot
#             ax_list[0, 0].scatter([q1_proposed[idx_chain]], [q2_proposed[idx_chain]], c="black", edgecolors="none", s=pt_size_proposed)
#         else:
#             # Empty circle
#             ax_list[0, 0].scatter([q1_proposed[idx_chain]], [q2_proposed[idx_chain]],  c="none", edgecolors="black", s=pt_size_proposed)
        
#         # Initial location
#         ax_list[0, 0].scatter([q1[idx_chain]], [q2[idx_chain]], c="red", s=pt_size_current, edgecolors="none")
        
#         # Line between current proposal and location
#         ax_list[0, 0].plot([q1[idx_chain], q1_proposed[idx_chain]], [q2[idx_chain], q2_proposed[idx_chain]], c="black", ls="--", lw = 1)
        
        
#         if plot_cov:
#             plot_cov_ellipse(ax_list[0, 0], [q0], [cov0], 0, 1, MoG_color="Blue")
#         ax_list[0, 0].set_xlabel("q1", fontsize=ft_size)
#         ax_list[0, 0].set_ylabel("q2", fontsize=ft_size)
#         ax_list[0, 0].axis("equal")
#         ax_list[0, 0].set_xlim([-xmax, xmax])
#         ax_list[0, 0].set_ylim([-xmax, xmax])

#         # q2 histogram
# #         if idx_chain_start < idx_chain:
#         ax_list[0, 1].hist(q2[idx_chain_start:idx_chain], bins=np.arange(-xmax, xmax, dx), histtype="step", color="black", orientation="horizontal")
#         if plot_normal:
#             assert (q0 is not None) and (cov0 is not None)
#             ax_list[0, 1].plot(q2_marginal, xgrid, c="green", lw=2)
#         ax_list[0, 1].set_ylim([-xmax, xmax])
#         ax_list[0, 1].set_xlim([0, q1_marginal.max() * 1.5])        
#         ax_list[0, 1].set_ylabel("q2", fontsize=ft_size)
#         # q1 histogram
# #         if idx_chain_start < idx_chain:        
#         ax_list[1, 0].hist(q1[idx_chain_start:idx_chain], bins=np.arange(-xmax, xmax, dx), histtype="step", color="black")
#         if plot_normal:
#             assert (q0 is not None) and (cov0 is not None)            
#             ax_list[1, 0].plot(xgrid, q1_marginal, c="green", lw=2)
#         ax_list[1, 0].set_xlim([-xmax, xmax])
#         ax_list[1, 0].set_ylim([0, q1_marginal.max() * 1.5])                
#         ax_list[1, 0].set_xlabel("q1", fontsize=ft_size)

#         # Show stats
#         ft_size2 = 15
#         ax_list[1, 1].scatter([0.0, 1.], [0.0, 1.], c="none")
#         ax_list[1, 1].text(0.1, 0.8, "D/Nchain/Niter/Warm-up/Thin", fontsize=ft_size2)
#         ax_list[1, 1].text(0.1, 0.7, "%d\%d\%d\%d\%d" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate), fontsize=ft_size2)        
#         ax_list[1, 1].text(0.1, 0.6, r"R q1/q2: %.3f/%.3f" % (self.R_q[0], self.R_q[1]), fontsize=ft_size2)
#         ax_list[1, 1].text(0.1, 0.5, "R median, std: %.3f/ %.3f" % (np.median(self.R_q), np.std(self.R_q)), fontsize=ft_size2)
#         ax_list[1, 1].text(0.1, 0.4, "Accept rate before warm up: %.3f" % (self.accept_R_warm_up), fontsize=ft_size2)
#         ax_list[1, 1].text(0.1, 0.3, "Accept rate after warm up: %.3f" % (self.accept_R), fontsize=ft_size2)        
#         ax_list[1, 1].set_xlim([0, 1])
#         ax_list[1, 1].set_ylim([0, 1])
        
#         fname = fname_prefix + ("-%d" % idx_slide)
#         plt.suptitle("T_MH = %d" % idx_chain, fontsize=25)
#         plt.savefig(fname, dpi=100, bbox_inches = "tight")
#         plt.close()        
