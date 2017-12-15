from utils import *

class sampler(object):
    """
    Parent class to MH_sampler and HMC_sampler. Contains common functions such as the plot function.
    """
    
    def __init__(self, D, target_lnL, Nchain=2, Niter=1000, thin_rate=5, warm_up_frac=0.5, warm_up_num=None):
        """
        Args:
        - D: Number of paramters to be inferred on.
        - target_lnL: Function that takes in a D-dimensional input q and outputs the log-like of the target distribution.
        - Nchain: Number of chains.
        - Niter: Number of iterations.
        - Thin rate
        - warm_up_frac: Fraction of samples to discard.
        - warm_up_num: Sample index at which to start computing statistics. Ignore warm_up_frac if specified.
        
        None of these variable can be changed. In other words, if you want to change the variables, 
        then one must create a new sampler object as only the appropriate number of samples are retained.        
        """
    
        self.D = D 
        self.target_lnL = target_lnL        
        self.Nchain = Nchain
        self.Niter = Niter
        self.thin_rate = thin_rate
        self.warm_up_frac = warm_up_frac
        self.warm_up_num = warm_up_num
        
        # If user did not fully specify, then automatically set.
        if warm_up_num is None:
            self.warm_up_num = int(Niter * self.warm_up_frac)
        else:
            self.warm_up_num = warm_up_num
            
        # Allocate memory for samples
        self.L_chain = (self.Niter-self.warm_up_num)//self.thin_rate # Length of individual chain
        self.q_chain = np.zeros((self.Nchain, self.L_chain, self.D), dtype=np.float) # Samples
        self.lnL_chain = np.zeros((self.Nchain, self.L_chain, 1)) # loglikelihood
        
        # Stats 
        self.R_q = None # R statistics for each parameter
        self.R_lnL = None # R stats for the loglikelihood.
        self.accept_R_warm_up = None # Acceptance rate during warm up
        self.accept_R = None # Acceptance rate after warm up

        
    def compute_R(self):
        """
        Compute stats on the chain and return.
        
        Remember, the chain has already been warmed up and thinned.
        """

        self.R_q = split_R(self.q_chain, warm_up_frac=0, thin_rate=1)
        self.R_lnL = split_R(self.lnL_chain, warm_up_frac=0, thin_rate=1)

        return
    
    
    def plot_samples(self, show=False, savefig=False, fname=None, xmax = 4, dx =0.1, plot_normal=True, q0=None, cov0=False, plot_cov=True):
        """
        Plot the samples after warm up and thinning.
        Args:
        - xmax: The plot limit. 
        - dx: histogram bin size
        - savefig: Saves figure with the name fname. 
        - fname: If none, then an informative name is assigned. 
        - plot_normal: If true, the user specified normal marginal for q1 and q2 are plotted. 
        - plot_cov: If true, the the user provided
        
        """
        
        # Take samples of the first and second variables
        q_chain_tmp_1 = self.q_chain[:, :, 0].flatten()
        q_chain_tmp_2 = self.q_chain[:, :, 1].flatten()
        
        # Setting boundary and bin size
        xmin = -xmax
        
        # If plotting normal model is requested
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)
            assert q0.size == self.D
            xgrid = np.arange(xmin, xmax, 1e-2)
            q1_marginal = multivariate_normal.pdf(xgrid, mean=q0[0], cov=cov0[0, 0]) * self.L_chain * dx * self.Nchain
            q2_marginal = multivariate_normal.pdf(xgrid, mean=q0[1], cov=cov0[1, 1]) * self.L_chain * dx * self.Nchain

        fig, ax_list = plt.subplots(2, 2, figsize = (12, 12))
        ft_size = 20
        # Scatter plot
        ax_list[0, 0].scatter(q_chain_tmp_1, q_chain_tmp_2, s=1, c="black")
        if plot_cov:
            plot_cov_ellipse(ax_list[0, 0], [q0], [cov0], 0, 1, MoG_color="Blue")
        ax_list[0, 0].set_xlabel("q1", fontsize=ft_size)
        ax_list[0, 0].set_ylabel("q2", fontsize=ft_size)
        ax_list[0, 0].axis("equal")
        ax_list[0, 0].set_xlim([-xmax, xmax])
        ax_list[0, 0].set_ylim([-xmax, xmax])
        # q2 histogram
        ax_list[0, 1].hist(q_chain_tmp_2, bins=np.arange(-xmax, xmax, dx), histtype="step", color="black", orientation="horizontal")
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)
            ax_list[0, 1].plot(q2_marginal, xgrid, c="green", lw=2)
        ax_list[0, 1].set_ylim([-xmax, xmax])
        ax_list[0, 1].set_ylabel("q2", fontsize=ft_size)
        # q1 histogram
        ax_list[1, 0].hist(q_chain_tmp_1, bins=np.arange(-xmax, xmax, dx), histtype="step", color="black")
        if plot_normal:        
            ax_list[1, 0].plot(xgrid, q1_marginal, c="green", lw=2)
        ax_list[1, 0].set_xlim([-xmax, xmax])
        ax_list[1, 0].set_xlabel("q1", fontsize=ft_size)

        # Show stats
        ft_size2 = 15
        ax_list[1, 1].scatter([0.0, 1.], [0.0, 1.], c="none")
        ax_list[1, 1].text(0.1, 0.8, r"R q1/q2: %.3f/%.3f" % (self.R_q[0], self.R_q[1]), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.7, "R median, std: %.3f/ %.3f" % (np.median(self.R_q), np.std(self.R_q)), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.6, "Accept rate before warm up: %.3f" % (self.accept_R_warm_up), fontsize=ft_size2)
        ax_list[1, 1].text(0.1, 0.5, "Accept rate after warm up: %.3f" % (self.accept_R), fontsize=ft_size2)        
        ax_list[1, 1].set_xlim([0, 1])
        ax_list[1, 1].set_ylim([0, 1])

        plt.suptitle("D/Nchain/Niter/Warm-up/Thin = %d\%d\%d\%d\%d" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate), fontsize=25)
        if savefig:
            if fname is None:
                fname = "samples-D%d-Nchain%d-Niter%d-Warm%d-Thin%d.png" % (self.D, self.Nchain, self.Niter, self.warm_up_num, self.thin_rate)
            plt.savefig(fname, dpi=100, bbox_inches = "tight")
        if show:
            plt.show()
        plt.close()

        

class MH_sampler(sampler):
    """
    Metropolis-Hastings sampler for a general distribution.
    
    The main user functions are the constructor and the gen_sample.
    """
    
    def __init__(self, D, target_lnL, Nchain=2, Niter=1000, thin_rate=5, warm_up_frac=0.5, warm_up_num=None, cov_jump=None):
        """
        Args: See Sampler class constructor for other variables.
        - cov_jump: covariance of the jumping proposal. If not, specified then use the "optimal" proposal.
        """
        # parent constructor
        sampler.__init__(self, D=D, target_lnL=target_lnL, Nchain=Nchain, Niter=Niter, thin_rate=thin_rate, warm_up_frac=warm_up_frac, warm_up_num=warm_up_num)
        
        # Jumping proprosal
        self.cov_jump = cov_jump        
        if cov_jump is None: # Proposal covariance
            self.cov_jump = np.diag(np.ones(D)) * (2.4**2/float(D))
        else: 
            self.cov_jump = cov_jump            
        
        
    def gen_sample(self, q_start, save_chain=False, verbose=True):
        """
        Save Nchain of (Niter-warm_up_num)//thin_rate samples.
        Also, record acceptance/rejection rate before and after warm-up.
        
        Args:
        - q_start: The function takes in the starting point as an input.
        This requirement gives the user a fine degree of choice on how to
        choose the starting point. Dimensions (self.Nchain, D)
        - save_chain: If True, then save the entire *first* chain including
        decision, proposed, and actual.
        - verbose: If true, then print how long each chain takes.
        """
        
        # Check if the correct number of starting points have been
        assert q_start.shape[0] == self.Nchain
        
        if save_chain:
            self.decision_chain = np.zeros((self.Niter, 1), dtype=np.int)
            self.q_proposed_chain = np.zeros((self.Niter, self.D), dtype=np.float)
            self.q_initial_chain = np.zeros((self.Niter, self.D), dtype=np.float)
            self.q_initial_chain[0, :] = q_start[0]# Initializing the chain
            
        # Variables for computing acceptance rate
        accept_counter_warm_up = 0
        accept_counter = 0
        
        # Executing MHMC            
        for m in xrange(self.Nchain): # For each chain
            start = time.time()   
            
            # Initializing the chain
            self.q_chain[m, 0, :] = q_start[m]
            if save_chain and (m==0):
                self.q_initial_chain[0, :] = self.q_chain[m, 0, :]
                self.q_proposed_chain[0, :] = self.q_chain[m, 0, :]
            
            if verbose:
                print "Running chain %d" % m                
            
            # Take the initial value
            q_initial = self.q_chain[m, 0, :]
            lnL_initial = self.target_lnL(q_initial) 

            for i in xrange(1, self.Niter): # For each step
                # Get the proposal
                q_proposed = q_initial + self.proposal()

                # Compute log-likelihood differece
                lnL_proposed = self.target_lnL(q_proposed)
                dlnL = lnL_proposed - lnL_initial

                if save_chain and (m==0):
                    self.q_initial_chain[i, :] = q_initial
                    self.q_proposed_chain[i, :] = q_proposed
                

                # Take the MHMC step according to dlnL
                lnu = np.log(np.random.random(1))        
                if (dlnL > 0) or (lnu < dlnL): # If the loglikelihood improves, or the dice roll works out
                    q_initial = q_proposed            
                    lnL_initial = lnL_proposed
                    if save_chain and (m==0):
                        self.decision_chain[i, 0] = 1
                        
                    if i >= self.warm_up_num: # Save the right cadence of samples.
                        self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_proposed                        
                        self.lnL_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = lnL_proposed
                        accept_counter +=1                            
                    else:
                        accept_counter_warm_up += 1
                else: # Otherwise proposal rejected.
                    if i >= self.warm_up_num: # Save the right cadence of samples.                        
                        self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_initial      
                        self.lnL_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = lnL_initial
            dt = time.time() - start
            print "Time taken: %.2f\n" % dt 
            
        print "Compute acceptance rate"
        self.accept_R_warm_up = accept_counter_warm_up / float(self.Nchain * self.warm_up_num)
        self.accept_R = accept_counter / float(self.Nchain * (self.Niter - self.warm_up_num))
        print "During warm up: %.3f" % self.accept_R_warm_up
        print "After warm up: %.3f" % self.accept_R

        return 

        
    
    def proposal(self):
        """
        Additive proposal. Add returned value to the current point to get
        the proposal.
        """
        
        return np.random.multivariate_normal(np.zeros(self.cov_jump.shape[0]), self.cov_jump, size=1)

    
    def make_movie(self, T_last=-1, warm_up=False, fname_prefix="test", dx =0.1, xmax=4, plot_normal=True, q0=None, cov0=False, plot_cov=True):
        """
        Creates a deck of png files that can be turned into a movie.
        
        Args:
        - T_last: The duration of the movie in units of MH step. 
        If T_last exceeds the total number of steps, then maximum is used.
        - warm_up: If True, start making the movie after the warm-up fraction.
        - See others
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
            if (counter % 10) == 0:
                print "%d: Making slide %d" % (counter, i)
            self.make_slide(fname_prefix, idx_start, i, i - idx_start, plot_normal=plot_normal, q0=q0, cov0=cov0, plot_cov=plot_cov, Ntotal=(idx_last-idx_start+1), xmax=xmax, dx=dx)
            counter +=1
        
        # Command for creating and movies and deleting slides.
        print "Use the following command to make a movie:\nffmpeg -r 1 -start_number 0 -i %s-%%d.png -vcodec mpeg4 -y output.mp4" % fname_prefix
        
        return 
    
    def make_slide(self, fname_prefix, idx_chain_start, idx_chain, idx_slide, dx =0.1, xmax=4, plot_normal=True, q0=None, cov0=False, plot_cov=True, Ntotal=100):
        """
        Make a slide for the movie.
        Args:
        - fname_prefix: 
        - idx_chain: Slide index.
        - idx: Index to be appended to the slide.
        - Ntotal: Total number of samples to be plotted
        """
        pt_size_previous = 20
        pt_size_current = pt_size_proposed = 50
        
        # Take samples of the first and second variables
        q1 = self.q_initial_chain[:, 0]
        q2 = self.q_initial_chain[:, 1]
        q1_proposed = self.q_proposed_chain[:, 0]
        q2_proposed = self.q_proposed_chain[:, 1]
        
        
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
        ax_list[0, 0].scatter(q1[idx_chain_start:idx_chain], q2[idx_chain_start:idx_chain], s=pt_size_previous, c="black", edgecolors="none")
        
        # Current proposal
        if self.decision_chain[idx_chain, 0] == 1: # If the proposal is accepted
            # Big black dot
            ax_list[0, 0].scatter([q1_proposed[idx_chain]], [q2_proposed[idx_chain]], c="black", edgecolors="none", s=pt_size_proposed)
        else:
            # Empty circle
            ax_list[0, 0].scatter([q1_proposed[idx_chain]], [q2_proposed[idx_chain]],  c="none", edgecolors="black", s=pt_size_proposed)
        
        # Initial location
        ax_list[0, 0].scatter([q1[idx_chain]], [q2[idx_chain]], c="red", s=pt_size_current, edgecolors="none")
        
        # Line between current proposal and location
        ax_list[0, 0].plot([q1[idx_chain], q1_proposed[idx_chain]], [q2[idx_chain], q2_proposed[idx_chain]], c="black", ls="--", lw = 1)
        
        
        if plot_cov:
            plot_cov_ellipse(ax_list[0, 0], [q0], [cov0], 0, 1, MoG_color="Blue")
        ax_list[0, 0].set_xlabel("q1", fontsize=ft_size)
        ax_list[0, 0].set_ylabel("q2", fontsize=ft_size)
        ax_list[0, 0].axis("equal")
        ax_list[0, 0].set_xlim([-xmax, xmax])
        ax_list[0, 0].set_ylim([-xmax, xmax])

        # q2 histogram
#         if idx_chain_start < idx_chain:
        ax_list[0, 1].hist(q2[idx_chain_start:idx_chain], bins=np.arange(-xmax, xmax, dx), histtype="step", color="black", orientation="horizontal")
        if plot_normal:
            assert (q0 is not None) and (cov0 is not None)
            ax_list[0, 1].plot(q2_marginal, xgrid, c="green", lw=2)
        ax_list[0, 1].set_ylim([-xmax, xmax])
        ax_list[0, 1].set_xlim([0, q1_marginal.max() * 1.5])        
        ax_list[0, 1].set_ylabel("q2", fontsize=ft_size)
        # q1 histogram
#         if idx_chain_start < idx_chain:        
        ax_list[1, 0].hist(q1[idx_chain_start:idx_chain], bins=np.arange(-xmax, xmax, dx), histtype="step", color="black")
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
        plt.suptitle("T_MH = %d" % idx_chain, fontsize=25)
        plt.savefig(fname, dpi=100, bbox_inches = "tight")
        plt.close()        



class HMC_sampler(sampler):
    """
    HMC sampler for a general distribution.
    
    The main user functions are the constructor and the gen_sample.
    
    Note: The energy diagonostic implementation is likely to be wrong.
    """
    
    def __init__(self, D, V, dVdq, Nchain=2, Niter=1000, thin_rate=1, warm_up_frac=0.1, warm_up_num=None, \
                 cov_p=None, sampler_type="Fixed", L=None, dt=None):
        """
        Args: See Sampler class constructor for other variables.
        - V: The potential function which is the negative lnL.
        - dVdq: The gradient of the potential function to be supplied by the user.
        - cov_p: Covariance matrix of the momentum distribution assuming Gaussian momentum distribution.
        - L, dt: Number of steps and time step to be taken for each sample if sampler type is "Fixed".
        - sampler_type = "Fixed", "Random", or "NUTS"
        """
        # parent constructor
        sampler.__init__(self, D=D, target_lnL=None, Nchain=Nchain, Niter=Niter, thin_rate=thin_rate, warm_up_frac=warm_up_frac, warm_up_num=warm_up_num)
        
        # Save the potential and its gradient functions
        self.V = V
        self.dVdq = dVdq
        
        # Which sampler to use?
        assert (sampler_type=="Fixed") or (sampler_type=="Random") or (sampler_type=="NUTS")
        self.sampler_type = sampler_type
        if self.sampler_type == "Fixed":
            assert (L is not None) and (dt is not None)
            self.L = L
            self.dt = dt
        
        # Momentum covariance matrix and its inverse
        if cov_p is None: 
            self.cov_p = np.diag(np.ones(self.D))
        else:
            self.cov_p = cov_p                    
        self.inv_cov_p = np.linalg.inv(self.cov_p)             
        
        # Save kinetic energy of proposal and total energy
        self.Eq_chain = np.zeros((self.Nchain, self.L_chain, 1), dtype=np.float)
        self.E_chain = np.zeros((self.Nchain, self.L_chain, 1), dtype=np.float)
        
        
    def gen_sample(self, q_start, save_chain=False, verbose=True):
        """
        Save Nchain of (Niter-warm_up_num)//thin_rate samples.
        Also, record acceptance/rejection rate before and after warm-up.
        
        Appropriate samplers "Fixed", "Random", or "NUTS" is called.
        
        Args:
        - q_start: The function takes in the starting point as an input.
        This requirement gives the user a fine degree of choice on how to
        choose the starting point. Dimensions (self.Nchain, D)
        - save_chain: If True, then save the entire *first* chain. Specifically
        what is saved depends on the sampler.
        - verbose: If true, then print how long each chain takes.
        """
        
        if self.sampler_type == "Fixed":
            self.gen_sample_fixed(q_start, save_chain, verbose)
        # Other samplers come here
            
        return
    
    
    def gen_sample_fixed(self, q_start, save_chain, verbose):
        """
        - save_chain: If True, in addition to saving all warmed-up and thinned samples,
        save in an array the following (only the first chain):
        1) phi_q (Niter, L, D): The full trajectory starting with the initial.
        2) decision_chain (Niter, 1): Whether the proposal was accepted or not.
        """
    
        # Check if the correct number of starting points have been        
        assert q_start.shape[0] == self.Nchain
    
        if (save_chain):
            self.decision_chain = np.zeros((self.Niter, 1), dtype=np.int)
            self.phi_q = np.zeros((self.Niter, self.L+1, self.D), dtype=np.float)
            # +1 because we want to count the initial AND final.
            
        # Variables for computing acceptance rate
        accept_counter_warm_up = 0
        accept_counter = 0
        
        # Executing HMC
        for m in xrange(self.Nchain): # For each chain
            start = time.time()   
            
            # Initializing the chain
            if verbose:
                print "Running chain %d" % m                
            
            # Take the initial value: We treat the first point to be accepted without movement.
            self.q_chain[m, 0, :] = q_start[m]
            q_initial = q_start[m]
            p_tmp = self.p_sample()[0] # Sample momentun
            self.E_chain[m, 0, 0] = self.E(q_initial, p_tmp)
            self.Eq_chain[m, 0, 0] = self.K(p_tmp)
            # Save the initial point of the trajectory if asked for.
            if save_chain and (m==0):
                self.phi_q[0, 0, :] = q_initial            

            for i in xrange(self.Niter): # For each step
                # Initial position/momenutm
                q_tmp = q_initial
                p_tmp = self.p_sample()[0] # Sample momentun

                # Compute kinetic initial energy and save
                K_initial = self.K(p_tmp)
                
                # Compute the initial Energy
                E_initial = self.E(q_tmp, p_tmp)
                
                # Save the initial point of the trajectory if asked for.
                if save_chain and (m==0):
                    self.phi_q[i, 0, :] = q_initial
                    
                # Perform HMC integration and save the trajectory if requested.
                for l in xrange(1, self.L+1):
                    p_tmp, q_tmp = self.leap_frog(p_tmp, q_tmp)
                    if save_chain and (m == 0):
                        self.phi_q[i, l, :] = q_tmp

                # Compute final energy and save.
                E_final = self.E(q_tmp, p_tmp)
                
                # Save Kinetic energy
                if i >= self.warm_up_num: # Save the right cadence of samples.
                    self.Eq_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = K_initial
                    self.E_chain[m, (i-self.warm_up_num)//self.thin_rate, 0] = E_final
                    
                # With correct probability, accept or reject the last proposal.
                dE = E_final - E_initial
                lnu = np.log(np.random.random(1))        
                if (dE < 0) or (lnu < -dE): # If accepted.
                    q_initial = q_tmp
                    if (save_chain) and (m==0):
                        self.decision_chain[i, 0] = 1
                    if i >= self.warm_up_num: # Save the right cadence of samples.
                        self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_tmp
                        accept_counter +=1                            
                    else:
                        accept_counter_warm_up += 1                        
                else: # Otherwise proposal rejected.
                    self.q_chain[m, (i-self.warm_up_num)//self.thin_rate, :] = q_initial
            
            dt = time.time() - start
            print "Time taken: %.2f\n" % dt 

        print "Compute acceptance rate"
        self.accept_R_warm_up = accept_counter_warm_up / float(self.Nchain * self.warm_up_num)
        self.accept_R = accept_counter / float(self.Nchain * (self.Niter - self.warm_up_num))
        print "During warm up: %.3f" % self.accept_R_warm_up
        print "After warm up: %.3f" % self.accept_R
        print "Complete"            

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
        print "Use the following command to make a movie:\nffmpeg -r 1 -start_number 0 -i %s-%%d.png -vcodec mpeg4 -y output.mp4" % fname_prefix
        
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