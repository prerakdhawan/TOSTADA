import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
import sys
from PIL import Image
import glob
import shutil
from time import time
#from scipy.ndimage import laplace
#from scipy.fft import fftn, ifftn, fftshift
from scipy.stats import norm
from tostada.PhaseDistribution import PhaseDistribution
try:
    from tqdm.notebook import tqdm # for progress bar
except ImportError:
    from tqdm import tqdm

try:
    import cupy as cp
    from cupyx.scipy.ndimage import laplace
    cp.asarray([0])
    #print ('GPU detected. Using CUDA.')
    def asnumpy(x):
        return cp.asnumpy(x)
    
except Exception as e:
    #print (f"GPU not available : {e}")
    import numpy as cp
    from scipy.ndimage import laplace
    def asnumpy(x):
        return x

class PhaseField:
    """
    Time evolve a `tostada.PhaseDistribution` in 2D/3D with GPU acceleration (CuPy) using phase field theory. 
    The phase evolution occurs with thermodynamic, symmetry and other considerations imposed via an appropriate free-energy for a given morphology.
    The `PhaseDistribution` modelled here can be interpretted as a spatially inhomogeneous field that, when optimized via a free-energy minimization protocol, 
    exhibits several spatial localized maximas, each of which can be identified as the average location of atoms/particles. 
    Such a phase evolution has been widely used to study grain boundary dynamics, crystal nucleation, crystal growth, glass formation, crack propagation and etc.      
    
    Parameters
    ----------

    model : str
        The basic free energy term that needs to be minimized. Since the phase evolution hinges on free-energy minimization, perturbative terms 
        can be added so long as the gradients w.r.t phase distribution (or the Chemical Potential) are analytic.
        Current available options are `cahn_hilliard`,`swift_hohenberg`,`ks_damped`,`pfc_hex`,`pfc_bcc`,`pfc_fcc`,`pfc_square`,`pfc_cubic`
    N : int, optional
        Number of pixels in each dimension. Default: 64.
    p : float, optional
        Volume-fraction control, where solid fraction = 0.5 - p/2. For p = 0.14, solid fraction = 0.43. Default: 0.14
    D : float, optional
        Mobility or diffusion coefficient. Default: 1.0. A larger value implies faster simulation. For convergence, reduce dt appropriately.
    num_iter : int, optional 
        Number of time-stepping iterations. Default: 5000.
    dt : float, optional 
        Time step size. Default: 0.001.
    frame_iter : int, optional 
        Interval for saving, evaluating spectral density and plotting frames Default: 50. 
        For 3D, keep this large as these are expensive evaluations and are only needed in few time steps.
    is_3D : bool, optional
        Is the phase distribution in 3D or 2D. Default: False (ndim=2D)
    resolution : float, optional
        Resolution of each pixel in microns. Default: 0.01 microns
    seed : int, optional, optional
        Seed for the random initialization. Change for different outcomes.
    initial_state : ndarray, optional
        Initialized state. If None, creates a random one with self.seed
    """
    def __init__(
        self,
        model=['cahn_hilliard','swift_hohenberg','ks_damped','pfc_hex','pfc_bcc','pfc_fcc','pfc_square','pfc_cubic'],
        N: int = 64,
        p: float = 0.14,  # p controls volume fraction: 0.5 + p/2 = solid fraction
        D: float = 1,
        num_iter: int = 5000,
        dt: float = 0.001,
        frame_iter: int = 50,
        is_3D = False,
        resolution = 0.01,
        seed = None,#12,
        initial_state=None
    ) -> None:
        self.model=model,
        self.N = N
        self.p = p  # p = 0.14 gives 43% solid (0.5 - 0.14/2 = 0.43)
        self.D = D
        self.num_iter = num_iter
        self.dt = dt
        self.frame_iter = frame_iter
        self.is_3D = is_3D
        self.ndim = int(3*self.is_3D + 2*np.logical_not(self.is_3D)) #dimensionality (2D or 3D)
        self.resolution = resolution
        self.seed = seed if seed is not None else 12
        self.initial_state = initial_state

    @staticmethod
    def mu_Cahn_Hilliard(psi: cp.ndarray, gamma: float, a: float = -1.0, b: float = 1.0): 
        """
        Chemical potential for solve the Cahn-Hilliard (CH) equation. With an unstable random distribution of concentration field c(r) for a particular
        volume fraction as an initial guess, the system evolves in a manner that minimizes the surface energy. 
        Depending on the initial conditions and the volume fraction, different two-phase distributions like nucleation and spinodal decomposition can be obtained.
        Final result is a phase distribution (`tostada.PhaseDistribution` object).

        Parameters
        ----------
        psi : cupy ndarray
            Phase field as a 2D/3D cupy array
        gamma : float
            Controls the spatial width of the transition regions between different phases. 
        a : float, optional
            Parameter controlling the quadratic term in the double-well potential of the local free energy. Default : -1.0
        b : float, optional
            Parameter controlling the 4th order term in the double-well potential of the local free energy. Default : +1.0
        """
        lap_c = laplace(input=psi, mode="wrap")
        mu = b * (psi**3) + a*psi - (1/4)-(gamma * lap_c)
        return -mu
    
    @staticmethod
    def mu_Swift_Hohenberg(psi: cp.ndarray,r: float, q0 : float = 1): 
        """
        Chemical potential for solving Swift-Hohenberg equation. When free energy is chosen for Swift-Hohenberg, the field is evolved using this mu. 
        Please note : since the free-energy for Swift-Hohenberg and for a phase-field crystal (PFC) are identical, the chemical potential for both cases are the same.
        Only difference is that PFC is mass-conserving (mean value of psi remains unchanged over time) and so laplacian of mu is time evolved.

        Parameters
        ----------
        psi : cupy ndarray
            Phase field as a 2D/3D cupy array
        r : float
            Effective Temperature control parameter. Acts similar to E/kT term. Controls order vs disorder.
        q0 : float, optional
            Charateristic spatial frequency. Controls the characteristic length scale of the domain (q0 = 2pi/Dmean).
        """
        lap_c = laplace(psi, mode="wrap")
        lap2_c = laplace(lap_c, mode="wrap")
        mu = psi * (q0**4+r) + 2*q0**2*lap_c + lap2_c + psi**3
        return -mu
    
    @staticmethod
    def mu_PFC_square(psi: cp.ndarray,r: float): 
        """
        Todo : Failed in testing. Recheck the expressions.
        Current expressions only evaluated for q0=1
        """
        lap_c = laplace(psi, mode="wrap")
        lap2_c = laplace(lap_c, mode="wrap") #k4 term
        lap4_c = laplace(lap2_c, mode="wrap")# k8 term
        mu = psi *(4+r) + 13*lap2_c + 7*lap4_c + 12*lap_c + psi**3
        return -mu
    
    @staticmethod
    def mu_PFC_square_spectral(psi, kx, ky, eps, r, q1=1.0, q2=cp.sqrt(2) ):
        """
        Semi-spectral chemical potential for square lattice.
        """
        psi_hat = cp.fft.fftn(psi)
        k2 = kx**2 + ky**2
        L1 = (q1**2 - k2)**2
        L2 = (q2**2 - k2)**2 + r
        # spectral operator multiplies these bands
        linear_hat = -((L1 * L2) + eps) * psi_hat
        mu = cp.fft.ifftn(linear_hat).real + psi**3
        return -mu
    
    @staticmethod
    def mu_PFC_fcc(psi: cp.ndarray,r: float,q0: float = 1,*args): 
        """
        Todo : NOT finished. Implement the correct expression. 
        """
        lap_c = laplace(psi, mode="wrap")
        lap2_c = laplace(lap_c, mode="wrap")
        mu = psi * (q0**4+r) + 2*q0**2*lap_c + lap2_c + psi**3
        return mu
    
    @staticmethod
    def KS_damped_spectral(psi, kx, ky, nu, kappa, lam, alpha=0.0):
        """
        Computes the RHS of Damped/Undamped Kuramoto-Sivashinsky equation. NOTE : this is not a chemical potential model.
        """
        psi_hat = cp.fft.fftn(psi)

        k2 = kx**2 + ky**2

        # Linear operator
        L_hat = (-alpha + nu * k2 - kappa * k2**2) * psi_hat

        # Gradients (spectral)
        ux = cp.fft.ifftn(1j * kx * psi_hat).real
        uy = cp.fft.ifftn(1j * ky * psi_hat).real

        grad_sq = ux**2 + uy**2
        N_hat = cp.fft.fftn(0.5 * lam * grad_sq)

        rhs_hat = L_hat + N_hat
        return cp.fft.ifftn(rhs_hat).real

    def chemical_potential(self,psi: cp.ndarray,additional_mufunc,*args,**kwargs):
        """
        EFFECTIVE Chemical potential for the system. The term "effective" because not all models have a well-defined free energy functional. 
        Since the time-evolution occurs through free-energy minimization, the effective chemical potential acts as gradients for the optimization (\\mu = dF/d\\psi). 
        Currently, the available models are `cahn_hilliard`,`swift_hohenberg`,`KS_damped`,`pfc_hex`,`pfc_square` and `pfc_bcc`. 
        Perturbations to these models can be passed through custom `additional_mufunc` with its respective keyword arguments.

        Parameters
        ----------
        psi : cupy ndarray
            Phase field as a 2D/3D cupy array

        additional_mufunc : additional function, optional
            Additional perturbative python function to base chemical potential. Arguments and keyword arguments can be passed through args and kwargs. Default : None 

        """
        mu_functions = {
            'cahn_hilliard': self.mu_Cahn_Hilliard,
            'swift_hohenberg': self.mu_Swift_Hohenberg,
            'ks_damped': self.KS_damped_spectral,
            'pfc_hex': self.mu_Swift_Hohenberg,
            'pfc_bcc': self.mu_Swift_Hohenberg,
            'pfc_square': self.mu_PFC_square_spectral,
        }
        if self.model[0] not in mu_functions:
            raise ValueError(f"Unknown option '{self.model[0]}'. Valid options are: {list(mu_functions.keys())}")
        mu = mu_functions[self.model[0]](psi,*args,**kwargs)
        if additional_mufunc is not None:
            mu += additional_mufunc(psi, **kwargs)
        return mu

    def PFC_eqn(self,psi: cp.ndarray, noise_amp : float = 0, 
                noise_std : float = 1.0, additional_func=None, 
                additional_mufunc=None, *args, **kwargs):
        """
        Phase field term (RHS) that gets time evolved. Different chemical potentials yield different morphologies in time. 
        The generalized expression is given as d\\phi/dt = D * laplacian(\\mu) + laplacian(eta) where eta(r) is the Gaussian white noise field with zero mean.
        Perturbations to the chemical potential can be passed using `additional_mufunc` while perturbations to the general expression can be passed through `additional_func`.

        Parameters
        ----------

        psi : cupy ndarray
            Phase field as a 2D/3D cupy array
        
        noise_amp : float
            Amplitude of the noise term
        
        noise_std : float
            Standard deviation of the noise term

        additional_func : list
            List of additional perturbative function and its arguments. The list must be arranged as [function_name, argument1, argument2, ...]. Default : None

        additional_mufunc : python function
            Perturbative python function for the chemical potential. Arguments and keyword arguments can be passed through args and kwargs. Default : None 

        """
        mu = self.chemical_potential(psi=psi,additional_mufunc=additional_mufunc, *args,**kwargs)

        if (np.logical_or(self.model[0]=='swift_hohenberg', self.model[0]=='ks_damped')):
            rhs = self.D * mu
        else:
            rhs = -self.D * laplace(mu, mode="wrap")

        # substrate noise (divergence of random flux)
        eta = noise_amp * cp.random.normal(0,2*noise_std,psi.shape)
        rhs += laplace(eta, mode="wrap")
        if additional_func is not None:
            rhs += additional_func[0](psi,*additional_func[1:])
        return rhs

    # Initialize random data on a 3D lattice with specified volume fraction
    def initialize_lattice(self):
        """
        Initialization of the field.
        """
        rng = np.random.default_rng(seed=self.seed)
        c = rng.choice([-1, 1], np.ones([self.N]*self.ndim).shape, p=[0.5 + (self.p / 2), 0.5 - (self.p / 2)])
        c_ = (c+1)/2
        print(f"Initial mean value: {cp.mean(c_):.4f}")
        return c

    def initialize_lattice_2(self, noise : float = 0.01):
        """
        Initialization of the field.
        """
        psi_i = cp.asarray(self.p+np.random.uniform(-noise,noise,[self.N]*self.ndim) )
        #c = rng.choice([-1, 1], np.ones([self.N]*self.ndim).shape, p=[0.5 + (self.p / 2), 0.5 - (self.p / 2)])
        psi_ = (psi_i - np.min(psi_i))/(np.max(psi_i)-np.min(psi_i))
        print(f"Initial mean value: {cp.mean(psi_):.4f}")
        return psi_i

    def forward_euler(self, x, func, dt, additional_mufunc=None, additional_func=None,*args,**kwargs):
        """
        Forward time updates for the phase field.

        Parameters
        ----------

        x : cupy array

        func : python function
            Function whose output needs to evolve in time. Example: `self.PFC_eqn()`
        
        dt : float
            Time step for each time iteration. Keep this small for stability. 
            NOTE: Different chemical potentials require different dt as the stiffness of the PDE is different for each case.
        
        additional_func : list
            List of additional perturbative function to the `func` and its arguments. The list must be arranged as [function_name, argument1, argument2, ...]. Default : None

        additional_mufunc : python function
            Perturbative python function for the chemical potential. Arguments and keyword arguments can be passed through args and kwargs. Default : None 
        """
        increment = func(x,additional_func=additional_func,additional_mufunc=additional_mufunc,*args,**kwargs) * dt
        return x + increment
    

    # solver function with performance tracking
    def solve(self, additional_mufunc=None, additional_func=None,compute_hyperuniformity=True,*args,**kwargs): 
        """
        Time-evolution for the Phase field. Returns a `tostada.PhaseDistribution` object.
        """
        if (self.initial_state is None):
            psi_ = cp.asarray(self.initialize_lattice_2(noise=0.01))
        else:
            psi_ = cp.asarray(self.initial_state)
        t = 0
        
        # Ensure directories exist
        os.makedirs("Phasefield_gifs/tmp", exist_ok=True)
        os.makedirs("Phasefield_data", exist_ok=True)

        # Set up the plotting environment
        _, ax = plt.subplots()
        self.plot_lattice(asnumpy(psi_), ax, t, i=0)
        
        # Track performance
        start_time = time()
        last_time = start_time
        
        # For tracking phase evolution
        times = []
        mean_psi_hist = []
        Feature_size=[]
        Hyp = []
        # Use tqdm for progress tracking
        for i in tqdm(range(1, self.num_iter + 1), desc="Solving Phase field equation"):
            # Time integration step
            #psi_new = self.forward_euler(c, self.CH_eqn, self.dt, *fargs)
            psi_new = self.forward_euler(x=psi_,func=self.PFC_eqn,dt=self.dt,additional_func=additional_func,additional_mufunc=additional_mufunc,*args,**kwargs) 
            psi_ = psi_new
            t = np.round(t + self.dt, 6)
            
            # Performance tracking
            #if i % 500 == 0:
            #    current_time = time()
            #    elapsed = current_time - last_time
            #    last_time = current_time
            #    tqdm.write(f"Time for 500 iterations: {elapsed:.2f}s ({500/elapsed:.2f} it/s)")
            
            # Only plot and save on specified intervals
            if i % self.frame_iter == 0:
                current_time = time()
                elapsed = current_time - last_time
                last_time = current_time
                tqdm.write(f"Time for {self.frame_iter} iterations: {elapsed:.2f}s ({self.frame_iter/elapsed:.2f} it/s)")

                ax.clear()
                self.plot_lattice(asnumpy(psi_), ax, t, i)
                psi__ = (psi_ - cp.min(psi_))/(cp.max(psi_)-cp.min(psi_))
                solid_fraction = cp.mean(psi__)
                mean_psi_hist.append(solid_fraction)
                tqdm.write(f"t={t:.4f}: Solid: {solid_fraction:.3%}")
                X = PhaseDistribution(asnumpy(psi__),resolution=self.resolution)
                xq = X.ReciprocalSpace()[1]
                if ( np.logical_and(i > 100, compute_hyperuniformity==True)):
                    stats = X.Hyperuniformity_data(fwhm=False)
                    Dmean = X.Dmean_from_q()
                    #print ('FWHM={f}, Hyperuniformity={h}, Dmean={dm}'.format(f=stats[0],h=stats[1],dm=Dmean))
                    tqdm.write('Hyperuniformity={h}, Dmean={dm}'.format(h=stats,dm=Dmean))
                    # Calculate and print structure size metric
                # Track evolution
                times.append(t)
                #fwhms.append(stats[0])
                if (compute_hyperuniformity==True):
                    Hyp.append(stats)#[1])
                    Feature_size.append(Dmean)
            # Save intermediate results less frequently
            if i % (10 * self.frame_iter) == 0:
                #np.save(f'Phasefield_data/pf_{self.ndim}D_t={i}.npy', asnumpy(psi_))
                X.save(f"Phasefield_data/phasedistribution-{self.ndim}d-D={self.D}-p={self.p}_{self.model[0]}_t={i}")

        total_time = time() - start_time
        print(f"Total computation time: {total_time:.2f}s ({self.num_iter/total_time:.2f} it/s)")
        
        # Final evolution plot
        X_final = PhaseDistribution(asnumpy((psi_ - cp.min(psi_))/(cp.max(psi_)-cp.min(psi_))),resolution=self.resolution)
        self.final_state = X_final
        self.mean_psi_history = asnumpy(cp.asarray(mean_psi_hist))
        if (compute_hyperuniformity==True):
            self.feature_hist = np.asarray(Feature_size)
            self.hyperuniformity_hist = np.asarray(Hyp)
            self.plot_evolution(times,self.hyperuniformity_hist,self.feature_hist)
        filestr = kwargs.get('filestr', '')
        self.animate(filestr=filestr)
        
        return None
    
    # Plot the evolution of phases and structure size
    def plot_evolution(self, times, hyperuniformity_hist, structure_size_hist):
        """Plot the evolution of phase fractions and structure size"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot solid fraction
        ax1.plot(times, hyperuniformity_hist, 'b-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Hyperuniformity H')
        ax1.set_title('Hyperuniformity history')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # Plot structure size metric
        ax2.plot(times, structure_size_hist, 'r-')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Structure Size ($2\\pi/k_0$)')
        ax2.set_title('Evolution of Structure Size')
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig('Phasefield_data/evolution_plot.png')
        plt.close(fig)
    

    # Plot multiple views of the 3D data
    def plot_lattice(self, c, ax, t, i) -> None:
        # Plot middle slice in XY plane
        #mid_z = int(c.shape[1]/2)
        if (self.is_3D):
            im = ax.imshow(c[:,c.shape[1]//2,:], cmap='RdBu_r')#, vmin=-1, vmax=1)
        else:
            im = ax.imshow(c, cmap='RdBu_r')#, vmin=-1, vmax=1)
        
        # Add colorbar
        #plt.colorbar()#(im, ax=ax)
        
        #ax.set_title(f"XY Plane (z={mid_z})")
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.annotate(
            f"$t = {t:.3f}$",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox={
                "color": "white",
                "alpha": 0.8,
                "boxstyle": "Round",
                "edgecolor": None,
            },
            ha="left",
            va="top",
            fontsize=14,
        )
        c_ = (c - np.min(c))/(np.max(c)-np.min(c))
        # Add solid fraction annotation
        solid_fraction = np.mean(c_)

        ax.annotate(
            f"Solid: {solid_fraction:.1%}",
            xy=(0.05, 0.85),
            xycoords="axes fraction",
            bbox={
                "color": "white",
                "alpha": 0.8,
                "boxstyle": "Round",
                "edgecolor": None,
            },
            ha="left",
            va="top",
            fontsize=14,
        )
        
        plt.tight_layout()

        # save snapshot
        plt.savefig(f"Phasefield_gifs/tmp/{i:06d}.png")

    def animate(self,filestr: str = '') -> None:
        """
        Create animation from the images produced from self.solve().
        """
        path_in: str = "Phasefield_gifs/tmp/*.png"
        path_out: str = f"Phasefield_gifs/phase_evolution-{self.ndim}d-D={self.D}-p={self.p}_{self.model[0]}{filestr}.gif"
        imgs: list[Image.Image] = []

        # grab all snapshots
        for f in sorted(glob.glob(path_in)):
            img: Image.Image = Image.open(f)
            imgs.append(img.copy())
            img.close()

        # convert snapshots to GIF
        if imgs:
            imgs[0].save(
                fp=path_out,
                format="GIF",
                append_images=imgs[1:],
                save_all=True,
                optimize=True,
                duration=100,  # Longer duration for better viewing of complex structures
                loop=0,
            )
            print(f"Animation saved to {path_out}")

        # delete snapshots
        if os.path.exists("Phasefield_gifs/tmp"):
            shutil.rmtree("Phasefield_gifs/tmp/")





