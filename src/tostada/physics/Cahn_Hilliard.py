import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
import sys
from PIL import Image
import glob
import shutil
from tqdm import tqdm  # For progress bar
from time import time
#from scipy.ndimage import laplace
#from scipy.fft import fftn, ifftn, fftshift
from scipy.stats import norm
from tostada.PhaseDistribution import PhaseDistribution

try:
    import cupy as cp
    from cupyx.scipy.ndimage import laplace
    cp.asarray([0])
    print ('GPU detected. Using CUDA.')
    def asnumpy(x):
        return cp.asnumpy(x)
    
except Exception as e:
    print (f"GPU not available : {e}")
    import numpy as cp
    from scipy.ndimage import laplace
    def asnumpy(x):
        return x


class CahnHilliard:
    """
    Solve the Cahn-Hilliard (CH) equation for a box in 2D/3D with GPU acceleration (CuPy). With an unstable random distribution of concentration field c(r) for a particular
    volume fraction as an initial guess, the system evolves in a manner that minimizes the surface energy. 
    Depending on the initial conditions and the volume fraction, different two-phase distributions like nucleation and spinodal decomposition can be obtained.
    Final result is a phase distribution (tostada.PhaseDistribution object).
    
    Parameters
    ----------
    N : int, optional
        Number of pixels in each dimension. Default: 64.
    p : float, optional
        Volume-fraction control, where solid fraction = 0.5 - p/2. For p = 0.14, solid fraction = 0.43. Default: 0.14
    D : float, optional
        Mobility or diffusion coefficient. Default: 1.0. A larger value implies faster simulation. For convergence, reduce dt appropriately.
    gamma : float, optional
        Controls the spatial width of the transition regions between different phases. Default: 0.5.
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
        N: int = 64,
        p: float = 0.14,  # p controls volume fraction: 0.5 + p/2 = solid fraction
        D: float = 1,
        gamma: float = 0.5,
        num_iter: int = 5000,
        dt: float = 0.001,
        frame_iter: int = 50,
        is_3D = False,
        resolution = 0.01,
        seed = None,#12,
        initial_state=None
    ) -> None:
        self.N = N
        self.p = p  # p = 0.14 gives 43% solid (0.5 - 0.14/2 = 0.43)
        self.D = D
        self.gamma = gamma
        self.num_iter = num_iter
        self.dt = dt
        self.frame_iter = frame_iter
        self.is_3D = is_3D
        self.ndim = int(3*self.is_3D + 2*np.logical_not(self.is_3D)) #dimensionality (2D or 3D)
        self.resolution = resolution
        self.seed = seed if seed is not None else 12
        self.initial_state = initial_state

    # Optimized Cahn-Hilliard equation using custom Laplacian
    def CH_eqn(
        self, c: npt.NDArray[np.float64], D: float, gamma: float
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the concentration field at each time step. Expensive computation in 3D. 

        Parameters
        ----------
        c : array_like
            concentration field c(r) in the range -1 < c(r) < +1
        D : float, optional
            Mobility or diffusion coefficient. Default: 1.0.
        gamma : float, optional
            Interfacial energy parameter. Default: 0.5.

        """
        c = cp.asarray(c,dtype=cp.float64)
        lap_c = laplace(input=c, mode="wrap")
        # Calculate the chemical potential: f'(c) - gamma * lap_c
        # f'(c) = c^3 - c for the double-well potential
        mu = (c**3) - c - (1/4)-(gamma * lap_c) #chemical potential
        #mu = c**7 - 3*c**3 - 3*c**5 + c - 1/8 - (gamma*lap_c)
        # Return the Laplacian of the chemical potential
        return asnumpy(D * laplace(input=mu, mode="wrap"))

    # Initialize random data on a 3D lattice with specified volume fraction
    def initialize_lattice(self) -> npt.NDArray[np.float64]:
        """
        Initialization of the field.
        """
        rng = np.random.default_rng(seed=self.seed)
        c = rng.choice([-1, 1], np.ones([self.N]*self.ndim).shape, p=[0.5 + (self.p / 2), 0.5 - (self.p / 2)])
        c_ = (c+1)/2
        print(f"Initial mean value: {np.mean(c_):.4f}")
        
        return c

    # forward Euler method with stability check
    def forward_euler(self, x, func, dt, *fargs) -> npt.NDArray[np.float64]:
        """
        Forward time updates for the concentration field.
        """
        increment = func(*fargs) * dt
        # Check for numerical stability
        #if np.max(np.abs(increment)) > 0.1:
        #    print(f"Warning: Large update step detected: {np.max(np.abs(increment))}")
        #    # Cap the maximum change to prevent instability
        return x + increment

    # solver function with performance tracking
    def solve(self) -> npt.NDArray[np.float64]:
        """
        Time-evolution for the Cahn-Hilliard equation. Returns a PhaseDistribution object.
        """
        if (self.initial_state is None):
            c: npt.NDArray[np.float64] = self.initialize_lattice()
        else:
            c = self.initial_state
        t = 0
        
        # Ensure directories exist
        os.makedirs("CH_gifs/tmp", exist_ok=True)
        os.makedirs("CH_data", exist_ok=True)

        # Set up the plotting environment
        _, ax = plt.subplots()
        self.plot_lattice(c, ax, t, i=0)
        
        # Track performance
        start_time = time()
        last_time = start_time
        
        # For tracking phase evolution
        times = []
        solid_fractions = []
        structure_sizes = []
        fwhms=[]
        Hyp = []
        # Use tqdm for progress tracking
        for i in tqdm(range(1, self.num_iter + 1), desc="Solving Cahn-Hilliard"):
            fargs: tuple[npt.NDArray[np.float64], float, float] = (
                c,
                self.D,
                self.gamma,
            )
            
            # Time integration step
            c_new = self.forward_euler(c, self.CH_eqn, self.dt, *fargs)
            
            # Check for conservation of total concentration (should be conserved)
            if i % 100 == 0:
                total_before = np.mean((c+1)/2)
                total_after = np.mean((c_new+1)/2)
                if abs(total_before - total_after) > 1e-10:
                    print(f"Warning: Conservation violated! Before: {total_before}, After: {total_after}")
            
            c = c_new
            t = np.round(t + self.dt, 6)
            
            # Performance tracking
            if i % 500 == 0:
                current_time = time()
                elapsed = current_time - last_time
                last_time = current_time
                print(f"Time for 500 iterations: {elapsed:.2f}s ({500/elapsed:.2f} it/s)")
            
            # Only plot and save on specified intervals
            if i % self.frame_iter == 0:
                ax.clear()
                self.plot_lattice(c, ax, t, i)
                c_ = (c+1)/2
                solid_fraction = np.mean(c_)
                print(f"t={t:.4f}: Solid: {solid_fraction:.3%}")
                X = PhaseDistribution(c_,resolution=self.resolution)
                xq = X.ReciprocalSpace()[1]
                if (i > 100):
                    stats = X.Hyperuniformity_data(fwhm=False)
                    Dmean = X.Dmean_from_q()
                    #print ('FWHM={f}, Hyperuniformity={h}, Dmean={dm}'.format(f=stats[0],h=stats[1],dm=Dmean))
                    print ('Hyperuniformity={h}, Dmean={dm}'.format(h=stats,dm=Dmean))
                    # Calculate and print structure size metric
                    structure_metric = self.calculate_structure_metric(c)
                    print(f"Structure size metric: {structure_metric:.4f}")
                else:
                    stats=10
                    structure_metric = 10
                # Track evolution
                times.append(t)
                #fwhms.append(stats[0])
                Hyp.append(stats)#[1])
                solid_fractions.append(solid_fraction)
                structure_sizes.append(structure_metric)
                
            # Save intermediate results less frequently
            if i % (10 * self.frame_iter) == 0:
                np.save(f'CH_data/c_{self.ndim}D_t={i}.npy', c)
                
                # Plot evolution so far
                self.plot_evolution(times, solid_fractions, structure_sizes)
                
        total_time = time() - start_time
        print(f"Total computation time: {total_time:.2f}s ({self.num_iter/total_time:.2f} it/s)")
        
        # Final evolution plot
        self.plot_evolution(times, solid_fractions, structure_sizes)
        #np.savez('CH_data/Hyperuniformity_evolution.npz',fwhm=np.asarray(fwhms),hyp=np.asarray(Hyp))
        #np.savez('CH_data/Hyperuniformity_evolution.npz',hyp=np.asarray(Hyp))
        X_final = PhaseDistribution((c+1)/2,resolution=self.resolution)
        self.c_final = (c+1)/2
        return X_final
    
    # Plot the evolution of phases and structure size
    def plot_evolution(self, times, solid_fractions, structure_sizes):
        """Plot the evolution of phase fractions and structure size"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot solid fraction
        ax1.plot(times, solid_fractions, 'b-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Solid Fraction')
        ax1.set_title('Evolution of Solid Fraction')
        ax1.grid(True)
        
        # Plot structure size metric
        ax2.plot(times, structure_sizes, 'r-')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Structure Size Metric')
        ax2.set_title('Evolution of Structure Size')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('CH_data/evolution_plot.png')
        plt.close(fig)
    
    # Calculate a metric for structure size 
    def calculate_structure_metric(self, c):
        """Calculate a metric for structure size using spatial autocorrelation"""
        # Calculate the gradient magnitude
        grad_x = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
        grad_y = np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)
        grad_z = 0#np.roll(c, -1, axis=2) - np.roll(c, 1, axis=2)
        
        # Interface density is related to the gradient magnitude
        interface_density = np.mean(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Inverse of interface density gives a measure of structure size
        # Higher values indicate larger structures
        return 1.0 / (interface_density + 1e-10)

    # Plot multiple views of the 3D data
    def plot_lattice(self, c, ax, t, i) -> None:
        # Plot middle slice in XY plane
        #mid_z = int(c.shape[1]/2)
        if (self.is_3D):
            im = ax.imshow(c[:,c.shape[1]//2,:], cmap='RdBu_r', vmin=-1, vmax=1)
        else:
            im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
        
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
        c_ = (c+1)/2
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
        plt.savefig(f"CH_gifs/tmp/{i:06d}.png")

    def animate(self) -> None:
        """
        Create animation from the images produced from self.solve().
        """
        path_in: str = "CH_gifs/tmp/*.png"
        path_out: str = f"CH_gifs/spin-decomp-{self.ndim}d-d_{self.D}-gamma_{self.gamma}-p_{self.p}.gif"
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
        if os.path.exists("CH_gifs/tmp"):
            shutil.rmtree("CH_gifs/tmp/")





