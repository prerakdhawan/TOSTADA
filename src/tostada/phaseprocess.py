from tostada.PointDistribution import PointDistribution
from tostada.PhaseDistribution import PhaseDistribution
from tostada.physics.phase_field_dynamics import PhaseField
from tostada.pointprocess import Pointprocess
import tostada.Statistics as stats
import tostada.util.Utility
import tostada.GaussianRandomFunction as grf
from tostada.physics import CahnHilliard
import numpy as np
import scipy as sp
import os
from scipy.stats import binned_statistic,norm,cauchy,halfnorm
import scipy.spatial
from sys import exit

class Phaseprocess:
    """
    Create a phase definition for a given volume fraction, size, resolution in 2D/3D. 
    This can be used to create common phase distributions (tostada.PhaseDistribution) obtained from a point distribution (tostada.PointDistribution object)
    or one obtained from Gaussian random fields.
    
    TODO: Add microstructure obtained from Cahn-Hilliard equation 
    """
    def __init__(self, volumefraction,BoxSize,resolution,interpore_distance=None,is_3D=False):
        self.volumefraction = volumefraction
        self.interpore_distance = interpore_distance
        self.is_3D = is_3D
        self.BoxSize = BoxSize
        self.resolution = resolution
        if (np.asarray(BoxSize).shape == ()):
            print ('Taking Lx,Ly,Lz=',BoxSize)
            self.BoxSize = [BoxSize]*(2*(np.logical_not(self.is_3D))+3*(self.is_3D))

    def GRF_microstructure(self,params,options=['gaussian']):
        """
        Generate a porous media (tostada.PhaseDistribution object) using 2D/3D Excursion sets of Gaussian Random Fields (GRF). 
        Since GRFs are continuous, thresholding is done to get desired volume fraction using standard normal distribution.

        Parameters
        ----------
        params : list of floats
            List of parameters for each `options`.
        options : list of strings
            Options for covariance functions for covariances for each param. See Statistics.py for reference.

        Returns
        -------

        PhaseDistribution object
        """
        correlation_length=self.interpore_distance
        image = grf.generate_grf(self.BoxSize,self.resolution,correlation_length,params,is_3D=self.is_3D,options=options)
        image = np.where(image>-norm.ppf(self.volumefraction),1,0)
        return PhaseDistribution(image,self.resolution)

    def spinodal_decomposed_microstructure(self, D = 1, gamma=0.5, num_iter=5000, 
                                            dt = 0.001, frame_iter=10,seed=44, 
                                            animate=False, binarize = False, initial_state=None):
        """
        Generate spinodal decomposed phase distribution using Cahn-Hilliard equations available in `tostada.physics`.

        Parameters
        ----------
        
        D : float
            Diffusion coefficient for the process. Tune this for different distributions in different dimensions.
        
        gamma : float
            Controls the spatial width of the transition regions between different phases. Default: 0.5.

        num_iter : int
            Number of iterations to run the Cahn-Hilliard solver.
        
        dt : float
            Time stepping resolution for each time iteration.

        frame_iter : int
            How frequently should the quantities be evaluated and saved. Keep this large for 3D as this may excessively slow down the simulation.

        seed : int
            Seed for random numbers for initialization. 

        animate : bool
            Whether an animated gif be created for time-evolution of phases. Default : False

        binarize : bool
            Whether the final distribution be binarized strictly to 0s and 1s. Default : False
        
        initial_state : ndarray
            Initialization state. If not provided, assumes random state with given seed. Default : None

        Returns
        -------

        tostada.PhaseDistribution object

        """
        

        N = int(self.BoxSize[0]/self.resolution) # Number of pixels along each axes.
        p = 0.5 * (self.volumefraction - 0.5) # Controls volume fraction: 0.5 + p/2 = solid fraction

        ch = CahnHilliard(
                        N=N, p = p, D = D, 
                        gamma= gamma, num_iter = num_iter, 
                        dt = dt, frame_iter = frame_iter, is_3D = self.is_3D, 
                        resolution = self.resolution, seed=seed, initial_state=initial_state)
        c_final = ch.solve()
        
        if (binarize==True):
            phase = np.where(ch.c_final>0.5,1,0)
        else:
            phase = ch.c_final
        X_final = PhaseDistribution(phase,resolution=self.resolution)
        if (animate == True):
            ch.animate()
        
        return X_final


    def microstructure_from_pointdistribution(self, type, diameter, ax, ay=None, az=None,
                                              is_periodic=False, noise=0,
                                              correlation_length=0, sdev=0.06):
        """
        Generate a two-phase heterogeneous media from a pre-defined point-distribution in 2D/3D. 
        Replaces the point-distribution with identical objects or pores with well-defined statistical properties accesible via tostada.PhaseDistribution.
        
        Parameters
        ----------
        type : string
            Type of the point distribution to be used. Currently available options: 
                'perturbed-rect': Perturbed rectangular lattice
                'perturbed-hex' : Perturbed hexagonal lattice
                'rsa' : Random Sequential Adsorption process
        
        diameter : float
            Diameter of the object/pore
        
        ax : float
            Mean inter-object or inter-pore distance between objects/pores along x-direction.

        ay : float : optional
            Mean inter-object or inter-pore distance between objects/pores along y-direction. If None, ay=ax (2D)

        az : float : optional
            Mean inter-object or inter-pore distance between objects/pores along z-direction. If None, chooses az based on self.is_3D
        
        is_periodic : Bool : optional
            Boundary condition for the phase distribution. If False, avoids any pores/objects within a distance of 'ax' from the boundary. 
            If True, creates periodic-boundaries.

        noise : float : optional
            Un-correlated noise to each point in 2D/3D. Only used in lattice definitions and hence normalized to lattice constant. Default: 0
        
        correlation_length : float : optional
            Length of the correlated noise. Only used in lattice definitions and hence normalized to lattice constant. Default: 0
        
        sdev : float : optional
            Standard deviation of the histogram used in the RSA process. Default: 0.06 microns.
        """

        point_process = Pointprocess(BoxSize=self.BoxSize,diameter=diameter,ax=ax,ay=ay,az=az, is_3D=self.is_3D)
        if (type=='perturbed-rect'):
            point_dist = point_process.rect_lattice(noise=noise,correlation_length=correlation_length)
        elif (type=='perturbed-hex'):
            point_dist = point_process.hex_lattice(noise=noise)
        elif (type=='rsa'):
            point_dist = point_process.RandomSequentialAdsorption(sdev_histo=sdev)

        image = point_dist.Phaseobject(self.resolution,shapes='circle',is_periodic=is_periodic) #currently only supports circular 
        return PhaseDistribution(image,self.resolution)

    def phasefield_microstructure(self, model=['cahn_hilliard','swift_hohenberg','ks_damped','pfc_hex','pfc_bcc','pfc_fcc','pfc_square','pfc_cubic'], 
                                r = -0.5, q0=1, D = None, dt = None, num_iter = None, 
                                frame_iter = None, seed=None, initial_state = None,
                                additional_func = None, additional_mufunc = None, compute_hyperuniformity = True,
                                *args, **kwargs):
        """
        Generate a PhaseDistribution object through a phase-field method simulation in `tostada.physics.phase_field_dynamics`. 
        TODO : Not tested. Use `tostada.physics.phase_field_dynamics` if gives an error.

        Parameters
        ----------

        model : str
            The basic free energy term that needs to be minimized. Since the phase evolution hinges on free-energy minimization, perturbative terms 
            can be added so long as the gradients w.r.t phase distribution (or the Chemical Potential) are analytic.
            Current available options are `cahn_hilliard`,`swift_hohenberg`,`ks_damped`,`pfc_hex`,`pfc_bcc`,`pfc_fcc`,`pfc_square`,`pfc_cubic`

        D : float, optional
            Mobility or diffusion coefficient. Default: 1.0. A larger value implies faster simulation. For convergence, reduce dt appropriately.

        num_iter : int, optional 
            Number of time-stepping iterations. Default: 5000.
        
        dt : float, optional 
            Time step size. Default: 0.001.
        
        frame_iter : int, optional 
            Interval for saving, evaluating spectral density and plotting frames Default: 50. 
            For 3D, keep this large as these are expensive evaluations and are only needed in few time steps.
        
        seed : int, optional, optional
            Seed for the random initialization. Change for different outcomes.
        
        initial_state : ndarray, optional
            Initialized state. If None, creates a random one with self.seed 

        dt : float
            Time step for each time iteration. Keep this small for stability. 
            NOTE: Different chemical potentials require different dt as the stiffness of the PDE is different for each case.
        
        additional_func : list
            List of additional perturbative function to the `func` and its arguments. The list must be arranged as [function_name, argument1, argument2, ...]. Default : None

        additional_mufunc : python function
            Perturbative python function for the chemical potential. Arguments and keyword arguments can be passed through args and kwargs. Default : None 
        """
        N = int(self.BoxSize[0]/self.resolution)
        phase_field = PhaseField(N = N, p = 1 - 2*self.volumefraction, D = D, num_iter = num_iter, dt=dt, frame_iter = frame_iter, is_3D = self.is_3D, resolution = self.resolution, seed=seed, initial_state = initial_state)
        phase_field.solve(additional_mufunc=additional_mufunc, additional_func=additional_func,
                        compute_hyperuniformity=compute_hyperuniformity,target_dmean = self.interpore_distance, *args,**kwargs)
        return phase_field.final_state