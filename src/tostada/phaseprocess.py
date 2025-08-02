from tostada.PointDistribution import PointDistribution
from tostada.PhaseDistribution import PhaseDistribution
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

    def load_distribution(self,folder_path, keyword='', N=0):
        """
        Loads the 'N'th file from the input 'folder_path' with filename having the given 'keyword' and converts to a PhaseDistribution object. 
        Uses the resolution that is either already present in the file (in npz format) or the user-provided.
        Note: Similar to load_distribution method in pointprocess. See Utility.py for details into the function.

        Parameters
        ----------
        folder_path : string
            Path of the folder. Eg: /home/user/tostada/Examples or wherever the files are stored.
        keyword : string
            Particular keyword in the file. If not provided, takes the N=0 file from the folder. 
        N : int
            Nth file from the folder with the given keyword. Useful for parametric loading of files.

        Returns
        -------

        PhaseDistribution object
        """
        data,filename = tostada.util.Utility.read_file(folder_path, keyword, N)

        if 'resolution' in data:
            #BoxSize = data['BoxSize']
            resolution = data['resolution'] 
        else:
            resolution = self.resolution
            print ('Resolution not found in file. Taking resolution={r}'.format(r=self.resolution))
        if 'image' in data:
            image = data['image']
        else:
            image = data
        return PhaseDistribution(image=image,resolution=resolution)

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
                                            animate=False, binarize = False):
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
            How frequently should the quantities be saved

        seed : int
            Seed for random numbers for initialization. 

        animate : bool
            Whether an animated gif be created for time-evolution of phases. Default : False

        binarize : bool
            Whether the final distribution be binarized strictly to 0s and 1s. Default : False

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
                        resolution = self.resolution, seed=seed)
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