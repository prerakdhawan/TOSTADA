import numpy as np
import matplotlib.pyplot as plt
import os
import re
import tifffile as tiff  
import pickle
from scipy.interpolate import interp1d,RegularGridInterpolator

#import color_system as csys

def read_file(folder_path, keyword, N):
    """
    Reads the Nth file from folder_path with the given keyword. If no keyword and/or N is passed, the 0th file from the folder is picked. 
    Makes it convenient to read files generated from a parametric study. Loading is according to the file extension. 
    Currently supports `tiff`, `png`, `npy`, `npz`, `csv` and `pkl` (from .save() functions of `PointDistribution` and `PhaseDistribution`).
    """
    def extract_number(filename):
        """
        Regex for extracting a given number (negative or positive, float or integer or scientific notation) from a given string.
        """
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", filename)
        return float(match.group()) if match else float('inf')
    
    all_files = os.listdir(folder_path)
    filtered_files = [f for f in all_files if keyword in f]
    print ('Found {n} files with the given keyword'.format(n=len(filtered_files)) )
    # Sort numerically by number in filename
    #filtered_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    filtered_files.sort(key=extract_number)
    selected_file = filtered_files[N]
    image_path = os.path.join(folder_path, selected_file)
    
    print('Filtered & sorted files =', filtered_files)
    print('Selected File =', selected_file)

    # Load based on file extension
    ext = os.path.splitext(selected_file)[1].lower()
    
    if ext == '.npy':
        image = np.load(image_path)
    elif ext == '.npz':
        image = np.load(image_path,allow_pickle=True)  # returns a NpzFile object
    elif ext == '.csv':
        image = np.loadtxt(image_path, delimiter=',')
    elif ext in ('.tif', '.tiff', '.png'):
        image = tiff.imread(image_path)
    elif ext == '.pkl':
        with open(image_path, 'rb') as f:
            image = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")    
    return image, selected_file

def save_distribution(filename):
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(f)
    print ('File saved with filename {f}'.format(filename+'.pkl'))

def load_distribution(filename):
    with open(filename+'.pkl', 'rb') as f:
        return pickle.load(f)
    print ('File loaded with filename {f}'.format(filename+'.pkl'))


class Spectrum:
    def __init__(self, wavelength, spectra, distribution=None,kmax=50,form_factor=None):
        """
        Define properties of a scattering response as a function of wave-vectors (not only a function of spatial frequencies but also of temporal frequencies).

        """
        self.wavelength = wavelength # in microns
        self.spectra = spectra
        self.distribution = distribution
        self.Sq_int, self.Sq1d_int = [None,None] if self.distribution is None else self.interpolate_kspace(kmax,dkx=2*np.pi/self.distribution.Lx,form_factor=form_factor)
        
    def interpolate_kspace(self,kmax,dkx,form_factor):
        """
        Interpolates the k-space response of the system. If the form_factor is not specified, assumes scattering from only structure factor.
        TODO : test for form-factor functionality.

        Parameters
        ----------
        kmax : float
            Maximum reciprocal-space wave vector.

        dkx : float, optional
            Reciprocal-space resolution. If unspecified, uses 2pi/L.
        
        form_factor : ndarray, optional
            Form factor f(qx,qy) of the individual particle response. Should be of the same shape as S(q). If unspecified, uses just S(q)

        Returns
        -------
        interp_Sq : scipy.RegularGridInterpolator
            Scipy interpolation object in 2D

        interp_Sq1d : scipy.interp1d
            Scipy interpolation object in 1D
        """
        Sq_ = self.distribution.ReciprocalSpace(kmax=kmax,dkx=dkx)
        Sq = Sq_[0]
        Interp_array = Sq[-1] if form_factor is None else Sq[-1]*form_factor
        Sq_averaged = Sq_[1]
        Kx_axis = np.array(Sq[0][0, :]) 
        Ky_axis = np.array(Sq[1][:, 0]) 
        interp_Sq = RegularGridInterpolator(
            (Ky_axis, Kx_axis),  
            np.array(Interp_array),
            bounds_error=False,
            fill_value=0.0  
            )
        interp_Sq1d = interp1d(Sq_averaged[:,0],Sq_averaged[:,1])
        return interp_Sq,interp_Sq1d
    
    def Angular_spectrum(self,theta_res=200):
        """
        Determines the angular spectrum of a system. It yields the scattering response I(\theta,\phi,\lambda). 
        Currently, the form-factor is not included, so I(\theta,\phi,\lambda) = S(\theta,\phi,\lambda).

        Parameters
        ----------

        theta_res : int
            Sampling frequency for polar angle \theta.
        
        Returns
        -------

        I_array : N_wvl x theta_res x theta_res
            2D Scattering response as a function of each wavelength. It essentially maps the incident wavevector (2\pi/wavelength) to kx,ky grid.

        I1d_array : N_wvl x N_theta
            Angular-averaged scattering response as a function of theta and wavelength.
        """
        theta_array = np.linspace(-np.pi/2,np.pi/2,theta_res)
        I_array = np.zeros([self.wavelength.shape[0],int(theta_array.shape[0]),int(theta_array.shape[0])])
        I1d_array = []
        for i in range(self.wavelength.shape[0]):
            kmod = (2*np.pi/(self.wavelength[i]))*np.sin(theta_array)
            karray = np.meshgrid(kmod,kmod)
            k_array = np.stack([karray[0].ravel(), karray[1].ravel()], axis=-1)
            k_int = kmod[kmod>0]
            k_int = np.where(k_int<np.min(self.distribution.Sq_averaged[:,0]),np.min(self.distribution.Sq_averaged[:,0]),k_int)
            
            S1dnew = self.Sq1d_int(k_int)
            Sqnew = self.Sq_int(k_array)

            I1d_array.append(S1dnew)
            I_array[i,:,:] = Sqnew.reshape(karray[0].shape) / np.max(Sqnew)

        I1d_array = np.asarray(I1d_array)
        return I_array,I1d_array
    


