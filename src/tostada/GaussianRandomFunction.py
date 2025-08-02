import tostada.Statistics as stats
#import importlib
#importlib.reload(stats)
import numpy as np
from scipy.stats import norm

def generate_grf(max_dist,dx,correlation_length,param,is_3D=False,options=['gaussian']):
    """
    Generate a gaussian random field X(r) in 2D or 3D.
    
    Parameters
    ----------
    max_dist : list of floats
        Size of the domain along each axis in 2D or 3D.
    dx : float
        Resolution of the fields.
    correlation_length : float
        Controls the spatial correlation of the field.
    param : list of floats
        parameters defining the perturbations to the perfect sinc correlation. 
    is_3D : Boolean
        2D or 3D domain. 
    options : list of strings
        options for covariances for each param. See Statistics.py for reference.
    Returns:
        np.ndarray: A 3D binary array representing the porous media (1 for solid, 0 for pore).
    """
    
    x = np.arange(-max_dist[0]/2,max_dist[0]/2,dx)
    y = np.arange(-max_dist[1]/2,max_dist[1]/2,dx)
    
    if (is_3D==True):
        z = np.arange(-max_dist[2]/2,max_dist[2]/2,dx)
        dd = np.meshgrid(y,x,z)
    else:
        dd = np.meshgrid(y,x)
    dd = np.linalg.norm(dd,axis=0)
    print (dd.shape)
    rho = stats.rho_analytic(dd,correlation_length,option='sinc') #builds rho with a prescribed correlation length
    for opt in enumerate(options):
        rho = rho*stats.rho_analytic(dd,param[opt[0]],opt[1])

    #power_spectrum = np.abs(np.fft.fftn(stats.rho_analytic(dd,params)  ))
    power_spectrum = np.abs(np.fft.fftn(rho))
    # Generate random Gaussian noise in Fourier space
    random_field = ((np.random.normal(size=dd.shape,loc=0,scale=1 ))) #/np.size(power_spectrum)
    # Apply the power spectrum to introduce spatial correlations
    correlated_field = np.real((np.fft.ifftn( (random_field) * (power_spectrum))))
    correlated_field = (correlated_field - (np.mean(correlated_field)*(1)) )/np.std(correlated_field)
    return correlated_field
    