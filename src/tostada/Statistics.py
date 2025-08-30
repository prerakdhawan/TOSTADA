import numpy as np
from scipy.stats import norm
from scipy.stats import binned_statistic,gaussian_kde
from skimage.morphology import ball,disk,dilation,binary_erosion
from scipy.ndimage import distance_transform_edt,zoom

def angular_average(data,dkx):
    """
    1D-Angular average of any object in real or reciprocal space. Can be used for Phase,Point,Texture distribution. 
    Should be of the form data[0],data[1],.., = mapping quantites (real space or reciprocal space) and data[-1]=quantity to be averaged

    Parameters
    ----------
    
    data : numpy.Ndarray
        Averaging dataset of the form N x Ndarray
    dkx : float
        Resolution for averaging 
    
    Returns
    -------
    data_avg : 2xN array
        Averaged quantity where 1st column is the mappable and 2nd column is angular averaged quantity. 
    """
    nkx=data[0].shape[0] 
    Q = np.linalg.norm(data[:data[-1].ndim],axis=0).flatten() #NxNxN if image is 3D else NxN
    #k1d=np.sqrt(np.square(X0)+np.square(Y0)).flatten()
    S1d=data[-1].flatten()#S0.flatten()
    data_avg=np.zeros((len(Q),2))
    data_avg[:,0]=Q
    data_avg[:,1]=S1d
    k1dbins=np.arange(dkx,(nkx+0.5)*dkx,dkx) #1D bins
    S_ks,S_kk,binindex=binned_statistic(data_avg[:,0],data_avg[:,1],statistic="mean",bins=k1dbins)
    S_kk,S_kktmp,binindextmp=binned_statistic(data_avg[:,0],data_avg[:,0],statistic="mean",bins=k1dbins)
    data_avg=np.zeros((len(k1dbins)-2,2))
    data_avg[:,0]=S_kk[:-1]
    data_avg[:,1]=S_ks[:-1]
    return data_avg

def fwhm_and_H(array,pad=2,roi=20):
    """
    Computes Full-width at Half Maximum (FWHM) and Hyperuniformity index H of the reciprocal data from a 1D SpectralDensity. Useful only for HuD type process.
    H is simply the smallest X(q) / largest X(q) and describes how strongly the structure is hyperuniform. 
    FWHM describes the "ordered-ness" of the structure. Smaller value = More periodic/amorphous. Extracted from the structural peak. 

    Parameters
    ----------
    array : 2xN array
        1D spectral density. Could be passed from angular_average() or separately. 
    
    pad : int
        M pixels to be ignored from q=0. Default : 2
    
    roi : int
        M pixels around the central peak that need to be considered for FWHM evaluation.

    Returns
    -------
    fwhm : float
        Spectral width 
    H : float
        Hyperuniformity index
    """
    array = array[(np.isnan(array)==False)[:,0]]
    pad = pad # M pixels away from zeroth peak (since X(0)=N)
    max_value = np.max(array[pad:,1])
    max_ind = np.where(array==max_value)[0]
    min_ind = np.where(array==np.min(array[pad:max_ind[0],1] ) )[0]
    #print (min_ind,max_ind)
    #print (array[min_ind],array[max_ind])
    half_max = max_value / 2
    indices = np.where(array[max(0,max_ind[0]-roi):max_ind[0]+roi,1] >= half_max)[0]
    #print (array[max_ind[0]-max(0,roi):max_ind[0]+roi,:][indices])
    if len(indices) > 1:
        fwhm = array[max(0,max_ind[0]-roi):max_ind[0]+roi,0][indices[-1]] - array[max(0,max_ind[0]-roi):max_ind[0]+roi,0][indices[0]]
    
    H = array[min_ind,1]/array[max_ind,1] #Xqmin/max_value
    return fwhm,H

def dmean_from_qpeak(array,factor=np.sqrt(3)/2):
    """
    Estimate mean inter- pore/object distance `dmean` from characteristic peak in the reciprocal space.
    
    
    Parameters
    ----------
    array : 2xN array
        1D spectral density. Could be passed from angular_average() or separately. 

    factor : float
        Factor by which the 2\pi/q0 should be scaled to get dmean.
        If the peak is at q0, factor = 1 represents system closer to square lattice and (sqrt(3)/2) closest to hexagonal lattice.

    Returns
    -------
    dmean : float
        Mean center to center distance of objects/pores.
        
    """
    array = array[(np.isnan(array)==False)[:,0]]
    max_value = np.max(array[:,1])
    index = np.where(array[:,1]==max_value)[0]
    q_peak = array[:,0][index]
    dmean = factor * (2*np.pi/q_peak)
    return dmean

def SphericalContactDistribution(image,resolution):
    """
    Relevant only for PhaseDistribution. Computes Spherical Contact Distribution for a porous media. H(r) : [0,infinity) -> [0,1] 
    For each r>=0, the value of H(r) is the conditional probability that the minimum distance from a randomly selected point of the pore phase to the solid phase is less or equal than r. 
    H(r) is computed using an increasing.

    Parameters
    ----------
    Image : numpy.Ndarray
        Image data (2D or 3D).  
    resolution : float
        Resolution of the pixel/voxel.

    Returns
    -------
    r,H(r) : 2xN numpy.Ndarray
        First column is the radius of the spheres. Second column is the Spherical contact distribution.
    """

    def compute_eroded_fraction(microstructure, max_r):
        H_r = []
        epsilon = np.mean(microstructure)
        pore_phase = (microstructure == 0)  # Binary mask for pores
        #total_pore_voxels = np.sum(pore_phase)
        #r_array = np.linspace(0,max_r+1,100)
        for r in range(0, max_r + 1):
        #for r in range(r_array)
            if (image.ndim==2):
                struct_elem = disk(r)  # Create spherical structuring element. Disk if microstructure is 2D
            else:
                struct_elem = ball(r) #if microstructure is 3D
            eroded_pores = binary_erosion(pore_phase, struct_elem)  # Perform erosion
            if (r==1):
                footprint_single = np.array([[True]])
                eroded_pores = binary_erosion(pore_phase,footprint=footprint_single)
            eroded_volume_fraction = np.mean(eroded_pores)#np.sum(eroded_pores) / total_pore_voxels  # Normalized erosion volume
            print ('ball radius={r}. Remaining vol={v}'.format(r=r,v=eroded_volume_fraction))
            H_r.append(1 - (eroded_volume_fraction / (1 - epsilon)))
        return np.array(H_r)
    
    max_radius = int(np.max(distance_transform_edt(image == 0) ) )  # Maximum meaningful radius
    H_r_values = compute_eroded_fraction(image, max_radius)
    R = np.arange(H_r_values.shape[0])*resolution
    return np.c_[R,H_r_values]

def rho_analytic(h,param,option=['gaussian','hyperbolic','sinc','exponential','stable','spherical']):
    """
    Analytic functions for correlation functions. Can be used to generate a gaussian random field which can then be used for a texture or a microstructure.

    """
    def gaussian(param,h):
        return np.exp(-param*h**2)
    def hyperbolic(param,h):
        return 1/(1+(param*h))
    def sinc(param,h): #param here is the correlation length
        return np.sinc((2*np.pi/param)*h/np.pi)
    def exponential(param,h):
        return np.exp(-param*h)
    def stable(param,h):
        return np.exp(-h*np.sqrt(param))
    def spherical(param,h):
        return 1 - (3/2)*(param*h) + (1/2)*param**3*h**3
    rho_functions = {
        'gaussian': gaussian,
        'hyperbolic': hyperbolic,
        'sinc': sinc,
        'exponential': exponential,
        'stable': stable,
        'spherical': spherical
    }

    if option not in rho_functions:
        raise ValueError(f"Unknown option '{option}'. Valid options are: {list(rho_functions.keys())}")

    return rho_functions[option](param,h)


def RSA_distribution_function(x,mean,var,scale=1):
    """
    Decides the sticking probability of a particle based on distance-dependent thinning rule of Random Sequential Adsorption process.
    """
    return np.piecewise(x,[x<mean,x>=mean],[lambda x: scale*np.exp(-(x-mean)**2/(2*var**2)),lambda x: scale*1])
        

def MaternIII_distribution_function(x, alpha, beta=None):
    """
    Decides the sticking probability of a particle based on distance-dependent thinning rule of soft-core Matern type III process. 
    If beta=None, assumes beta=alpha (hard-core process)
    """
    beta = alpha if beta is None else beta
    return np.piecewise(
        x,
        [x < alpha, (x >= alpha) & (x < beta), x >= beta],
        [1,
         lambda x: (beta - x) / (beta - alpha),
         0]
    )

