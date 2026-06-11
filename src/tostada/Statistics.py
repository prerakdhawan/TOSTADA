import numpy as np
from scipy.stats import binned_statistic,gaussian_kde,norm
from skimage.morphology import ball,disk,dilation,binary_erosion
from skimage import measure, draw
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt,zoom
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
    
def fwhm_and_H(array,hud_class = False,pad=2,roi=20,q_ind_max = 4,fwhm=True, smooth = False, smooth_window = 5, peak_factor=0.5, fit_param=False):
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
    def moving_average(y, w):
        if w <= 1:
            return y.copy()
        kernel = np.ones(w, dtype=float) / float(w)
        ys = np.convolve(y, kernel, mode="same")
        return ys
    
    def powerlaw(q, A, alpha):
        return A * q**alpha

    def linearlaw(q,A,c):
            return A*q + c
    
    array = array[(np.isnan(array)==False)[:,0]]
    pad = pad # M pixels away from zeroth peak 
    q = array[pad:q_ind_max,0]
    y = array[pad:q_ind_max,1]
    y_peak = moving_average(y, smooth_window) if smooth else y.copy()
    popt, _ = curve_fit(
                        linearlaw, 
                        q,
                        y_peak,
                        p0=(1,2),
                        bounds=([0.0, 0.0], [np.inf, 8.0]),
                        maxfev=20000,
                    )

    H = popt[1]/np.max(array[pad:,1])
    if hud_class==True:
        popt_class, _ = curve_fit(
                    powerlaw, 
                    q,
                    y_peak,
                    p0=(1,2),
                    bounds=([0.0, 0.0], [np.inf, 8.0]),
                    maxfev=20000)
        HU_data = np.array([H,popt_class[0],popt_class[1]])
    else:
        HU_data = H

    if (fit_param==True):
        print ('fit parameters for linear trend = {p}'.format(p=popt))
        if (hud_class==True):
            print ('fit parameters for class trend = {p}'.format(p=popt_class))
    max_value = np.max(array[pad:,1])
    max_ind = np.where(array==max_value)[0]
    ratio_max = peak_factor * max_value 
    indices = np.where(array[max(0,max_ind[0]-roi):max_ind[0]+roi,1] >= ratio_max)[0]
    #H = array[min_ind,1]/array[max_ind,1] #Xqmin/max_value
    if (fwhm==True):
        if len(indices) > 1:
            fwhm = array[max(0,max_ind[0]-roi):max_ind[0]+roi,0][indices[-1]] - array[max(0,max_ind[0]-roi):max_ind[0]+roi,0][indices[0]]
        return np.append(fwhm,HU_data)
    else:
        return HU_data

def dmean_from_qpeak(array,factor=np.sqrt(3)/2):
    """
    Estimate mean inter- pore/object distance `dmean` from characteristic peak in the reciprocal space.
    
    
    Parameters
    ----------
    array : 2xN array
        1D spectral density. Could be passed from angular_average() or separately. 

    factor : float
        Factor by which the 2pi/q0 should be scaled to get dmean.
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

def Sq_analytic(q,dmean,param,option=['ginibre','fourier_dual_ocp','hermite_gaussian','anti_hud']):
    """
    Analytic functions for structure factor. Can be used to generate a point distribution with prescribed S(q) using tostada.Optimization.
    """
    def ginibre(q,param=None):
        return 1 - np.exp(-(dmean*q)**2/(4*np.pi))
    def fourier_dual_ocp(q,param=None):
        return 1 - np.exp(-dmean * q/(np.pi))
    def hermite_gaussian(q,param=1/15): 
        return 1 + param*np.sqrt(2*np.pi)*(-4*(dmean*q)**4 + 12*(dmean*q)**2 -3)*np.exp(-(dmean*q)**2/2)
    def anti_hud(q,param=None):
        return 1 + 1/(dmean * q)
    Sq_functions = {
        'ginibre': ginibre,
        'fourier_dual_ocp': fourier_dual_ocp,
        'hermite_gaussian': hermite_gaussian,
        'anti_hud' : anti_hud
    }

    if option not in Sq_functions:
        raise ValueError(f"Unknown option '{option}'. Valid options are: {list(Sq_functions.keys())}")

    return Sq_functions[option](q,param)

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

def yukawa_potential_hard(x, eps, R1, R2, kappa, r_c):
    """
    Hard-sphere Yukawa potential function (DLVO). 

    Parameters
    ----------
    x : float or ndarray  
        Distance of the particle-pair. 
    eps : float
        Strength of the interaction.
    R1 : float
        Diameter of particle 1
    R2 : float 
        Diameter of particle 2
    kappa : float
        Inverse screening length of the interaction. For long-range interactions, kappa should be small (1/kappa is longer)
    r_c : float
        Critical length beyond which the distances are ignored
 
    """
    Dij = R1+R2
    return np.piecewise(
        x,
        [x <= Dij, (x >= Dij) & (x <= r_c), x > r_c],
        [1e10,
         lambda x: eps*(((np.exp(-kappa * (x - Dij)))/(x/Dij)) - ((np.exp(-kappa*(r_c - Dij)))/(r_c/Dij))),
         0])

def harmonic_potential(pos,A,r0=0,axis=2):
    """
    Trapping potential for the particles along the direction of the axis. 

    Parameters
    ----------

    pos :  float or ndarray
        Position of the particle
    
    A : float
        Strength of the trapping potential
    
    r0 : float 
        Position along the axis where potential is minimum.

    axis : int
        Axis along which the potential is applied. Axis=0 for x-direction, 1 for y-direction and 2 for z-direction. Default : z-direction.
    
    """
    coord = pos[axis]
    return 0.5 * A * (coord-r0)**2

class Morphology:
    """
    Evaluate the morphological properties of the distribution. Discrete objects/voids are detected by first finding contours and later approximating each with a closed polygon.
    Can be used to obtain center-of-mass of each arbitrarily shaped object/void, their interfacial length (perimeter) and the shape descriptors capturing the local s-fold symmetries in the system. 
    ! Currently, only implemented/tested for 2D distributions. Future release will also have area and curvature of each object/void.

    Parameters
    ----------

    distribution : tostada.PhaseDistribution object
        Distribution for which the morphological parameters are to be evaluated. 
        Both `PointDistribution` and `PhaseDistribution` can create a `Morphology` object through a `get_morphological_parameters()` method.

    smax : int
        Maximum order until which the structure metrics are to be evaluated. For example, smax=6 captures q0, q1, ..., q6. 
    
    """
    def __init__(self, distribution=None,smax=6,**kwargs):
        self.distribution = distribution
        self.smax=smax
        self.positions, self.polygons,self.psi, self.orientations, self.properties = self.morphology(**kwargs)

    def morphology(self,**kwargs):
        polygons,positions,regionprops = self.distribution.get_morphological_parameters(**kwargs)
        psi,orientations = self.structure_metrics(polygons)
        return positions, polygons, psi, orientations, regionprops
    
    @staticmethod
    def polygon_normals_and_lengths(polygon):
        """
        Quantifies the interface (boundary in 1D) of a polygon by evaluating the length of each segment and the angle its corresponding normal makes with the horizontal axis.
        This is used in `structure_metrics` to decompose the obtained quantity into irreducible representation of the rotation group.

        Parameters
        ----------

        polygon : N x 2 array
            Array of points comprising the input polygon. If using a `PhaseDistribution`, this can be the shape of detected objects/pores.

        Returns
        -------

        out : N x 2 array
            Array where the zeroth column contains length of each edge and first column are the outward normal angles in radians.

        """
        coords = polygon
        N = len(coords) #- 1  # last point = first point
        out = []
        for i in range(N):
            i1 = i
            i2 = i+1 if (i<N-1) else 0

            x0, y0 = coords[i1]
            x1, y1 = coords[i2]
            dx = x1 - x0
            dy = y1 - y0
            L = np.hypot(dx, dy)
            if L < 1e-12:
                continue
            # outward normal: if poly is CCW, outward normal is (dy, -dx) / L
            nx =  dy / L
            ny = -dx / L
            phi = np.arctan2(ny, nx)  # angle in [-π, +π]
            if phi < 0:
                phi += 2*np.pi
            out.append((L, phi))
        return np.array(out)

    def structure_metrics(self,polygons):
        """
        Evaluates the normalized Minkowski Structure Metrics for each object/void in the PhaseDistribution. 
        Each structure metric q0,q1,...,qs qualitatively captures the s-fold symmetry. For more details, see https://morphometry.org/theory/anisotropy-analysis-by-imt. 

        Returns
        -------

        psi : M x smax array
            Minkowski structure metrics up till the order `smax` for each M polygons. psi[:,0] contains perimeter of each polygon.
        """

        s = np.arange(self.smax+1)
        num_poly = len(polygons)
        psi = np.zeros([num_poly,s.shape[0]])
        thetas = np.zeros([num_poly,s.shape[0]])
        for i in range(num_poly):
            poly = polygons[i].reshape(-1,2)
            den = self.polygon_normals_and_lengths(poly)
            psi_ = np.sum(den[:,0]*np.exp(1j*np.outer(s,den[:,1])),axis=1)
            psi[i,:] = np.abs(psi_)/np.abs(psi_[0])
            psi[i,0] = np.abs(psi_[0])
            thetas[i,:] = np.nan_to_num(np.angle(psi_)/s)
        return psi,thetas

    def misorientation_angles(self, k_neighbors=6, target=np.pi/3):
        """
        Computes the mean angle of displacement for the given PointDistribution or PhaseDistribution. Can be used to study grain-boundaries in the system.
        """
        tree = cKDTree(self.positions)
        # FIXED: dists first, then idx
        pairlist = tree.query(self.positions, k=k_neighbors+1)[1]
        misorientation_metric = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            # Use angular sort of k-nearest (not radius) for consistency
            neigh_idx = pairlist[i, 1:k_neighbors+1]
            if len(neigh_idx) < 3: continue
            vecs = self.positions[neigh_idx] - self.positions[i]
            angles = np.arctan2(vecs[:,1], vecs[:,0])  # angles with respect to x-axis
            order = np.argsort(angles)
            sorted_angles = angles[order]
            diffs = np.diff(sorted_angles) # 3-point angles
            diffs = np.append(diffs, sorted_angles[0] + 2*np.pi - sorted_angles[-1])
            misorientation_metric[i] = np.mean(np.abs(diffs - target)) / target
        return misorientation_metric

    def get_local_properties(self,options=['area','euler_number','orientation','eccentricity','perimeter']):
        """
        Gives the local properties of each detected region. Essentially a wrapper around skimage's regionprop.

        Parameters
        ----------
        options : str
            Quantity that needs to be evaluated for each detected morphological object. Possible options are:
                area : area of each object
                euler number : connectivity of the objects
                eccentricity : anisotropy of the objects
                perimeter : perimeter of the objects

        Returns
        -------
        prop_array : M x 1 array
            Array of M scalar values for each detected object. 
        """
        prop_array = np.array([getattr(p,options) for p in self.properties])
        return prop_array

    def get_morphology_stats(self,ax=None,skipind=1,orientations=False,plot_results=True,**kwargs):
        """
        Morphological statistics of the detected shapes. Takes mean and standard deviation of each Minkowski structure metrics and plots a bar chart for summary.
        The mean and standard deviation are later accesible as `.mean_stats` and `.std_stats`, respectively.

        Parameters
        ----------

        skipind : int
            Number of initial polygons to be skipped (first one or two are generally the simulation domain itself)

        orientations : bool
            If True, plots the orientation statistics
        Returns
        -------

        ax : matplotlib.axes, optional
            Matplotlib axes containing the detected polygons. Can be used to combine several plots. If not provided, creates a new figure.

        """
        self.mean_stats = np.mean(self.psi[skipind:,:],axis=0)
        self.std_stats = np.std(self.psi[skipind:,:],axis=0)
        self.mean_stats_orientations = np.mean(self.orientations[skipind:,:],axis=0)
        self.std_stats_orientations = np.std(self.orientations[skipind:,:],axis=0)
        if plot_results==True:
            if ax is None:
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(1,1,1)
            print ('Mean interfacial length = {m}+/-{st}'.format(m=self.mean_stats[0],st=self.std_stats[0]))
            if (orientations==True):
                ax.bar(np.arange(self.smax+1)[1:],self.mean_stats_orientations[1:],width=0.2,yerr=self.std_stats_orientations[1:],**kwargs)
            else:
                ax.bar(np.arange(self.smax+1)[1:],self.mean_stats[1:],width=0.2,yerr=self.std_stats[1:],**kwargs)
            #ax.set_ylim(0,1)
            ax.set_xlabel('Structure metrics $q_m$',fontsize=14)
            return ax
        else:
            return self.mean_stats,self.std_stats
        
    def draw_polygon(self,resolution,polygon_ind, shape, shift_vec=[0,0]):
            """
            Draw a given polygon detected by the self.polygons. Creates a two-phase media such that I(polygon) = 1 and I(~polygon) = 0.

            """
            label_img = np.zeros(shape, dtype=np.int32)
            if type(polygon_ind) == int:
                polygon = np.asarray(self.polygons[polygon_ind]).squeeze() + shift_vec
                rr, cc = draw.polygon( (polygon[:, 0] )/resolution, (polygon[:, 1] ) /resolution, shape=shape)
                label_img[rr, cc] = 1
            else:
                for i in range(len(polygon_ind)):
                    polygon = np.asarray(self.polygons[polygon_ind[i]]).squeeze() + shift_vec
                    rr, cc = draw.polygon( (polygon[:, 0] )/resolution, (polygon[:, 1] ) /resolution, shape=shape)
                    label_img[rr, cc] = 1
            return label_img