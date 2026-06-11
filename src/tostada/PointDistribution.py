import numpy as np
from scipy.spatial.distance import cdist
from tostada.PhaseDistribution import PhaseDistribution
import tostada.Statistics as stats
from tostada.Statistics import Morphology
from scipy.stats import binned_statistic
import jax
import jax.numpy as jnp
import scipy.special as ss
from skimage.morphology import disk,ball,binary_dilation
from scipy.spatial import cKDTree
from skimage.draw import line,polygon
from scipy.spatial import Voronoi
import pickle
from skimage import measure, draw
from matplotlib.path import Path

try:
    import cupy as cp
    from cupyx.scipy.ndimage import binary_erosion
    cp.asarray([0])
    print ('GPU detected. Using CUDA.')
    def asnumpy(x):
        return cp.asnumpy(x)
    
except Exception as e:
    print (f"GPU not available : {e}")
    import numpy as cp
    from scipy.ndimage import binary_erosion
    def asnumpy(x):
        return x
    
class PointDistribution:
    """
    Create a point-distribution object from a given point-cloud of particle/pore positions in 2D/3D. 

    Parameters
    ----------
    positions : N x 3 ndarray
        Positions of particles in a 2D or 3D box. 
    diameter : float
        Diameter of each particle
    BoxSize : list
        Size of the computational domain along each axes as [Lx,Ly,Lz]. If Lz=0, assumes a 2D instance.
    """
    def __init__(self, positions, diameter, BoxSize):
        self.positions = positions
        if (self.positions.shape[1]==2):
            self.positions = np.hstack([self.positions,np.zeros(self.positions.shape[0])[:,None]])
        self.diameter = diameter
        self.BoxSize = BoxSize 
        self.is_3D = np.unique(self.positions[:,2]).shape[0]>1
        self.ndim = 2*np.logical_not(self.is_3D) + 3*self.is_3D
        if (np.isscalar(self.BoxSize)==True):
            self.BoxSize = [self.BoxSize,self.BoxSize,self.is_3D*self.BoxSize]
        elif (len(self.BoxSize)==2):
            self.BoxSize = [self.BoxSize[0],self.BoxSize[1],0]
        self.Lx = np.max(self.positions[:,0]) - np.min(self.positions[:,0]) #may not be equal to BoxSize
        self.Ly = np.max(self.positions[:,1]) - np.min(self.positions[:,1]) #may not be equal to BoxSize
        self.Lz = np.max(self.positions[:,2]) - np.min(self.positions[:,2]) #will be zero if is_3D=False
        self.totalparticles = self.positions.shape[0]
        self.particledensity = self.totalparticles / (self.BoxSize[0]*self.BoxSize[1]*self.BoxSize[2]**self.is_3D)  
        self.dmean = 1/np.power(self.particledensity,1/self.ndim) #mean inter-particle distance

    def build_tree(self, is_periodic=True, z_periodicity = False):
        """
        Build a k-nearest neighbour distance tree for the given positions in 2D/3D with/without periodic boundaries.

        Parameters
        ----------
        is_periodic : bool
            Should the tree assume periodic boundaries. NOTE : Creating a periodic tree does not imply the point distribution will obey the periodic condition as it is process-dependent.
        z_periodicity : bool
            Should the system obey periodic boundaries along z-direction. NOTE : a 3D object need not necessarily obey this as the system may be finite thickness along this dimension.
        """
        positions = self.positions[:,:3] if self.is_3D else self.positions[:,:2]
        if (is_periodic):
            tree = cKDTree(positions, boxsize=[self.BoxSize[0],self.BoxSize[1],self.BoxSize[2]*z_periodicity])
        else:
            tree = cKDTree(positions)
        return tree, positions 

    def save(self, filename):
        """
        Save the all the properties of the PointDistribution in a pickled state.

        Parameters
        ----------
        filename : str
            Name of the file to be saved. The extension of the file need not be mentioned.
        """
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(self,f)
        print ('File saved with filename {f}'.format(f=filename+'.pkl'))

    @staticmethod
    def load(filename):
        """
        Load an instance of the saved PointDistribution created from PointDistribution.save().

        Parameters
        ----------
        filename : str
            Name of the file to be loaded. 
        """
        with open(filename+'.pkl', 'rb') as f:
            A = pickle.load(f)
            print ('File loaded with filename {f}'.format(f=filename+'.pkl'))
        return A 
        
    def ReciprocalSpace(self,kmax,dkx=None,**kwargs):
        dkx=2*np.pi/self.BoxSize[0] if dkx==None else dkx
        """
        Computes the reciprocal space response of the given point-distribution. For details, refer to Sk_jax_custom.
        Identical to ReciprocalSpace in PhaseDistribution. 

        Parameters
        ----------

        kmax : float
            Maximum wave-vector for the S(q) computation.
        
        dkx : float
            Reciprocal-space resolution.

        kwargs: 
        
        Additional arguments for how to compute S(q). Possible inputs:
        
        batch_size
            How to chunk the (qx,qy,qz,x,y,z) mesh for the computation. For large kmax or large spatial domains, use a large batch_size to avoid memory overload 

        Returns
        -------

        Sq: array_like
            StructureFactor (2D or 3D) along with it's reciprocal space vectors.
        Sq_averaged : array_like
            Angular-average of the StructureFactor computed in Statistics.py.

        """
        batch_size=kwargs.get('batch_size',100)
        self.Sq = self.Sk_jax_custom(kmax=kmax,batch_size_fac=batch_size)
        self.Sq_averaged = stats.angular_average(self.Sq,dkx)
        self.Sq_averaged = self.Sq_averaged[~np.isnan(self.Sq_averaged).any(axis=1)] #removes nan elements
        return self.Sq,self.Sq_averaged
    
    def RealSpaceCorrelations(self,dr=None):
        """
        Computes real-space correlation function (two-point correlation function). For details, refer to `Pair_correlation()` method.
        """
        dr = np.sqrt(2)*6*self.dmean/100 if dr is None else dr
        self.G2r = self.Pair_correlation()
        self.G2r_averaged = stats.angular_average(self.G2r,dr)
        return self.G2r,self.G2r_averaged
    
    def Hyperuniformity_data(self,hud_class=False,kmax=80,dkx=None,pad=1,roi=10,fwhm=True,**kwargs):
        """
        Spectral width (FWHM) and Hyperuniformity index of the point distribution. See Statistics.py for further details. 
        Note: FWHM may not always be meaningfull for point-distributions as a measure of spectral width here since S(q) -> 1 for q>>0
        """
        #if not hasattr(self, 'Sq_averaged'):
        q_ind_max = kwargs.get('q_ind_max',4)
        smooth = kwargs.get('smooth',False)
        smooth_window = kwargs.get('smooth_window',5)
        peak_factor = kwargs.get('peak_factor',0.5)
        fit_param = kwargs.get('fit_param',False)

        dkx=2*np.pi/self.BoxSize[0] if dkx==None else dkx
        Sq = self.ReciprocalSpace(kmax,dkx)
        Hdata = stats.fwhm_and_H(self.Sq_averaged,hud_class=hud_class,smooth=smooth,
                                 smooth_window=smooth_window,peak_factor=peak_factor,
                                 pad=pad,roi=roi,fwhm=fwhm,q_ind_max=q_ind_max,fit_param=fit_param)
        if (fwhm==True):
            print ('Spectral Width of the structural peak={p}'.format(p=Hdata[0]))
            print ('Hyperuniformity of the structure = {h}'.format(h=Hdata[1]))
        else:
            print ('Hyperuniformity of the structure = {h}'.format(h=Hdata))
        if (hud_class==True):
            print ('HUD class with alpha={a}'.format(a=Hdata[3]))
        return Hdata
    
    def Dmean_from_q(self,factor=np.sqrt(3)/2,kmax=100):
        """
        Calculate mean center-to-center distance of objects from structure factor S(q).
        """
        if not hasattr(self, 'Sq_averaged'):
            Sq = self.ReciprocalSpace(kmax=kmax)
        Dmean = stats.dmean_from_qpeak(self.Sq_averaged,factor=factor)
        print ('Mean center-to-center distance = {d}'.format(d=Dmean))
        return np.float32(Dmean)

    def get_morphological_parameters(self,**kwargs):
        pad = kwargs.get('pad',1)
        method = kwargs.get('method','voronoi')
        resolution = kwargs.get('resolution',0.01)
        if (method=='voronoi'):
            phase = self.Phaseobject_voronoi(resolution,rad=kwargs.get('rad',2),output=False)
        elif (method=='trivalent'):
            phase = self.Phaseobject_trivalent(resolution,rad=kwargs.get('rad',2),output=False)
        pad_int = int(pad/phase.resolution)
        def isin_box(v):
            return np.logical_and(np.greater(v, coords_min ), np.less(v, coords_max ) )

        coords_max = np.array([int(2*self.BoxSize[0]/resolution) + pad_int , int(2*self.BoxSize[1]/resolution) + pad_int ])
        coords_min = np.array([ int(1*self.BoxSize[0]/resolution) - pad_int , int(1*self.BoxSize[1]/resolution)  - pad_int ])
        level = kwargs.get('level',0.05)
        contours = measure.find_contours(phase.tessellate(),level)
        tol = kwargs.get('tolerance',1)
        polygons = []
        for i in range(len(contours)):
            poly = contours[i]
            mask = np.bool(np.prod(isin_box(poly),axis=1))
            if (np.bool(np.prod(mask))):
                poly = measure.approximate_polygon(poly,tolerance=tol)[:,None]*resolution
                polygons.append(poly)
        filtered = self.filter_polygons(polygons)
        label_img = np.zeros(coords_max-coords_min, dtype=np.int32)
        for i, poly in enumerate(filtered, start=1):
            poly = np.asarray(poly).squeeze()
            if poly.shape[0] < 3:
                continue
            rr, cc = draw.polygon( (poly[:, 0] - (self.BoxSize[0]-pad))/resolution, (poly[:, 1] - (self.BoxSize[1]-pad) ) /resolution, shape=label_img.shape)
            label_img[rr, cc] = i
        properties = measure.regionprops(label_img)
        positions = np.array([p.centroid for p in properties])*resolution - (pad + np.array(self.BoxSize)[:self.ndim]/2)
        selected_polygons = [arr - (3/2)*np.array(self.BoxSize)[:self.ndim] for arr in filtered]
        return selected_polygons, positions, np.array(properties)
    
    def filter_polygons(self,polygons):
        def point_in_poly(pt, poly):
            return Path(np.asarray(poly)).contains_point(pt)
        
        points = (self.positions + (3/2)*np.array(self.BoxSize)) [:,:self.ndim]
        N = self.totalparticles
        chosen = [None] * N
        used = set()
        shifted_polys = polygons
        for i, p in enumerate(points):
            found = []
            for j, poly in enumerate(shifted_polys):
                if j in used:
                    continue
                if len(poly) >= 3 and point_in_poly(p, poly[:,0,:]):
                    found.append(j)

            if len(found) == 1:
                j = found[0]
            elif len(found) > 1:
                # pick the polygon with centroid closest to the point
                d2 = []
                for j in found:
                    cen = shifted_polys[j].mean(axis=0)
                    d2.append(np.sum((cen - p)**2))
                j = found[int(np.argmin(d2))]
            else:
                # fallback: nearest polygon centroid
                d2 = []
                valid = []
                for j, poly in enumerate(shifted_polys):
                    if j in used or len(poly) < 3:
                        continue
                    cen = poly.mean(axis=0)
                    valid.append(j)
                    d2.append(np.sum((cen - p)**2))
                j = valid[int(np.argmin(d2))]

            chosen[i] = shifted_polys[j]
            used.add(j)
        return chosen
    
    def compute_morphology(self,smax=6,**kwargs):
        """
        Construct a `Morphology` object containing the shape properties of the objects/voids. 
        Currently, yields center-of-mass (comparable to `self.positions`) of each arbitrarily shaped object/void, their interfacial length (perimeter) 
        and the shape descriptors capturing the local s-fold symmetries in the system. See tostada.Statistics for further details.

        Parameters
        ----------

        smax : int
            Maximum order until which the structure metrics are to be evaluated. For example, smax=6 captures q0, q1, ..., q6. 

        Returns
        -------

        Mor : Morphology object
            This object contains information about the center-of-masses of detected objects, their coordinates as well as their normalized Minkowski structure metrics.
            The detected objects can be visualized using `tostada.Visualize.plot_detected_polygons()`
        
        """

        Mor = Morphology(self,smax,**kwargs)
        self.morphology = Mor
        print ('Morphology object created')
        return Mor
    
    #Delete this after checking Optimization.py. Now implemented in Statistics.py        
    def angular_average(self,XYS,dkx):
        """
        Parameters
        ----------
        X : ...
        Y : ...
        S : ...

        Returns
        -------
        angular average of structure factor, excluding kx,ky=0 components.
        """
        
        X0=XYS[0]
        Y0=XYS[1]
        S0=XYS[2]
        
        nkx=X0.shape[0] #same shape as the x-axis of the 2D plot
        
        k1d=np.sqrt(np.square(X0)+np.square(Y0)).flatten()
        S1d=S0.flatten()
        S_k=np.zeros((len(k1d),2))
        S_k[:,0]=k1d
        S_k[:,1]=S1d

        k1dbins=np.arange(dkx,(nkx+0.5)*dkx,dkx) #1D bins
        S_ks,S_kk,binindex=binned_statistic(S_k[:,0],S_k[:,1],statistic="mean",bins=k1dbins)
        S_kk,S_kktmp,binindextmp=binned_statistic(S_k[:,0],S_k[:,0],statistic="mean",bins=k1dbins)
        
        S_k=np.zeros((len(k1dbins)-2,2))
        S_k[:,0]=S_kk[:-1]
        S_k[:,1]=S_ks[:-1]

        return S_k
    
    def tessellate(self, copies=1, cell_size=None, return_dist=False):
        """
        Tessellate M copies of N particles in 2D/3D. Useful for periodic boundaries. Similar to tesselate in PhaseDistribution.
        
        Parameters
        ----------
        copies : int
            Number of copies along each dimension
        cell_size : list
            List of length self.ndim containing the unit cell sizes for each dimension. If None, uses BoxSize[0],BoxSize[1],BoxSize[2] for translations.
                    
        Returns
        -------    
        tesselated : numpy.ndarray
            Array of shape ((2*copies+1)**ndim * N, 3) of tessellated positions.
        """
        cell_size = cell_size if cell_size is not None else self.BoxSize
        #cell_sizes = np.array(cell_sizes)
        ranges = [np.arange(-copies, copies + 1) for _ in range(self.ndim)]
        grid = np.array(np.meshgrid(*ranges))
        shift_indices = grid.reshape(self.ndim, -1).T  # Each row represents a combination of indices.
        shifts = shift_indices * cell_size[:self.ndim]
        tessellated = self.positions[None, :, :self.ndim] + shifts[:, None, :]  
        tessellated = tessellated.reshape(-1, self.ndim)  
        if (self.ndim==2):
            tessellated = np.column_stack([tessellated,np.zeros(tessellated.shape[0])])
        if (return_dist==True):
            tessellated = PointDistribution(tessellated,diameter=self.diameter,BoxSize=list(self.tesselate_BoxSize()) )
        return tessellated
    
    def tesselate_BoxSize(self,copies=1):
        """
        BoxSize after tesselating M copies of original box.

        Parameters
        ----------
        copies : int
            Number of copies along each dimension

        Returns
        -------
        tesselated_box : numpy.ndarray
            New boxSize after tesselation. NewBoxSize = (2*copies + 1) * [Lx,Ly,Lz]
        """
        return (2*copies + 1) * np.array(self.BoxSize)


    def Hermitiansymmetry(self,data):
        """
        Uses Hermitian Symmetry for creating full data in 2D/3D. Currently only used for StructureFactor for S(-k) = S(k). 
        For mapping vectors (without the symmetry), the signs are flipped.
        Parameters:
        data (numpy.ndarray): half-arrays to be extruded with symmetry condition. data[0],data[1],... are mapping vectors and data[-1] is array with hermitian symmetry.
        """
        Data_full = []
        for i in range(data.shape[0]):
            _shape = data[i].shape[1] #shape0 is half 
            slice_full =  jnp.zeros([_shape]*self.ndim,dtype=jnp.float32)
            slice_full = slice_full.at[data[i].shape[0]:].set(data[i][1:])
            #slice_full[data[i].shape[0]:] = data[i][1:]
            obj = jnp.rot90(jnp.rot90((data[i]),k=self.ndim-1,axes=(0,1)),k=self.ndim-1,axes=(int(0+self.is_3D),int(1+self.is_3D))) 
            if self.is_3D==True:
                obj = jnp.flip(obj,axis=1)
            #slice_full[:data[i].shape[0]] = jnp.power(-1,i<self.ndim)*obj
            #slice_full = slice_full.at[:data[i].shape[0]].set(jnp.power(-1,int(i<self.ndim))*obj)
            slice_full = slice_full.at[:data[i].shape[0]].set(jnp.power(-1,int(i<self.ndim))*jnp.conjugate(obj))
            Data_full.append(slice_full)
        return jnp.asarray(Data_full)

    def Sk_jax_custom(self,kmax,custom_pos=None,dkx=None,dky=None,dkz=None,k_vecs=None,batch_size_fac=30):
        """
        Compute the Structure Factor of the point-distribution in 2D or 3D using Hermitian Symmetry. Also CUDA enabled for 3D point-distributions. 
        For optimizations, use gradients of this function.

        Parameters
        ----------
        kmax : float
            Maximum k value in the reciprocal space,
        custom_pos : Ndarray, optional
            If None, uses self.positions,
        dkx : float, optional
            Reciprocal space resolution along x-axis,
        dky : float, optional
            Reciprocal space resolution along y-axis,
        dkz : float, optional
            Reciprocal space resolution along z-axis,
        k_vecs : Ndarray, optional
            Meshgrid of k vectors [KX,KY,KZ] for the tensor product. If None, constructs KX,KY,KZ from dkx,dky,dkz. Note: Overrules kmax and dkx
        """
        custom_pos = custom_pos if custom_pos is not None else self.positions
        if dkx is None:
            dkx = 2 * jnp.pi / self.BoxSize[0]
            dky = 2 * jnp.pi / self.BoxSize[1]
            dkz = 2 * jnp.pi / (self.BoxSize[2]**self.is_3D) 
        x = jnp.array(custom_pos[:, 0], dtype=jnp.float32)  # Changed from jnp.cast
        y = jnp.array(custom_pos[:, 1], dtype=jnp.float32)  # Changed from jnp.cast
        #z = jnp.array(custom_pos[:, 2], dtype=jnp.float32)
        z = jnp.array(custom_pos[:, 2], dtype=jnp.float32) if self.is_3D==True else jnp.zeros(x.shape,dtype=jnp.float32)
        nky = int(round(kmax / dky))#,dtype=jnp.int32)
        nkx = int(round(kmax / dkx))#,dtype=jnp.int32)
        nkz = int(round(self.is_3D*kmax / dkz))#,dtype=jnp.int32)

        kx1d = jnp.arange(0, (nkx + 0.5) * dkx, dkx, dtype=jnp.float32)
        #kx1d_2 = np.arange(-nkx * dkx, -dkx + 0.5 * dkx, dkx, dtype=np.float32)
        #kx1d = np.concatenate([kx1d_2, kx1d_1], axis=0)
        ky1d = jnp.arange(0, (nky + 0.5) * dky, dky, dtype=jnp.float32)
        ky1d_2 = jnp.arange(-nky * dky, -dky + 0.5 * dky, dky, dtype=jnp.float32)
        ky1d = jnp.concatenate([ky1d_2, ky1d], axis=0)
        kz1d = jnp.arange(0, (nkz + 0.5) * dkz, dkz, dtype=jnp.float32)
        kz1d_2 = jnp.arange(-nkz * dkz, -dkz + 0.5 * dkz, dkz, dtype=jnp.float32)
        kz1d = jnp.concatenate([kz1d_2, kz1d], axis=0)

        if (self.is_3D==False):
            kx_grid, ky_grid = jnp.meshgrid(jnp.around(ky1d,4), jnp.around(kx1d,4))#, indexing='ij')
            KZ = jnp.zeros(jnp.ravel(kx_grid).shape[0],dtype=jnp.float32)
            data = jnp.asarray([kx_grid,ky_grid])
        else:
            kx_grid, ky_grid, kz_grid = jnp.meshgrid(jnp.around(ky1d,4),jnp.around(kx1d,4),jnp.around(kz1d,4))
            KZ = jnp.ravel(kz_grid)
            data = jnp.asarray([kx_grid,ky_grid,kz_grid])
        
        if (k_vecs is not None):
            data = jnp.asarray(k_vecs,jnp.float32)
            kx_grid,ky_grid,*KZ = jnp.around(data,4) #k_vecs[:,0],k_vecs[:,1],k_vecs[:,2]
            KZ = jnp.ravel(KZ[0]) if KZ else jnp.zeros(jnp.ravel(kx_grid).shape[0],dtype=jnp.float32)

        KX = jnp.ravel(kx_grid)
        KY = jnp.ravel(ky_grid)
        Sk_chunks = []
        batch_size = int(KX.shape[0]/(batch_size_fac*self.ndim))#,dtype=jnp.int32) #3D needs smaller batch for GPU-memory limit
        for i in range(0, KX.shape[0], batch_size):
            KX_chunk = KX[i:i+batch_size]
            KY_chunk = KY[i:i+batch_size]
            KZ_chunk = KZ[i:i+batch_size]
            # Compute arg = x*KX + y*KY + z*KZ
            arg_chunk = jnp.outer(x, KX_chunk) + jnp.outer(y, KY_chunk) + jnp.outer(z, KZ_chunk)
            cosi = jnp.sum(jnp.cos(arg_chunk), axis=0)
            sinu = jnp.sum(jnp.sin(arg_chunk), axis=0)
            Sk_chunk = (cosi**2 + sinu**2) / custom_pos.shape[0] #self.totalparticles
            Sk_chunks.append(Sk_chunk)
        Sk_flat = jnp.concatenate(Sk_chunks)
        S = Sk_flat.reshape(data[0].shape)
        Sdata = jnp.concatenate((data, S[None,:,:]), axis=0)
        Sdata = self.Hermitiansymmetry(Sdata)
        #print (Sdata.devices())
        return Sdata
    
    def nk_jax_custom(self,kmax,custom_pos=None,dkx=None,dky=None,dkz=None,k_vecs=None):
        """
        Compute the Collective-densities n(k) of the point-distribution in 2D or 3D using Hermitian Symmetry. Also CUDA enabled for 3D point-distributions. 
        For optimizations, use gradients of this function.
        
        Parameters
        ----------
        kmax (float): Maximum k value in the reciprocal space
        custom_pos (Optional | Ndarray): If None, uses self.positions
        dkx (Optional | float): Reciprocal space resolution along x-axis
        dky (Optional | float): "" y-axis
        dkz (Optional | float): "" z-axis
        k_vecs (Optional | Ndarray): Meshgrid of k vectors [KX,KY,KZ] for the tensor product. If None, constructs KX,KY,KZ from dkx,dky,dkz. Note: Overrules kmax and dkx
        """
        custom_pos = custom_pos if custom_pos is not None else self.positions
        if dkx is None:
            dkx = 2 * jnp.pi / self.BoxSize[0]
            dky = 2 * jnp.pi / self.BoxSize[1]
            dkz = 2 * jnp.pi / (self.BoxSize[2]**self.is_3D) 
        x = jnp.array(custom_pos[:, 0], dtype=jnp.float32)  # Changed from jnp.cast
        y = jnp.array(custom_pos[:, 1], dtype=jnp.float32)  # Changed from jnp.cast
        z = jnp.array(custom_pos[:, 2], dtype=jnp.float32)
        nky = int(round(kmax / dky))
        nkx = int(round(kmax / dkx))
        nkz = int(round(self.is_3D*kmax / dkz))

        kx1d = jnp.arange(0, (nkx + 0.5) * dkx, dkx, dtype=jnp.float32)
        #kx1d_2 = np.arange(-nkx * dkx, -dkx + 0.5 * dkx, dkx, dtype=np.float32)
        #kx1d = np.concatenate([kx1d_2, kx1d_1], axis=0)
        ky1d = jnp.arange(0, (nky + 0.5) * dky, dky, dtype=jnp.float32)
        ky1d_2 = jnp.arange(-nky * dky, -dky + 0.5 * dky, dky, dtype=jnp.float32)
        ky1d = jnp.concatenate([ky1d_2, ky1d], axis=0)
        kz1d = jnp.arange(0, (nkz + 0.5) * dkz, dkz, dtype=jnp.float32)
        kz1d_2 = jnp.arange(-nkz * dkz, -dkz + 0.5 * dkz, dkz, dtype=jnp.float32)
        kz1d = jnp.concatenate([kz1d_2, kz1d], axis=0)

        if (self.is_3D==False):
            kx_grid, ky_grid = jnp.meshgrid(jnp.around(ky1d,4), jnp.around(kx1d,4))#, indexing='ij')
            KZ = jnp.zeros(jnp.ravel(kx_grid).shape[0],dtype=jnp.float32)
            data = jnp.asarray([kx_grid,ky_grid])
        else:
            kx_grid, ky_grid, kz_grid = jnp.meshgrid(jnp.around(ky1d,4),jnp.around(kx1d,4),jnp.around(kz1d,4))
            KZ = jnp.ravel(kz_grid)
            data = jnp.asarray([kx_grid,ky_grid,kz_grid])
        
        if (k_vecs is not None):
            data = jnp.asarray(k_vecs,jnp.float32)
            kx_grid,ky_grid,*KZ = np.around(data,4) #k_vecs[:,0],k_vecs[:,1],k_vecs[:,2]
            KZ = jnp.ravel(KZ[0]) if KZ else jnp.zeros(jnp.ravel(kx_grid).shape[0],dtype=jnp.float32)

        KX = jnp.ravel(kx_grid)
        KY = jnp.ravel(ky_grid)
        nk_chunks = []
        batch_size = int(KX.shape[0]/(10*self.ndim)) #3D needs smaller batch for GPU-memory limit
        for i in range(0, KX.shape[0], batch_size):
            KX_chunk = KX[i:i+batch_size]
            KY_chunk = KY[i:i+batch_size]
            KZ_chunk = KZ[i:i+batch_size]
            # Compute arg = x*KX + y*KY + z*KZ
            arg_chunk = jnp.outer(x, KX_chunk) + jnp.outer(y, KY_chunk) + jnp.outer(z, KZ_chunk)
            #cosi = jnp.sum(jnp.cos(arg_chunk), axis=0)
            #sinu = jnp.sum(jnp.sin(arg_chunk), axis=0)
            nk_chunk = jnp.sum(jnp.exp(-1j*arg_chunk),axis=0)
            #nk_chunk = (cosi**2 + sinu**2) / self.totalparticles
            nk_chunks.append(nk_chunk)
        Nk_flat = jnp.concatenate(nk_chunks)
        Nk = Nk_flat.reshape(data[0].shape)
        Ndata = jnp.concatenate((data, Nk[None,:,:]), axis=0)
        Ndata = self.Hermitiansymmetry(Ndata)
        print (Ndata.devices())
        return Ndata

    def Pair_correlation(self,custom_pos=None, r_max=None, dr=None,custom_bins=None):
        """
        Two-point correlation function G_2(r) for the point-distribution in 2D/3D. Uses Nd-histogram and hence non-differentiable. 
        For differentiability, either use Kernel-Density-Estimation or the StructureFactor (FFT) implementation. 
        
        Parameters
        ----------
        r_max : float
            Maximum length to which point-correlations are evaluated. Default: 6 x mean interparticle distance
        dr : float
            Resolution of the point-correlations. Default: r_max/100.

        Returns
        -------
        G2data : numpy.Ndarray
            G2data[0],G2data[1] ... are distances and G2data[-1] is the point-correlation in ndim.
        """
        if r_max is None:
            r_max = 6 * self.dmean
        if dr is None:
            dr = r_max / 100  

        custom_pos = custom_pos if custom_pos is not None else self.positions

        # Calculate histogram bin edges for pair distances
        bins2 = jnp.arange(dr, r_max / 2 + dr, dr)
        _bins2 = jnp.arange(-r_max / 2 - dr, 0, dr)
        bins2 = jnp.concatenate([_bins2, bins2])

        # Calculate pairwise differences
        delta = custom_pos[:, jnp.newaxis, :] - custom_pos[jnp.newaxis, :, :]
        delta = delta.reshape(-1, custom_pos.shape[1])
        
        _bins = custom_bins if custom_bins is not None else bins2
        #Vol = ((jnp.max(_bins) - jnp.min(_bins))/2)**2
        Data = jnp.histogramdd(delta[:, :self.ndim], bins=[_bins]*self.ndim)
        axes = [e[:-1] for e in Data[1]]
        # Calculate two-point correlation function grid
        Grid = jnp.array(jnp.meshgrid(*axes))
        denom = self.totalparticles*self.particledensity * jnp.power(dr,self.ndim)
        G2data = jnp.concatenate((Grid,Data[0][None,:,:]/denom ), axis=0) 
        return G2data

    def pair_corr_Sq_jax(self,Sq=None,kmax=None):
        """
        Pair correlation function in 2D/3D using pre-computed Structure Factor. 
        Since structure factor is computed using point distribution, this implementation is differentiable but S must be computed for q>>0 for fine real-space resolution.
        
        Parameters
        ----------
        Sq : ndarray, optional
            Structure factor to be used for pair correlation evaluation. If None, computed structure factor using self.Sk_jax_custom().
        kmax : float
            Maximum k-vector to be considered for pair-correlation. Since dr=2pi/kmax, larger kmax, better the resolution in pair-correlation (smaller dr). 
    
        Returns
        -------
        G2data : ndarray
            Pair correlation in 2D/3D using structure factor.
        """
        #Sq = self.Sk_jax_custom(kmax=kmax)
        #Sq = self.Sk_jax_custom(kmax=kmax) if jnp.logical_and(kmax is not None,Sq is None) else Sq
        #Sq = jnp.where(Sq is None, self.Sk_jax_custom(kmax=kmax), Sq)
        
        #if not hasattr(self, 'Sq'):
        #    S = self.Sk_jax_custom()
        kmax=jnp.max(Sq[0])
        Gr = jnp.abs(jnp.fft.fftshift(jnp.fft.ifftn(Sq[-1] - 1)))
        # Generate x and y spatial grids
        Shape = Sq[0].shape
        Grid = jnp.array(jnp.meshgrid(*[jnp.linspace(-self.BoxSize[0]/2,self.BoxSize[0]/2,Shape[0])]*self.ndim))
        G2data = jnp.concatenate((Grid,Gr[None,:,:]*kmax ), axis=0) 
        return G2data

    def overlap(self,D0,custom_pos=None):
        """
        Checks for overlapping particles in 2D/3D. 
        
        Parameters
        ----------
        D0 : float
            Distance for sorting. 

        Returns
        -------
        obj : float
            Zero if all particles have a mean-interparticle distance greater than D0. 
        cdist : array
            Distance matrix where any distance greater than D0 is 0.
        """
        custom_pos = custom_pos if custom_pos is not None else self.positions
        delta = custom_pos[:,np.newaxis,:] - custom_pos[np.newaxis,:,:]
        cdist = np.linalg.norm(delta,axis=2)
        _cdist=np.where(cdist<=D0,cdist,0)
        obj = np.sum(_cdist)
        return obj,cdist

    def zoom(self,Box,shape='rect'):
        """
        Queries all the particles inside a bounding region of a given shape. Currently, only circular/spherical and rectangular/cuboidal regions are possible.
        
        Parameters
        ----------
        shape : str
            Shape of the bounding region. Possible options: `rect` or `circ`
        
        Box : list of floats
            If shape == 'rect', allowed input is [[xmin,xmax], [ymin,ymax] [zmin,zmax]] where xmin,xmax, ... etc are the coordinates of the bounding box. If 2D, providing just xy is sufficient.  
            If shape == 'circ', allowed input is [[x0,y0,z0],[R]] where [x0,y0,z0] are center coordinates and R is the radius of the bounding circle. If 2D, providing x0,y0 for center coordinates is sufficient.

        Returns
        -------
        PointDistribution object: The M particles inside the newBoxSize with the same diameter.  
        
        """
        if (shape=='rect'):
            xmin,xmax = Box[0]
            ymin,ymax = Box[1]
            if (self.is_3D==True):
                zmin,zmax = Box[2]
            else:
                zmin,zmax = 0,0
            newBoxSize = [xmax - xmin, ymax - ymin, zmax - zmin]
            mask = np.logical_and(np.logical_and(self.positions[:,0]<xmax,self.positions[:,0]>xmin),
                                np.logical_and(self.positions[:,1]<ymax,self.positions[:,1]>ymin),
                                np.logical_and(self.positions[:,2]<zmax,self.positions[:,2]>zmin) )
            shifts = np.array([newBoxSize[0]/2 - xmax, newBoxSize[1]/2 - ymax, newBoxSize[2]/2 - zmax])
        elif (shape=='circ'):
            x0,y0 = Box[0][0],Box[1][1] 
            if (self.is_3D==True):
                z0 = Box[0][2]
            else:
                z0 = 0
            R = Box[1]
            newBoxSize=[2*R,2*R,self.is_3D*2*R]
            norms=np.linalg.norm(self.positions - np.array([x0,y0,z0]),axis=1)
            mask = np.where(norms<R)
            shifts = - np.array([x0,y0,z0])

        newpositions = self.positions[mask]
        newpositions = newpositions + shifts 
        return PointDistribution(newpositions,diameter=self.diameter,BoxSize=newBoxSize)
    
    def Phaseobject(self,dx,shape=None,diameter=None,mode=None):
        """
        Creates a 2D/3D two-phase media from the point-pattern. Currently only replaces points with sphere of a given radii. 
        
        Parameters
        ----------
        
        dx : float
            Desired resolution for the PhaseDistribution (in microns)

        shape : str, optional
            Shape of the pores. If None, the pores are spheres in 3D and disks in 2D. 
            Other option currently: 'hexagon' (only available in 2D)

        diameter : float, optional
            Diameter of the pores. If None, uses default diameter from PointDistribution.

        mode : str
            If None, keeps particles away from the boundary by padding*radius of particles. 
            If the point-distribution is periodic, the phase distribution can inherit the periodicity by setting 'mode=periodic'.
        
        Returns
        -------
        PhaseDistribution object
        
        """
        res = 1/dx #resolution in pixels / microns
        diameter = self.diameter if diameter is None else diameter
        if (mode=='periodic'):
            #rxy = X.tessellate()
            #Box = X.tesselate_BoxSize()
            rxy_ = PointDistribution(self.tessellate(),diameter=diameter,BoxSize=self.tesselate_BoxSize())
            periodic_mask = 4*diameter
            rxy_ = rxy_.zoom(Box = [
                [-periodic_mask + (-self.BoxSize[0]/2),(self.BoxSize[0]/2)+periodic_mask], 
                [-periodic_mask + (-self.BoxSize[1]/2),(self.BoxSize[1]/2)+periodic_mask], 
                [(-periodic_mask - self.BoxSize[2]/2)*self.is_3D ,(self.BoxSize[2]/2+periodic_mask)*self.is_3D]
                ])
            factor=3
            padding=0
            Box = np.array(rxy_.BoxSize)
            rxy = rxy_.positions
            #print (Box)
        else:
            rxy = self.positions
            factor=1
            #padding=1.4
            padding = 5 #5 pixels
            Box = factor*np.array(self.BoxSize)
        
        rxy = rxy + (1/2)*Box #0.5*np.array(X.BoxSize)
        psi = np.ones(np.int32(res*Box[:self.ndim]  ))
        r0 = np.int32(rxy*res)[:,:self.ndim]
        _mask = r0[(r0<psi.shape).all(axis=1)]
        rad = int(diameter*res/2) #if diameter is None else int(diameter*res/2)
        #_mask = r0[np.all(np.logical_and(r0<(np.array(psi.shape)-padding*rad) , (r0>0+padding*rad)) ,axis=1) ]
        _mask = r0[np.all(np.logical_and(r0<(np.array(psi.shape)-rad - padding) , (r0>0+padding + rad)) ,axis=1) ] #leave a distance of r+padding from the boundaries.
        psi[tuple(_mask.T)] = 0
        #print ()
        if (np.isscalar(self.diameter)==True):
            if (self.is_3D):
                single_object = ball(int(rad))
            else:    
                single_object = disk(int(rad))
            theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
            # polygon vertices around origin
            if (shape=='hexagon'): 
                poly_x = rad * np.cos(theta)
                poly_y = rad * np.sin(theta)
                single_object = np.zeros((2*rad+1, 2*rad+1), dtype=bool)
                rr, cc = polygon(poly_y + rad, poly_x + rad, single_object.shape)
                single_object[rr, cc] = True
            
        psi = asnumpy(binary_erosion(cp.asarray(psi),cp.asarray(single_object),border_value=1))#mask=cp.asarray(mask)) )
        #psi = (binary_erosion((psi),(single_object))) 
        if(mode=='periodic'):
            if (self.is_3D):
                psi = psi[int(periodic_mask*res):-int(periodic_mask*res),
                        int(periodic_mask*res):-int(periodic_mask*res), 
                        int(periodic_mask*res):-int(periodic_mask*res)
                        ]
            else:
                #psi = psi[int(psi.shape[0]/3):int((2/3)*psi.shape[0])+1,int(psi.shape[1]/3):int((2/3)*psi.shape[1])+1]
                psi = psi[
                        int(periodic_mask*res):-int(periodic_mask*res),
                        int(periodic_mask*res):-int(periodic_mask*res)
                        ]
        return PhaseDistribution(psi,dx)
    
    def Phaseobject_voronoi(self,resolution,
                            rad=1, boundary_mask = None,skip_edges=None, output=True):
        """
        Creates a "2D" two-phase media from the point-pattern through Voronoi tessellation. Currently only implemented in 2D and with periodic boundaries.
        It finds the finite vertices from the Voronoi cells and dilates them using given radii to form a porous microstructure.
        If the point distribution is periodic, the resulting phase distribution will also inherit the periodicity.

        Parameters
        ----------
        resolution : float
            Resolution of each pixel in microns.
        
        rad : int, optional
            Radius of the footprint used for dilating the network. If not provided, the result phase distribution is simply a one-pixel network.
            The radius of the footprint can be tuned to obtain a given volume fraction.
        
        boundary_mask : int, optional
            Treatment of the boundary with a mask of M pixels on each edge. 
            If unspecified, assumes no masking. In such a case, phase is periodic only if underlying point-distribution is periodic.

        Returns
        -------
        PhaseDistribution object
        
        """
        def isin_box(v):
            return np.logical_and( (x_min < v[0] < x_max), (y_min < v[1] < y_max) )
        #vor = Voronoi(self.tessellate()[:,:2])
        vor = Voronoi(self.tessellate()[:,:self.ndim])
        padding = int(1.0/resolution)*resolution
        resolution = (padding+self.BoxSize[0]+padding)/((padding+self.BoxSize[0]+padding)//resolution)
        x_min, x_max = - self.BoxSize[0]/2 - padding, self.BoxSize[0]/2 + padding
        y_min, y_max = - self.BoxSize[1]/2 - padding, self.BoxSize[1]/2 + padding
        valid_edges = []
        for vpair in vor.ridge_vertices:
            if -1 in vpair:
                continue  # Skip infinite ridges
            v0, v1 = vor.vertices[vpair[0]], vor.vertices[vpair[1]]
            if isin_box(v0) and isin_box(v1):
                valid_edges.append((v0, v1))
        valid_edges = np.array(valid_edges)
        rand_edges = np.random.randint(0,valid_edges.shape[0],skip_edges) if skip_edges is not None else None
        psi = np.zeros([int((x_max-x_min)/resolution),int((y_max - y_min)/resolution) ])
        #psi = np.zeros(np.int32(np.array(self.BoxSize)[:self.ndim]/resolution  ))
        for i,(v0, v1) in enumerate(valid_edges):
            v0_ = np.int64((v0 + x_max)/resolution)
            v1_ = np.int64((v1 + x_max)/resolution)
                #line(v0_,v0_[1])
            line_mask = line(v0_[0],v0_[1],v1_[0],v1_[1])
            if (np.isin(i,rand_edges)):
                continue
            else:
                psi[line_mask] = 1
        
        #psi = binary_dilation(psi[int(padding/resolution):-int(padding/resolution),
        #                            int(padding/resolution):-int(padding/resolution)],
        #                            footprint=np.ones([rad,rad]))
        psi = binary_dilation(psi,footprint=np.ones([rad,rad]))

        clipped_phase = psi[int(padding/resolution):-int(padding/resolution),
                                    int(padding/resolution):-int(padding/resolution)]
        if (boundary_mask is not None):
            print ('Masking boundaries by {b} microns'.format(b=boundary_mask*resolution))
            clipped_phase[:boundary_mask,:] = 1
            clipped_phase[-boundary_mask:,:] = 1
            clipped_phase[:,:boundary_mask] = 1
            clipped_phase[:,-boundary_mask:] = 1

        Phase = PhaseDistribution(clipped_phase, resolution=self.BoxSize[0]/clipped_phase.shape[0])
        if (output==True):
            print ('Volume fraction of generated phase = {v} with resolution = {r}'.format(v=Phase.volumefraction,r=Phase.resolution))
        return Phase 

    def Phaseobject_trivalent(self, resolution, rad=1, boundary_mask=None,output=True):
        """
        Creates a "2D" trivalent two-phase media from the point-pattern through Centroidal Tessellation. Creates a trivalent network whose volume fraction is decided by `rad`.
        Works with periodic point patterns by tessellating and then cropping to the central box.
        
        Parameters
        ----------
        resolution : float
            Resolution of each pixel in microns.
        rad : int, optional
            Radius of dilation footprint (controls wall thickness).
        boundary_mask : int, optional
            Masking of boundary region.
        """
        # Create Voronoi from tessellated points
        pts = self.tessellate()[:,:2]
        vor = Voronoi(pts)
        padding = int(1.0/resolution)*resolution
        resolution = (padding+self.BoxSize[0]+padding)/((padding+self.BoxSize[0]+padding)//resolution)
        x_min, x_max = - self.BoxSize[0]/2 - padding, self.BoxSize[0]/2 + padding
        y_min, y_max = - self.BoxSize[1]/2 - padding, self.BoxSize[1]/2 + padding
        
        # --- Step 1: Compute centroids of each finite cell ---
        centroids = {}
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0:
                continue  # skip infinite cells
            verts = vor.vertices[region]
            if np.all((verts[:,0] > x_min) & (verts[:,0] < x_max) &
                    (verts[:,1] > y_min) & (verts[:,1] < y_max)):
                centroids[i] = np.mean(verts, axis=0)
        
        # --- Step 2: Connect centroids of neighboring cells ---
        psi = np.zeros([int((x_max-x_min)/resolution),
                        int((y_max-y_min)/resolution)], dtype=bool)
        
        for (p0, p1) in vor.ridge_points:
            if p0 in centroids and p1 in centroids:
                c0, c1 = centroids[p0], centroids[p1]
                c0_pix = np.int64((c0 + np.array([x_max, y_max]))/resolution)
                c1_pix = np.int64((c1 + np.array([x_max, y_max]))/resolution)
                rr, cc = line(c0_pix[0], c0_pix[1], c1_pix[0], c1_pix[1])
                psi[rr, cc] = 1

        # --- Step 3: Dilate walls to desired thickness ---
        psi = binary_dilation(psi, footprint=np.ones([rad, rad]))
        
        # --- Step 4: Clip back to central cell ---
        clipped_phase = psi[int(padding/resolution):-int(padding/resolution),
                            int(padding/resolution):-int(padding/resolution)]
        
        if boundary_mask is not None:
            clipped_phase[:boundary_mask,:] = 1
            clipped_phase[-boundary_mask:,:] = 1
            clipped_phase[:,:boundary_mask] = 1
            clipped_phase[:,-boundary_mask:] = 1
        
        Phase = PhaseDistribution(clipped_phase, resolution=self.BoxSize[0]/clipped_phase.shape[0])
        if (output==True):
            print ('Volume fraction of generated phase = {v} with resolution = {r}'.format(v=Phase.volumefraction,r=Phase.resolution))
        return Phase

    def lloyd_relaxation(self, points=None, n_iter=10,**kwargs):
        """
        Lloyd's algorithm with periodic boundaries.
        Keeps number of points FIXED.
        """
        if points is None:
            pts = self.positions.copy()[:,:2]  
        else:
            pts = points.copy()
        
        for _ in range(n_iter):
            pts_ = PointDistribution(pts,diameter=self.diameter,BoxSize=self.BoxSize)
            mor = pts_.compute_morphology(**kwargs)
            pts = mor.positions
        return pts

    def relax_points(self,points=None,n_iter=10):
        """
        Relax the original point distribution through the Lloyd algorithm.
        """
        pts = self.lloyd_relaxation(points,n_iter)
        print ('Relaxed point distribution by {n} iterations'.format(n=n_iter))
        self.positions = pts
        return None

    def Phaseobject_CVT(self, resolution, rad=1, n_lloyd=10, boundary_mask=None,output=True):
        """
        Creates 2D phase distribution using Centroidal Voronoi Tessellation (CVT).
        Uses Voronoi vertices (trihedral coordination) like in the literature.
        """
        # --- Step 1: Lloyd relaxation ---
        pts = self.tessellate()[:,:2]
        relaxed_pts = self.lloyd_relaxation(pts, n_iter=n_lloyd)

        # --- Step 2: Voronoi from relaxed points ---
        vor = Voronoi(relaxed_pts)
        padding = int(1.0/resolution)*resolution
        resolution = (padding+self.BoxSize[0]+padding)/((padding+self.BoxSize[0]+padding)//resolution)
        x_min, x_max = - self.BoxSize[0]/2 - padding, self.BoxSize[0]/2 + padding
        y_min, y_max = - self.BoxSize[1]/2 - padding, self.BoxSize[1]/2 + padding

        # --- Step 3: Draw Voronoi edges (between vertices) ---
        psi = np.zeros([int((x_max-x_min)/resolution),
                        int((y_max-y_min)/resolution)], dtype=bool)

        for vpair in vor.ridge_vertices:
            if -1 in vpair:
                continue  # skip infinite
            v0, v1 = vor.vertices[vpair[0]], vor.vertices[vpair[1]]
            if (x_min <= v0[0] <= x_max and y_min <= v0[1] <= y_max and
                x_min <= v1[0] <= x_max and y_min <= v1[1] <= y_max):
                p0 = np.int64((v0 - [x_min, y_min]) / resolution)
                p1 = np.int64((v1 - [x_min, y_min]) / resolution)
                rr, cc = line(p0[0], p0[1], p1[0], p1[1])
                psi[rr, cc] = 1

        # --- Step 4: Dilate walls ---
        psi = binary_dilation(psi, footprint=np.ones((rad, rad)))

        # --- Step 5: Crop to central box ---
        clipped_phase = psi[int(padding/resolution):-int(padding/resolution),
                            int(padding/resolution):-int(padding/resolution)]

        if boundary_mask is not None:
            clipped_phase[:boundary_mask,:] = 1
            clipped_phase[-boundary_mask:,:] = 1
            clipped_phase[:,:boundary_mask] = 1
            clipped_phase[:,-boundary_mask:] = 1

        Phase = PhaseDistribution(clipped_phase, resolution=self.BoxSize[0]/clipped_phase.shape[0])
        if (output==True):
            print ('Volume fraction of generated phase = {v} with resolution = {r}'.format(v=Phase.volumefraction,r=Phase.resolution))
        return Phase

    def SpectralDensity(self):
        """
        Spectral Density for the point distribution in 2D/3D when treated as disks/spheres.
        """
        Grid = np.linalg.norm(self.Sq[:-1],axis=0)
        J = 2*np.pi*(self.diameter/2)**2*ss.jv(1,(self.diameter/2)*Grid) /((self.diameter/2)*Grid  )
        J = np.where(Grid==0,2*np.pi*(self.diameter/2)**2,J)
        Xq_j = J**2*self.Sq[-1]*self.totalparticles
        return Xq_j

    def hyperuniformity_chi(self,Q,mode='circle'):
        """
        Chi value for a stealthy HuD pattern for a given Q. It is the number of restricted Q-vectors in the reciprocal space (Q for which Sq=0)

        Parameters
        ----------
        Q : float or list
            Q until which S(Q) = 0. 
        mode : str
            If 'circle', assumes S(Q) = 0 in a circle of radius Q. If not circle, assume rectangle of size 2K[0] x 2K[1]

        Returns
        -------
        chi : float
            Chi value for the HuD pattern. The normalization factor 'k0' is taken as 2pi/self.dmean.
        """
        if (mode!='circle'):
            if (np.isscalar(Q)):
                Qx,Qy = Q,Q
            else:
                Qx,Qy = Q[0],Q[1]
            return (Qx*Qy*self.dmean**2)/(4*np.pi**2)
        else:
            return (self.dmean * Q)**2/(16*np.pi)
        
    def Q_from_chi(self,chi,dmean=None,mode='circle'):
        """
        For a given chi, returns the Q_limit until which S(Q) should be zero.
        
        Parameter
        ---------
        dmean : float
            Mean interparticle distance for which chi needs to be normalized.

        Returns
        -------
        chi : float
            Stealthy Hyperuniformity parameter. 0 - 0.54 is disordered system, greater than that quasi-periodic/periodic.
        """
        Dmean = self.dmean if dmean is None else dmean
        if (mode!='circle'):
            return np.sqrt(chi * 4 * np.pi **2 / Dmean**2 )
        else:
            return np.sqrt(chi*16*np.pi/Dmean**2)

# Move to Hyperuniformity in the future
    def getMk_theory(self,K,L):
        M_square = (((2*K*L)/(2*np.pi))**2 - 1) / 2
        M_circle = np.pi * ((K)/(2*np.pi/L))**2
        return M_circle



