import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
from cupyx.scipy.ndimage import gaussian_filter
from tostada.util.materials import Material
import pickle

try:
    import cupy as cp
    cp.asarray([0])
    print ('GPU detected. Using CUDA')
    def asnumpy(x):
        return cp.asnumpy(x)
except Exception as e:
    print (f"GPU not available : {e}")
    import numpy as cp
    def asnumpy(x):
        return x
    
class LatticeParticleMethod:
    """
    Simulate mechanical response of a `tostada.PhaseDistribution` using prescribed boundary conditions. 
    NOTE : Currently only implemented for 2D systems. 

    Parameters
    ----------
    
    Phase : tostada.PhaseDistribution object
        Binarized phase distribution indicating pores and solid regions. 
    
    Material : tostada.Material object
        Material with a well-defined Young's modulus (N/m^2), Poisson ratio and mass density (kg/m^3).
    
    thickness : float
        Out-of-plane thickness (in meters) of the material. Controls the spring's constant and non-local volume parameter. Default: 1 m.

    scale : float
        Scaling of the lengths from PhaseDistribution (since PhaseDistribution can often be in microns). Default : 1e-6 m.
    
    mode : str
        Two possible modes in 2D. Either `plane-stress` or `plain-strain`.
    """
    #def __init__(self, Phase, Y=6.9e10, v=0.25, rho = 3900, scale=1e-6,mode='plane-stress'):
    def __init__(self, Phase, Material, thickness=1.0,scale=1e-6,mode='plane-stress', is_periodic=False):
        self.scale = scale #converting to meters
        self.Phase = Phase
        self.image = self.Phase.image
        self.thickness = thickness
        self.n_particles = np.size(self.image)#positions.shape[0]
        self.dx = Phase.resolution*self.scale  
        self.Lx = self.Phase.Lx*self.scale
        self.Ly = self.Phase.Ly*self.scale
        self._positions = self.prepare_geometry() #positions
        self.mode = mode
        self.is_periodic = is_periodic
        self.tree = cKDTree(self._positions,boxsize=[self.Lx,self.Ly]) if self.is_periodic else cKDTree(self._positions) 
        self.init_positions = self.prepare_geometry() #init_positions
        self.Y = Material.youngs_modulus  # Young's modulus. default=6.9e10
        self.v = Material.poisson_ratio  # Poisson's ratio. default=0.2
        self.K = self.Y / (3 * (1 - 2 * self.v))  # Bulk Modulus
        self.G = self.Y / (2 * (1 + 2 * self.v))  # Shear Modulus
        self.R = self.dx / 2  # radius of each particle
        self.k1, self.k2, self.T, self.Cxx, self.Cxy, self.Czz = self.get_strain_parameters()

        # Pairlist using KDTree - only done once
        self.pairlist0 = np.asarray([self.tree.query(self.init_positions[i], k=9)[1] for i in range(self.init_positions.shape[0])])
        self.corner_ind, self.bulk_ind, self.edge_ind = self.boundary_indices()
        self.corner_ind_cp = cp.asarray(self.corner_ind)
        self.edge_ind_cp = cp.asarray(self.edge_ind)
        self.bulk_ind_cp = cp.asarray(self.bulk_ind)
        # Convert pairlist and initial distances to CuPy arrays for GPU acceleration
        self.pairlist0_cp = cp.asarray(self.pairlist0)
        self.init_positions_cp = cp.asarray(self.init_positions)
        self.rho = Material.density #rho
        self.mass = self.rho * np.power(self.dx,2) * self.thickness #change later when moving to 3D
        self.vp = np.sqrt((self.K + (4*self.G/3))/self.rho) #maximum velocity of the waves
        self.dtc = self.dx/self.vp #critical time step for the simulation. Keep dt smaller than dtc for numerical stability.
        self.exclusions = np.where(Phase.image.flatten()==0)[0] #excluded regions in the material like pores.
        self.exclusions_cp = cp.asarray(self.exclusions)
        #self.D0 = self.get_initial_dist_new()
        mask_ = ~np.isin(self.pairlist0, self.exclusions)
        self.mask = mask_.astype(float) # pairs where exclusions anywhere in the list are False
        self.mask_cp = cp.asarray(self.mask)
        exclusion_bulk = np.where(np.sum(self.mask[self.exclusions],axis=1)==0)[0]
        self.exclusions_bulk = self.exclusions[exclusion_bulk]
        self.inclusions = np.where(np.isin(np.arange(self.init_positions.shape[0]), self.exclusions), False, True) 
        self.inclusions_cp = cp.asarray(self.inclusions)
        self.right_edge = cp.asarray(np.bool(self.boundary_condition(edge='right',value=True,remove_exclusions=False)))
        self.left_edge = cp.asarray(np.bool(self.boundary_condition(edge='left',value=True,remove_exclusions=False)))
        self.top_edge = cp.asarray(np.bool(self.boundary_condition(edge='top',value=True,remove_exclusions=False)))
        self.bottom_edge = cp.asarray(np.bool(self.boundary_condition(edge='bottom',value=True,remove_exclusions=False)))
        self.D0 = self.get_initial_dist_new()
        
    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, new_positions):
        self._positions = new_positions  # Update the internal positions
        #self.tree = cKDTree(self._positions)  # Rebuild the KDTree

    def get_strain_parameters(self):
        self.k1 = 2*self.Y*self.thickness /(1+self.v)
        self.k2 = self.Y*self.thickness /(1+self.v)
        if (self.mode=='plane-strain'):
            #self.T = self.Y * (4*self.v - 1) / ( (2+np.sqrt(2) )**2 * (1+self.v)*(1 - 2*self.v))
            self.T = self.Y * self.thickness * (4*self.v - 1) / ( (2) * (1+self.v)*(1 - 2*self.v))
            self.Cxx = self.Y * (1 - self.v) / ((1+self.v)*(1 - 2*self.v))
            self.Cxy = self.Y * (self.v) / ((1+self.v)*(1 - 2*self.v))
            self.Czz = 0.5 * self.Y * (1 - 2 * self.v) / ((1+self.v)*(1 - 2*self.v))
        else:
            #self.T = (0.5*self.Y * (3*self.v-1) * (1-self.v) ) / ( (1+self.v) * (1-2*self.v)**2) #works with get_Es. 
            self.T = self.Y * (3*self.v - 1) * self.thickness/ ( (2+np.sqrt(2) )**2 * (1 - self.v**2))
            self.Cxx = self.Y * (1) / ((1 - self.v**2))
            self.Cxy = self.Y * (self.v) / ((1 - self.v**2))
            self.Czz = 1 * self.Y * (1 - self.v) / ((1 - self.v**2))

        return self.k1,self.k2,self.T, self.Cxx, self.Cxy, self.Czz
    
    def save(self, filename):
        """
        Save the all the properties/simulation results in a pickled state.

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
        Load an instance of the saved results created from `.save()`.

        Parameters
        ----------
        filename : str
            Name of the file to be loaded. 
        """
        with open(filename+'.pkl', 'rb') as f:
            A = pickle.load(f)
            print ('File loaded with filename {f}'.format(f=filename+'.pkl'))
        return A 

    def prepare_geometry(self):
        [_y,_x] = np.meshgrid(np.arange(self.image.shape[1]),np.arange(self.image.shape[0])) # 0,1
        positions = np.c_[_x.ravel(),_y.ravel()]*self.dx
        return positions

    def boundary_indices(self):
        ind = self.pairlist0[:,0]
        corner_ind = np.where( (ind == 0) | (ind == self.image.shape[0] - 1) | (ind == self.image.shape[0] * (self.image.shape[1]-1)) | (ind == self.image.size-1))[0]
        edge_ind = np.where( ( (ind < self.image.shape[0] - 1) & (ind > 0) ) | (ind % self.image.shape[0] == 0) | ((ind - (self.image.shape[0]-1)) % self.image.shape[0] == 0) | ( (ind > (self.image.shape[0] * (self.image.shape[1]-1))) & (ind < self.image.size) ))[0]
        bulk_ind = ind[~np.isin(ind,edge_ind)]
        edge_ind = edge_ind[~np.isin(edge_ind, corner_ind)]
        return corner_ind,bulk_ind,edge_ind

    def periodize_boundaries(self,Distance_pairs,Lx=None,Ly=None,only_inclusions=True):
        """
        Periodizes the input distance array based on the BoxSize of the simulation. 
        This essentially corrects the distances of the periodic particles assuming left-right and bottom-top periodic boundary pairs. This is needed since cKDtree does not give corrected lengths.

        Parameters
        ----------

        Distance_pairs : N x 9 x 2 array
            x and y distances of each n particle with respect to its 9 closest neighbors. Note : If `only_inclusions=True`, the distances must not contain any `self.exclusions` (pore regions).

        Lx : float
            Length along x axis that needs to be corrected. If None, uses the x length of the simulation domain.

        Lx : float
            Length along x axis that needs to be corrected. If None, uses the x length of the simulation domain.
        
        only_inclusions : bool
            Decides whether periodization happens on the entire domain or only the `self.inclusions` regions. 

        Returns
        -------

        Distance_pairs_ : N x 9 x 2 array
            Corrected distances at the boundaries (right particles at left boundaries become left ghost particles and etc). Bulk nodes remain unchanged.
            
        """
        Distance_pairs_ = Distance_pairs.copy()
        Lx = self.Lx if Lx is None else Lx
        Ly = self.Ly if Ly is None else Ly

        inclusion_roi = cp.bool_(self.inclusions**only_inclusions)
        right_ind = cp.where(self.right_edge[inclusion_roi] )[0]
        left_ind = cp.where(self.left_edge[inclusion_roi] )[0]
        top_ind = cp.where(self.top_edge[inclusion_roi])[0]
        bottom_ind = cp.where(self.bottom_edge[inclusion_roi])[0]
        
        _maskleft = cp.isin(self.pairlist0_cp[inclusion_roi][left_ind],cp.where(self.right_edge)[0])
        _maskright = cp.isin(self.pairlist0_cp[inclusion_roi][right_ind],cp.where(self.left_edge)[0])
        _masktop = cp.isin(self.pairlist0_cp[inclusion_roi][top_ind],cp.where(self.bottom_edge)[0])
        _maskbottom = cp.isin(self.pairlist0_cp[inclusion_roi][bottom_ind],cp.where(self.top_edge)[0])

        Distance_pairs_[left_ind] = Distance_pairs_[left_ind] + _maskleft[...,None] * cp.array([Lx,0])
        Distance_pairs_[right_ind] = Distance_pairs_[right_ind] - _maskright[...,None] * cp.array([Lx,0])
        Distance_pairs_[top_ind] = Distance_pairs_[top_ind] - _masktop[...,None] * cp.array([0,Ly])
        Distance_pairs_[bottom_ind] = Distance_pairs_[bottom_ind] + _maskbottom[...,None] * cp.array([0,Ly])
        return cp.asarray(Distance_pairs_)

    def get_spring_matrix(self):
        # Using CuPy for GPU acceleration if available
        SpringMatrix = cp.zeros((self.init_positions_cp.shape[0], 9), dtype=cp.float32) #64
        if (self.is_periodic==False):
        # Assign spring constants for corner particles
            SpringMatrix[self.corner_ind_cp, 1:3] = self.k1
            SpringMatrix[self.corner_ind_cp, 3:4] = self.k2

            # Assign spring constants for edge particles
            SpringMatrix[self.edge_ind_cp, 1:4] = self.k1
            SpringMatrix[self.edge_ind_cp, 4:6] = self.k2

            # Assign spring constants for bulk particles
            SpringMatrix[self.bulk_ind_cp, 1:5] = self.k1
            SpringMatrix[self.bulk_ind_cp, 5:9] = self.k2
        else:
            SpringMatrix[:,1:5] = self.k1
            SpringMatrix[:,5:9] = self.k2
        #return SpringMatrix.get()  # Return as NumPy array if needed
        #return asnumpy(SpringMatrix)
        return SpringMatrix
    
    def get_initial_dist_new(self):
        _idx = self.pairlist0_cp
        D0_new = self.init_positions_cp[_idx[:, 0]][:, None] - self.init_positions_cp[_idx]
        if (self.is_periodic):
            D0_new = self.periodize_boundaries(Distance_pairs=D0_new,only_inclusions=False)
        D0_new = cp.linalg.norm(D0_new, axis=2)
        ###corner_ind_cp = cp.asarray(self.corner_ind)
        ###edge_ind_cp = cp.asarray(self.edge_ind)
        if (self.is_periodic==False):
            D0_new[self.corner_ind_cp, 4:] = 0
            D0_new[self.edge_ind_cp, 6:] = 0
        #return D0_new.get()  # Converting back to NumPy if necessary
        #return asnumpy(D0_new)#.get()  
        return D0_new
    
    def Global_Energies(self,velocities):
        #KinE = np.sum(0.5*self.mass*np.linalg.norm(velocities,axis=1)**2)
        KinE = cp.sum(0.5*self.mass*cp.sum(velocities**2,axis=1))
        #return U_strain,KinE
        return KinE
    
    def get_dists(self):
        _idx = self.pairlist0_cp
        positions_cp = cp.asarray(self.positions)
        _Dnew = positions_cp[_idx[:, 0]][:, None] - positions_cp[_idx]
        Dnew = cp.linalg.norm(_Dnew, axis=2)
        unitvec = cp.nan_to_num(_Dnew / Dnew[:, :, None])
        UnitVectorMatrix_x = unitvec[:, :, 0]
        UnitVectorMatrix_y = unitvec[:, :, 1]
        ####corner_ind_cp = cp.asarray(self.corner_ind)
        ####edge_ind_cp = cp.asarray(self.edge_ind)
#        bulk_ind_cp = cp.asarray(self.bulk_ind)
        Dnew[self.corner_ind_cp, 4:] = 0
        Dnew[self.edge_ind_cp, 6:] = 0
        UnitVectorMatrix_x[self.corner_ind_cp, 4:] = 0
        UnitVectorMatrix_y[self.corner_ind_cp, 4:] = 0
        UnitVectorMatrix_x[self.edge_ind_cp, 6:] = 0
        UnitVectorMatrix_y[self.edge_ind_cp, 6:] = 0
        #return Dnew.get(), UnitVectorMatrix_x.get(), UnitVectorMatrix_y.get()
        return asnumpy(Dnew), asnumpy(UnitVectorMatrix_x), asnumpy(UnitVectorMatrix_y)
    
    def get_dists_exc(self,**kwargs):
        """
        Computes distances and unit-vectors only for regions that are not excluded. Should be much faster than `self.get_dists` for high porosities. Damages are identical for mode II.
        """
        _idx = self.pairlist0_cp
        positions_cp = cp.asarray(self.positions)
        #mask = ~cp.isin(_idx[:, 0], self.exclusions)  # Exclude based on the first index in pairs
        mask = self.inclusions_cp
        _idx = _idx[mask]  # Filtered pairlist
        
        _Dnew = positions_cp[_idx[:, 0]][:, None] - positions_cp[_idx]
        
        #__Dnew[_idx] = _Dnew
        if (self.is_periodic):
            L_x = kwargs.get('Lx',None)
            L_y = kwargs.get('Ly',None)
            _Dnew = self.periodize_boundaries(Distance_pairs=_Dnew,Lx=L_x,Ly=L_y,only_inclusions=True)
        __Dnew = cp.linalg.norm(_Dnew, axis=2)
        Dnew = cp.zeros([self.positions.shape[0],9])
        Dnew[_idx[:,0]] = __Dnew
#        Dnew = cp.linalg.norm(__Dnew, axis=2)
        unitvec = cp.nan_to_num(_Dnew / __Dnew[:, :, None])
        #unitvec = cp.nan_to_num(__Dnew / Dnew[:, :, None])
        #print (unitvec.shape)
        _Nx = unitvec[:, :, 0]
        _Ny = unitvec[:, :, 1]
        UnitVectorMatrix_x = cp.zeros([self.positions.shape[0],9])
        UnitVectorMatrix_y = cp.zeros([self.positions.shape[0],9])
        UnitVectorMatrix_x[_idx[:,0]] = _Nx
        UnitVectorMatrix_y[_idx[:,0]] = _Ny
        
        ###corner_ind_cp = cp.asarray(self.corner_ind)
        ###edge_ind_cp = cp.asarray(self.edge_ind)
        
        # Zero out elements that are invalid (keeping only valid pairs)
        if (self.is_periodic==False):
            Dnew[self.corner_ind_cp, 4:] = 0
            Dnew[self.edge_ind_cp, 6:] = 0
            UnitVectorMatrix_x[self.corner_ind_cp, 4:] = 0
            UnitVectorMatrix_y[self.corner_ind_cp, 4:] = 0
            UnitVectorMatrix_x[self.edge_ind_cp, 6:] = 0
            UnitVectorMatrix_y[self.edge_ind_cp, 6:] = 0

        #_Nx = cp.asarray(np.zeros_like(self.D0))
        #_Ny = cp.asarray(np.zeros_like(self.D0))
        #_Nx[_idx[:,0]] = UnitVectorMatrix_x
        #_Ny[_idx[:,0]] = UnitVectorMatrix_y
        #return Dnew.get(), UnitVectorMatrix_x.get(), UnitVectorMatrix_y.get() #_Nx.get(), _Ny.get() 
        #return asnumpy(Dnew), asnumpy(UnitVectorMatrix_x), asnumpy(UnitVectorMatrix_y)
        return Dnew, UnitVectorMatrix_x, UnitVectorMatrix_y

#    def compute_local_properties(self, roi, exclusions):
    def compute_local_properties(self, roi,non_locality=True, **kwargs):
        """
        Computes internal forces in the system for the given region of interest by comparing current positions with positions at t=0. 
        Local displacements leads to strain which in return leads to local stresses. For damage, use `self.compute_local_properties_damage()`
        """
        is_non_local = non_locality
        outside = np.where(roi == False)[0]
        pairs_list = self.pairlist0
        self.pairlist = pairs_list
        #mask = np.where(np.isin(self.pairlist0, self.exclusions), 0, 1)
        #mask = ~np.isin(self.pairlist0, self.exclusions)
        #mask = mask.astype(float)
        ####Dnew, self.UnitVectorMatrix_x, self.UnitVectorMatrix_y = self.get_dists()
        Dnew, self.UnitVectorMatrix_x, self.UnitVectorMatrix_y = self.get_dists_exc(**kwargs)
#        self.StrainMatrix = np.zeros_like(self.D0)
#        self.StrainMatrix[_mask] = Dnew - self.D0[_mask]
        self.StrainMatrix = Dnew - self.D0
        self.SpringMatrix = self.get_spring_matrix()
        self.StrainMatrix = self.StrainMatrix*self.mask_cp
#        self.StrainMatrix = self.StrainMatrix*mask # New. Forces strains to be zero at exclusion sites
        Force_local = self.SpringMatrix * self.StrainMatrix
        sum_contrib_dense = cp.sum(self.StrainMatrix, axis=1)
        sum_contrib_dense[self.exclusions] = 0 #new 
        Force_nonlocal = sum_contrib_dense[self.pairlist0_cp]
        if (self.mode=='plane-strain'):
            ###self.Force = Force_local + 0.5* (2*np.sqrt(2)-3)*self.T * sum_contrib_dense[:, None] * mask #np.ones([self.positions.shape[0], 9])
            Force = Force_local + is_non_local * 0.5* (2*cp.sqrt(2)-3)*self.T * sum_contrib_dense[:, None] * self.mask_cp #np.ones([self.positions.shape[0], 9])
            Force_nonlocal = Force_nonlocal#*mask . not needed
            Force_nonlocal[:,0] = 0 #zeroth column is local force
            Force = Force + is_non_local * 0.5*(2*cp.sqrt(2)-3)*self.T*Force_nonlocal
            ###self.Force = self.Force + 0.5*(2*np.sqrt(2)-3)*self.T*Force_nonlocal
#            self.Force = self.Force + sum_contrib_dense[self.pairlist0]
        else:
            #Es = self.get_Es(cp.asarray(sum_contrib_dense))
            ###self.Force = Force_local + 0.5 * self.T * sum_contrib_dense[:, None] * mask
            Force = Force_local + is_non_local * 0.5 * self.T * sum_contrib_dense[:, None] * self.mask_cp
            #self.Force = Force_local + 0.5*self.T * Es[:, None] * mask#np.ones([self.positions.shape[0], 9])
            Force_nonlocal = Force_nonlocal#*mask
            Force_nonlocal[:,0] = 0
            ###self.Force = self.Force + 0.5*self.T*Force_nonlocal
            Force = Force + is_non_local * 0.5*self.T*Force_nonlocal
            #self.Force = Force_local + self.T * sum_contrib_dense[:, None] * np.ones([self.positions.shape[0], 9])

        ###self.Force[outside] = 0
        #Force[outside] = 0
        ###self.Force = self.Force * mask #. Not needed
        Force = Force * self.mask_cp
        #Force_x = self.Force * self.UnitVectorMatrix_x
        #Force_y = self.Force * self.UnitVectorMatrix_y
        Force_x = Force * self.UnitVectorMatrix_x
        Force_y = Force * self.UnitVectorMatrix_y

        Force_i = cp.column_stack((cp.sum(Force_x, axis=1), cp.sum(Force_y, axis=1)))
        #return Force_i, Force
        return Force_i,Force
    
    def compute_local_properties_damage(self, roi, damage_array):
        """
        Computes internal forces in the system for the given region of interest by comparing current positions with positions at t=0. 
        Local displacements leads to strain which in return leads to local stresses. Damage is accounted for in the computed of forces.
        """
        #####outside = np.where(roi == False)[0] 
        pairs_list = self.pairlist0
        self.pairlist = pairs_list
        #mask = np.where(np.isin(self.pairlist0, self.exclusions), 0, 1)
        #mask = ~np.isin(self.pairlist0, self.exclusions)
        #mask = mask.astype(float)
        #Dnew, self.UnitVectorMatrix_x, self.UnitVectorMatrix_y = self.get_dists()
        Dnew, self.UnitVectorMatrix_x, self.UnitVectorMatrix_y = self.get_dists_exc()
    #        self.StrainMatrix = np.zeros_like(self.D0)
    #        self.StrainMatrix[_mask] = Dnew - self.D0[_mask]
        self.StrainMatrix = Dnew - self.D0
        self.SpringMatrix = self.get_spring_matrix()
        self.StrainMatrix = self.StrainMatrix*self.mask_cp # New. Forces strains to be zero at exclusion sites
        Force_local = self.SpringMatrix * self.StrainMatrix
        sum_contrib_dense = cp.sum(self.StrainMatrix, axis=1)
        sum_contrib_dense[self.exclusions] = 0 #new 

        Force_nonlocal = sum_contrib_dense[self.pairlist0_cp]
        
        dij = damage_array[self.pairlist0_cp]*1
        dij_avg = 1*0.5 * (damage_array[:,None]*cp.ones_like(self.pairlist0_cp) ) +1* 0.5 * dij

        if (self.mode=='plane-strain'):
            ###self.Force = Force_local*(1-dij_avg) + 0.5* (2*np.sqrt(2)-3)*self.T * sum_contrib_dense[:, None] * mask * (1 - 1*damage_array[:,None])  #np.ones([self.positions.shape[0], 9])
            Force = Force_local*(1-dij_avg) + 0.5* (2*cp.sqrt(2)-3)*self.T * sum_contrib_dense[:, None] * self.mask_cp * (1 - 1*damage_array[:,None])  #np.ones([self.positions.shape[0], 9])
            Force_nonlocal = Force_nonlocal#*mask . not needed
            Force_nonlocal[:,0] = 0 #zeroth column is local force
            dij_ = dij.copy()
            dij_[:,0] = 0
            ###self.Force = self.Force + 0.5*(2*np.sqrt(2)-3)*self.T*Force_nonlocal * (1 - dij_)
            Force = Force + 0.5*(2*cp.sqrt(2)-3)*self.T*Force_nonlocal * (1 - dij_)
    #            self.Force = self.Force + sum_contrib_dense[self.pairlist0]
        else:
            #Es = self.get_Es(cp.asarray(sum_contrib_dense))
            ###self.Force = Force_local*(1-dij_avg) + 0.5 * self.T * sum_contrib_dense[:, None] * mask * (1 - damage_array[:,None])
            Force = Force_local*(1-dij_avg) + 0.5 * self.T * sum_contrib_dense[:, None] * self.mask_cp * (1 - damage_array[:,None])
            #self.Force = Force_local + 0.5*self.T * Es[:, None] * mask#np.ones([self.positions.shape[0], 9])
            Force_nonlocal = Force_nonlocal#*mask
            Force_nonlocal[:,0] = 0
            dij_ = dij.copy()
            dij_[:,0] = 0
            ###self.Force = self.Force + 0.5*self.T*Force_nonlocal * (1 - dij_)
            Force = Force + 0.5*self.T*Force_nonlocal * (1 - dij_)
            #self.Force = Force_local + self.T * sum_contrib_dense[:, None] * np.ones([self.positions.shape[0], 9])

        ###self.Force[outside] = 0
        #####Force[outside] = 0
        ###self.Force = self.Force * mask #. Not needed
        Force = Force * self.mask_cp
        ###Force_x = self.Force * self.UnitVectorMatrix_x
        ###Force_y = self.Force * self.UnitVectorMatrix_y
        Force_x = Force * self.UnitVectorMatrix_x
        Force_y = Force * self.UnitVectorMatrix_y
        Force_i = cp.column_stack((cp.sum(Force_x, axis=1), cp.sum(Force_y, axis=1)))
        return Force_i, Force
    

    def get_stresses(self,Force_pairs):
        """
        Computes the local stress tensor components. Directly comparable with COMSOL.

        Parameters
        ----------        
        Force_pairs : (N x 6) cupy array
            Force on particle i due to its j neighbours. This is accesible after running `self.compute_local_properties()` or `self.compute_local_properties_damage()`.
        
        Returns
        -------
        Stress tensor components. Since the code is 2D, currently, only gives sigma_xx,sigma_xy and sigma_yy.
        """
        #LF = (1/(2*self.dx*self.dx))*sp.csr_matrix(self.D0).multiply(self.Force)
        ####LF = (1/(2*self.dx*self.dx))*self.D0*self.Force
        LF = (1/(2*self.dx*self.dx))*self.D0*Force_pairs
        sigmaxx = cp.sum(LF*self.UnitVectorMatrix_x*self.UnitVectorMatrix_x,axis=1)
        sigmaxy = cp.sum(LF*self.UnitVectorMatrix_x*self.UnitVectorMatrix_y,axis=1)
        sigmayy = cp.sum(LF*self.UnitVectorMatrix_y*self.UnitVectorMatrix_y,axis=1)
        von_mises = cp.sqrt( np.power(sigmaxx,2) + np.power(sigmayy,2) + 3*np.power(sigmaxy,2) - (sigmaxx*sigmayy) )

        sigma1=(sigmaxx+sigmayy)/2 + cp.sqrt( cp.power((sigmaxx-sigmayy),2)/4 + cp.power(sigmaxy,2) )
        sigma2=(sigmaxx+sigmayy)/2 - cp.sqrt( cp.power((sigmaxx-sigmayy),2)/4 + cp.power(sigmaxy,2) )
        return sigmaxx,sigmaxy,sigmayy,von_mises,sigma1,sigma2
        #return sigmaxx*self.dx,sigmaxy*self.dx,sigmayy*self.dx,von_mises*self.dx,sigma1*self.dx,sigma2*self.dx
    
    def reshape_fields(self,quantity):
        """
        Quick function for reshaping M local fields of size [(N_x N_y) x 1] into [N_x x N_y]. 
        
        Returns
        -------
        Reshape_fields (numpy.Ndarray or list of Ndarray): List of reshaped arrays. If single quantity, returns a single reshaped object. 
        """
        if type(quantity) == list:
            Reshape_fields = []
            for i in range(len(quantity)):
                Reshape_fields.append(np.reshape(quantity[i],self.image.shape))
        else:
            Reshape_fields = np.reshape(quantity,self.image.shape)
        return Reshape_fields

    def get_displacements(self):
        """
        Displacement fields u,v. Note: displacements are scaled by a factor of 2 to account for nodes vs edges discrepancy. 
        
        Returns
        -------
        [U,V] (list): List of displacements along each axes. Size [(Nx x Ny) x 1]. Reshape them using self.reshape_fields() 
        """
        Displacements = 2*(self.positions - self.init_positions)
        U = Displacements[:,0]
        V = Displacements[:,1]
        return [U,V]
    
    def total_displacements(self):
        """
        Total change of length along each axes. Directly comparable to COMSOL. Note: displacements are scaled by a factor of 2 to account for nodes vs edges discrepancy."
        
        Returns
        -------
        dLx (float): Change of length along x axis.
        dLy (float): Change of length along y axis.
        L_x + dLx (float): New length along x axis.
        L_y + dLy (float): New length along y axis.
        """
        dLx = 2*((np.max(self.positions[:,0]) - np.max(self.init_positions[:,0])))
        dLy = 2*((np.max(self.positions[:,0]) - np.max(self.init_positions[:,1])))
        return dLx,dLy,dLx+np.max(self.init_positions[:,0]),dLy+np.max(self.init_positions[:,1])
    
    def get_strains(self,sigma=None,mode=None):#0.6):
        """
        Computes the Green-Lagrange components of the strain tensor from the displacements. Directly comparable with COMSOL.
        
        Parameters
        ----------

        sigma : float
            Smoothening parameter for the derivate at the pore/solid interface. If None, computes sigma based on the resolution.

        mode : str
            Mode for applying gaussian filter. Options are `nearest`, `wrap` (for periodic), 'reflect', 'constant', 'mirror'.

        Returns
        -------
        Strain tensor components. Since the code is 2D, currently, only gives Eps_xx,Eps_xy and Eps_yy.
        """
        mode = 'nearest' if mode is None else mode
        sigma_gradient = 1.3*((self.Lx/100) / (2*(cp.sqrt(2*cp.log(2)))*self.dx) ) if sigma is None else sigma
        self.sigma_gradient = sigma_gradient
        #print ('selected sigma = {s}'.format(s=self.sigma_gradient))
        mask = cp.asarray(self.image)
        U,V = self.reshape_fields(self.get_displacements())
        den = gaussian_filter(mask.astype(float), sigma=sigma_gradient, mode=mode)#'nearest')
        U_masked = cp.where(mask,cp.asarray(U),0)
        V_masked = cp.where(mask,cp.asarray(V),0)
        numU = gaussian_filter(U_masked,sigma=sigma_gradient,mode=mode)#'nearest')
        numV = gaussian_filter(V_masked,sigma=sigma_gradient,mode=mode)#'nearest')
        U_s = cp.where(den>1e-20,numU/(den+1e-20), 0.0)
        V_s = cp.where(den>1e-20,numV/(den+1e-20), 0.0)
        #U_s = masked_gaussian(cp.asarray(U), mask, sigma=sigma_gradient)
        #V_s = masked_gaussian(cp.asarray(V), mask, sigma=sigma_gradient)
        dU_dy, dU_dx = cp.gradient(U_s.T, self.dx, self.dx)
        dV_dy, dV_dx = cp.gradient(V_s.T, self.dx, self.dx) 
        Eps_xx = 0.5*(2*dU_dx + 0*dU_dx**2 + 0*dV_dx**2)
        Eps_yy = 0.5*(2*dV_dy + 0*dU_dy**2 + 0*dV_dy**2)
        Eps_xy = 0.5*(dU_dy + dV_dx + 0*dU_dx*dU_dy + 0*dV_dx*dV_dy)
        return Eps_xx,Eps_xy,Eps_yy

    def strain_energy(self):
        StrainEnergy=cp.sum(self.SpringMatrix*self.StrainMatrix**2,axis=1) + 0.5 * self.T * cp.sum(self.StrainMatrix,axis=1)*cp.sum(self.StrainMatrix,axis=1)
        StrainEnergy[self.exclusions] = 0
        return StrainEnergy

    def get_Es(self,sumdist):
        num1 = (2*self.R)*(cp.sqrt(2)-1)*(2*self.v-1)*sumdist
        denom1 = 2*(1-self.v)*self.R - (cp.sqrt(2)-1)*self.v*sumdist

        num2 = (4*self.R**2)*(cp.sqrt(2)-1)*(2*self.v-1)*(1-self.v)
        denom2 = denom1**2

        Es = (num1*num2)/(denom1*denom2)
        #return Es.get()
        return asnumpy(Es)
    
    def run_sim(self,roi,Force,is_velocity=False,
                dt=None,num_steps=3000,tolerance=1.5e-3,
                affine_matrix=None, damping_factor=0, at_every=200, **kwargs):
        """
        Runs quasi-static LPM simulation using applied forces at interfaces. For damage simulation, use 'run_sim_damage'.

        Parameters
        ----------

        roi : [N x 2] boolean array
            Region of interest indicating fixed points in the domain. If False, the particle does not get displaced (vx=vy=0).

        Force : [N x 2] float array
            Applied forces or velocities for each particle. Initialized the external conditions. If Force, units are in Newton and if velocity, m/s.
        
        is_velocity : bool
            Decides whether `Force` is a prescribed load or a prescribed velocity condition. 
            If True, `Force` is treated as prescribed velocity for given nodes at each time step and sets external forces as 0.
            If False, `Force` is treated as prescribed load for given nodes and sets initial velocity as `dt`*`Force`/`mass`.

        dt : float
            Time resolution for the evolution of simulation. Lower the value, more time-stable the results. Default : dt_c (critical time-stepping obtained from material properties).

        num_steps : int
            Total number of time steps to run the simulation.

        tolerance : float
            Tolerance for terminating the simulation. This is the ratio of maximum Kinetic energy attained till time t vs Kinetic energy at time t.

        affine_matrix : 2x2 numpy array
            Affine transformation applid to the entire macroscopic domain. Useful for macroscopic strains in homogenization problems. 

        at_every : int
            Frequency of recording the time-fields. NOTE: Smaller the value, more data is cached at each time-step which may slow down the computations.  
        """
        boxsize = cp.array([self.Lx,self.Ly])
        particle_roi = roi
        outside = cp.asarray(np.where(particle_roi == False)[0])
        dt=0.6*self.dtc if dt is None else dt 
        self.dt = dt #save it for further post-process
        self.positions = cp.asarray(self.positions)
        self.init_positions = self.init_positions_cp
        Force_ext = cp.asarray(Force) if is_velocity==False else cp.zeros(self.positions.shape) #np.zeros(positions.shape)
        velocities = (dt/self.mass) * Force_ext if is_velocity==False else cp.asarray(Force)
        Force_hist = []
        pos_hist = []
        vel_hist = []
        Force_hist = []
        sigma_hist=[]
        Sigma_rnk = []
        Ustrain,KinE = np.zeros(num_steps),np.zeros(num_steps)
        iter_array = [] #to record the steps where quantities are recorded
        KinE0 = self.Global_Energies(velocities)
        affine_matrix = cp.eye(2) if affine_matrix is None else cp.asarray(affine_matrix)
        self.positions = ((affine_matrix)@ self.positions.T).T
        boxsize = affine_matrix @ boxsize
        for i in range(1, num_steps):        
            #self.positions = ((affine_matrix)@ self.positions.T).T
            #boxsize = affine_matrix @ boxsize
            r0 = self.positions
            v0 = velocities - self.is_periodic*cp.mean(velocities,axis=0)
            KinE0=self.Global_Energies(v0) # here for prescribed velocity
            if (is_velocity==True):
                v0[Force!=0] = v0[Force!=0] + Force[Force!=0] ################## NEW for force vs velocity input. To be tested
            v0[outside] = 0 # particles that are outside roi are fixed (vx=vy=0)
            #KinE0=self.Global_Energies(v0) # here for prescribed load
            half_pos = r0 + (dt/2)*v0

            self.positions = half_pos

            Force_i, Forces = self.compute_local_properties(particle_roi,Lx=float(boxsize[0]),Ly=float(boxsize[1]))
            
            velocities = v0 + (dt)*(-damping_factor*v0  + (Force_ext - Force_i)/self.mass)
            self.positions = r0 + (dt/2)*(v0 + velocities)/2
            #Ustrain[i],KinE[i] = self.Global_Energies(velocities)
            KinE[i] = self.Global_Energies(velocities)
            Ustrain[i] = cp.sum(self.strain_energy())
            sigma_rnk = self.stress_truncated_rankine()
            sigma_rnk = sigma_rnk.T.flatten()
            #sigma_rnk[outside]=0
            sigma_rnk[self.exclusions] = 0
            #if (KinE[i]<=self.Global_Energies(v0)[1]):
            if (KinE[i]<=KinE0):
                velocities = v0*0 #cease the motion
                print ('current KE=',KinE[i])
                print ('Ratio=',KinE[i]/cp.max(KinE))
                print ('velocities set to zero at t=',i)
                if (KinE[i]/cp.max(KinE)<tolerance):
                    print ('Simulation converged')
                    Force_hist.append(Force_i)
                    pos_hist.append(self.positions)
                    #vel_hist.append(velocities)
                    stresses = self.get_stresses(Forces)
                    sigma_hist.append(stresses)
                    iter_array.append(i)
                    self.iter_array = np.asarray(iter_array)
                    self.sigma_history = cp.asarray(sigma_hist) #stresses recorded 
                    self.position_history = cp.asarray(pos_hist) #positions recorded
                    self.Force = Force_i
                    self.KE_history = KinE
                    self.U_history = Ustrain
                    self.Force_pairs = Forces
                    self.Sigma_rnk = cp.asarray(Sigma_rnk)
                    self.Force_hist = Force_hist
                    break
            
            if (np.mod(i,at_every)==0):
                Force_hist.append(Force_i)
                pos_hist.append(self.positions)
                #vel_hist.append(velocities)
                stresses = self.get_stresses(Forces)
                sigma_hist.append(stresses)
                iter_array.append(i)
                Sigma_rnk.append(sigma_rnk)
                print ('current time step={n}, boxsize=[{lx},{ly}]'.format(n=i,lx=boxsize[0],ly=boxsize[1]))

            #KinE0 = KinE[i] #save current step's energy for next time step
            self.iter_array = np.asarray(iter_array)
            self.sigma_history = cp.asarray(sigma_hist) #stresses recorded 
            self.position_history = cp.asarray(pos_hist) #positions recorded
            self.Sigma_rnk = cp.asarray(Sigma_rnk)

        self.Force_hist = Force_hist
        self.Force = Force_i
        self.KE_history = KinE
        self.U_history = Ustrain
        self.Force_pairs = Forces
        return None

    def run_sim_damage(self,roi,force_roi,applied_velocity,
                    tensile_strength,degradation_rate,
                    dt=None,num_steps=3000,
                    at_every=20):
        """
        Runs dynamic LPM simulation using applied velocities at interfaces. 

        applied_velocity : float of 1x2 array
            Applied velocity at the regions specified by force_roi. If float, makes vx=vy=v0 
        
        tensile_strength : float
            Maximum load the material can bear before failure
        """        
        num_steps = int((num_steps/at_every)*at_every)
        particle_roi = roi
        outside = np.where(particle_roi == False)[0]
        num_particles = self.positions.shape[0]
        applied_velocity = np.array(applied_velocity)
        dt=0.6*self.dtc if dt is None else dt 
        self.dt = dt #save it for further post-process
        #Force_ext = Force #np.zeros(positions.shape)
        Force_ext = np.zeros(self.positions.shape) # zero because external force applied here using applied velocities
        #velocities = (dt/self.mass) * Force_ext
        velocities = np.zeros(self.positions.shape) 
        velocities = applied_velocity * force_roi 

        num_T = int((num_steps - at_every)/at_every)+1 # number of time slices to be saved
        
        iter_array = np.zeros(num_T)
        sigma_history = np.zeros([num_T,6,num_particles])
        position_history = np.zeros([num_T,num_particles,2])
        Damage_history = np.zeros([num_T,num_particles])
        Sigma_rnk_history = np.zeros([num_T,num_particles])
        Force_history = np.zeros([num_T,num_particles,2])
        self.init_exclusions = self.exclusions
        max_stresses = cp.ones(num_particles) * tensile_strength #r value for each particle. Initial value: sigma_t (tensile strength)
        #Edensity = []
        KinE = []
        #iter_array = [] #to record the steps where quantities are recorded
        KinE0 = self.Global_Energies(velocities)
        damage_i = cp.zeros(num_particles)
        ind_ = -1
        for i in range(1, num_steps+1):        
            damaged_particles = cp.where(damage_i>0.99)[0]#,True,False)
            self.exclusions = np.append(self.exclusions,damaged_particles)
            r0 = self.positions
            v0 = velocities
            v0[force_roi] = applied_velocity if np.isscalar(applied_velocity) else applied_velocity[force_roi] #* force_roi
            v0[outside] = 0 # particles that are outside roi are fixed (vx=vy=0)
            half_pos = r0 + (dt/2)*v0

            self.positions = half_pos

            Force_i, Forces = self.compute_local_properties_damage(particle_roi,damage_i)
            velocities = v0 + (dt)*(Force_ext - Force_i)/self.mass
            self.positions = r0 + (dt/2)*(v0 + velocities)/2
            KinE.append(self.Global_Energies(velocities))
            sigma_rnk = self.stress_truncated_rankine()
            sigma_rnk = sigma_rnk.T.flatten()
            #sigma_rnk[outside]=0
            sigma_rnk[self.exclusions] = 0
            damage_i, sigma_max_i = self.damage_evolution(sigma_rnk, max_stresses, tensile_strength, degradation_rate)
            max_stresses = cp.max(cp.stack([sigma_max_i,max_stresses]),axis=0)
            if np.logical_and((np.mod(i,at_every)==0),(damage_i > 0.99).any()):
                print("Damage initiated at time={n}".format(n=i))
            if (np.mod(i,at_every)==0):
                #Force_hist.append(Force_i)
                #Force_hist.append(Force_i)
                
                ###pos_hist.append(self.positions)
                ind_ = ind_ + 1
                position_history[int(ind_)] = self.positions
                
                stresses = cp.asarray(self.get_stresses(Forces))

                ###sigma_hist.append(stresses)
                sigma_history[int(ind_)] = asnumpy(stresses)

                ###iter_array.append(i)

                iter_array[int(ind_)] = i
                ###Damage.append(damage_i)
                Damage_history[int(ind_)] = asnumpy(damage_i)
                ###sigma_rnk_hist.append(sigma_rnk)
                Sigma_rnk_history[int(ind_)] = asnumpy(sigma_rnk)
                Force_history[int(ind_)] = asnumpy(Force_i)
                #Edensity.append(self.EnergyDensity(Forces))
                print ('current time step={n}'.format(n=i))

        self.iter_array = iter_array
        self.sigma_history = sigma_history
        self.position_history = position_history
        self.Damage_history = Damage_history
        self.Sigma_rnk_history = Sigma_rnk_history
        self.Force_history = Force_history
        self.KE_history = np.asarray(KinE)
        return None

    def boundary_condition(self,edge,value=None,remove_exclusions=True):
        """
        Apply boundary condition for a variable of shape Nx1 where N is the number of pixels. 

        Parameters
        ----------

        edge : str
            Which edge to apply the condition. Options currently: 'left','right','top','bottom'
         
        value : float
            Value to be applied to the given component at the given edge.
        """
        Arr = np.full(self.positions.shape[0],False)
        if (edge=='right'):
            mask = np.where((self.positions[:,0]==np.max(self.positions[:,0])))[0]
        elif (edge=='left'):
            mask = np.where((self.positions[:,0]==0 ) )[0]
        elif (edge=='top'):
            #mask = np.where( np.logical_and( (pos[:,1]==0), (pos[:,0]==np.max(pos[:,0]))  ) )[0]
            mask = np.where(self.positions[:,1]==np.max(self.positions[:,1]))[0]
        elif (edge=='bottom'):
            mask = np.where(self.positions[:,1]==0)[0]
        Arr[mask] = True
        if (remove_exclusions==True):
            Arr[self.exclusions] = False #Remove spuriously included regions, if any
        if (value is not None):
            Component = np.zeros(self.positions.shape[0])
            Component[Arr] = value
        else:
            Component = Arr
        return Component
    
    def surf_integration(self,quantity):
        """
        Integrates a local quantity (like stress/strain components or etc) over the solid phase region (excludes the pores).
        """
        inclusions = np.where(np.isin(np.arange(self.positions.shape[0]), self.exclusions), False, True)
        _surf_int = np.trapz(quantity[inclusions],dx=self.dx**2)
        return _surf_int
    
    def EnergyDensity(self, Force_pairs):
        """
        Energy Density from sigma and eps tensor. Essentially does a Frobenius norm: sigma_ij epsilon_ij.

        Parameters
        ----------

        Force_pairs : (N x 6) array
            Force on particle i due to its j neighbours. This is accesible as `self.Force_pairs` after running the quasi-static `self.run_sim()`.
        """
        Stresses = self.get_stresses(Force_pairs)
        Strains = self.get_strains()
        return 0.5*(Stresses[0]*Strains[0].T.flatten() + Stresses[1]*Strains[1].T.flatten() + Stresses[2]*Strains[2].T.flatten())

    def effective_quantities(self, Force_pairs):
        """
        Effective global quantities. If the initial `PhaseDistribution` obeys periodicity, use `Homogenize()` instead.

        Parameters
        ----------

        Force_pairs : (N x 6) array
            Force on particle i due to its j neighbours. This is accesible as `self.Force_pairs` after running the quasi-static `self.run_sim()` or after `self.compute_local_properties()`.

        Returns
        -------
        stress_conc : float
            Global stress concentration in the system. Can be used to qualitatively predict potential failures. Higher the value, more localized stresses.

        poisson_eff : float
            Effective poisson ratio of the system

        youngs_eff_xx : float
            Effective young's modulus component xx
        
        youngs_eff_xy : float
            Effective young's modulus component xy
        
        youngs_eff_yy : float
            Effective young's modulus component yy

        """
        eps_xx,eps_xy,eps_yy = self.get_strains()
        sigma_xx,sigma_xy,sigma_yy,sigma_vm,sigma_p1,sigma_p2 = self.get_stresses(Force_pairs)
        youngs_eff_xx = np.mean(sigma_xx)/np.mean(eps_xx)
        poisson_eff = -np.mean(eps_yy.flatten())/np.mean(eps_xx.flatten())
        youngs_eff_xy = np.mean(sigma_xy)/np.mean(eps_xy)
        youngs_eff_yy = np.mean(sigma_yy)/np.mean(eps_yy)
        stress_conc = np.max(sigma_vm)/np.mean(sigma_vm[self.inclusions])
        return stress_conc,poisson_eff,youngs_eff_xx,youngs_eff_xy,youngs_eff_yy
    
    def stress_from_constitutive_relations(self):
        """
        Stress components obtained from the constitutive relations sigma = C eps 
        Uses strain components evaluated from `self.get_strains()`. Currently only implemented for 2D.
        Some papers call this as effective stresses. Not to be confused with the homogenized effective stress tensor components obtained using `self.effective_quantities`.
        """
        eps_xx,eps_xy,eps_yy = self.get_strains()
        sigma_eff_xx = self.Cxx * eps_xx + self.Cxy * eps_yy
        sigma_eff_yy = self.Cxy * eps_xx + self.Cxx * eps_yy
        sigma_eff_xy = self.Czz * eps_xy
        return sigma_eff_xx,sigma_eff_xy,sigma_eff_yy
    
    def stress_truncated_rankine(self):
        """
        Get stresses in the system according to truncated Rankine model (https://doi.org/10.1016/j.engfracmech.2024.110203).
        Uses principal values of stresses obtained from the constitutive relations and computes their Macaulay sum.

        Returns
        -------
        sigma_rankine_t : [Nx x Ny] array
            Sigma rankine for each particle. Since `self.get_strains()` reshapes the data, the sigma here is [Nx x Ny] as opposed to usual [(N_x N_y) x 1].
            To put it back in same ordering (to remove exclusions), use `.T.flatten()` 
        """
        
        sigma_eff_xx, sigma_eff_xy, sigma_eff_yy = self.stress_from_constitutive_relations()
        sigma1_eff=(sigma_eff_xx+sigma_eff_yy)/2 + cp.sqrt( cp.power((sigma_eff_xx-sigma_eff_yy),2)/4 + cp.power(sigma_eff_xy,2) )
        sigma2_eff=(sigma_eff_xx+sigma_eff_yy)/2 - cp.sqrt( cp.power((sigma_eff_xx-sigma_eff_yy),2)/4 + cp.power(sigma_eff_xy,2) )
        #sigma_rankine_t = 0.5 *(sigma1_eff + np.abs(sigma1_eff)) + 0.5 * (sigma2_eff + np.abs(sigma2_eff))
        sigma_rankine_t = cp.sqrt((cp.where(sigma1_eff>0,sigma1_eff,0))**2 + (cp.where(sigma2_eff > 0, sigma2_eff, 0))**2)
        #sigma_rankine_t = self.mac_func(sigma1_eff) + self.mac_func(sigma2_eff)
        return sigma_rankine_t
        #sigma_xx,sigma_xy,sigma_yy,sigma_vm,sigma_p1,sigma_p2 = self.get_stresses()
        #sigma_ = self.mac_func(sigma_p1) + self.mac_func(sigma_p2)
        #return sigma_
    
    def damage_evolution(self,current_stresses, max_stresses, tensile_strength, degradation_rate):
        """
        Evolution of internal damage variable of each particle according to the exponential law.

        Parameters
        ----------

        current_stresses : [(N_x N_y) x 1] array
            Local stresses for the current time-step. To compute these, use `self.stress_truncated_rankine()`.
        
        max_stresses : [(N_x N_y) x 1] array
            Array of maximum stresses for each particle from any time : [0, t - 1]. 
        
        tensile_strength : float
            Tensile strength of the material. Calibrate with experiments.
        
        degradation_rate : float
            Rate of degradation of the material. Related to energy dissipated in uniaxial tensile strength. 

        Returns
        -------

        d_array : [(N_x N_y) x 1] array
            Damage at each particle for a given time-step.

        sigma_max : [(N_x N_y) x 1] array
            Updated maximum stresses for each particle from any time : [0, t]
        """
        sigma_max = cp.max(np.stack([current_stresses,max_stresses]),axis=0)
        stress_threshold_array = cp.where(sigma_max < tensile_strength, tensile_strength, sigma_max)
        d_array = 1 - ((tensile_strength / stress_threshold_array) *  cp.exp(degradation_rate * (1 - (stress_threshold_array / tensile_strength))))
        return d_array, sigma_max

    def mac_func(self,arr):
        return 0.5*arr + 0.5*np.abs(arr)

    def animate_fields(self, X, filename='animated_field',fps=10, dpi=150,vmin=None,vmax=None,**kwargs):
        """
        Animate M time-slices of a given field (stress or strain or damage).

        Parameters
        ----------

        X : M x N x 1 array
            Time (and space) series data to be plotted. First, the data is reshaped to X(t,x,y) and then animated along the time axis.

        fps : int
            Frame rate of the animation. Default : 10

        dpi : int
            Resolution of the animation. Default : 150

        filename : str
            Name of the `.mp4` file to be generated. File format need not be specified.

        """
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        nframes = X.shape[0]
        fig, ax = plt.subplots()
        #cmap = kwargs.get('cmap','jet')
        im = ax.imshow((self.reshape_fields(X[0])).T, origin='lower' , vmax = vmax,vmin=vmin,animated=True,interpolation='nearest',**kwargs)
        cbar = fig.colorbar(im,ax=ax)
        def update(frame):
            X_ = (self.reshape_fields(X[frame]))#, origin='lower' ,cmap='jet', animated=True)
            im.set_array(X_.T)
            #im.set_clim(vmin=X_.min(), vmax=X_.max())
            #cbar.update_normal(im)
            vmin_ = np.nanmin(X_) if vmin is None else vmin
            vmax_ = np.nanmax(X_) if vmax is None else vmax
            im.set_clim(vmin_, vmax_)
            #cbar.set_clim(vmin, vmax)
            #cbar.draw_all()   # force redraw of colorbar
            ax.set_title(f"Simulation time: {self.dt*(frame+1)}s")
            return [im]
        
        ani = animation.FuncAnimation(fig, update, frames=nframes, blit=False)#True)

        # Save animation as mp4
        ani.save('{f}.mp4'.format(f=filename), fps=fps, dpi=dpi, writer="ffmpeg")

        plt.close(fig)  # prevent extra static figure from showing

    @staticmethod
    def Homogenize(phase, Mat, strain_rate, scale=1e-6,mode='plane-stress', 
                thickness=1.0, tol = 0.005, courant_number=0.4, 
                total_time = 1000, at_every = 100, save_states = False
                ):
        """
        Compute the Homogenized Stiffness matrix in Cartesian space for the `tostada.PhaseDistribution` for a given young's modulus, poisson ratio and mass density. Only relevant if the system obeys periodic boundaries.
        Since this module is only in 2D currently, only three quasi-static tests are needed to extract the 3x3 homogenized stiffness matrix: i) uniaxial x-loading ii) uniaxial y-loading and iii) shear loading
        For each tests, mean stress and strain components are computed. A system of linear equations is then solved to obtain a homogenized stiffness matrix.

        Parameters
        ---------

        phase : tostada.PhaseDistribution object
            Binarized phase distribution indicating pores and solid regions. 
        
        Mat : tostada.Material object
            Material with a well-defined Young's modulus (N/m^2), Poisson ratio and mass density (kg/m^3).
        
        strain_rate : float
            Strain rate for the loading

        thickness : float
            Out-of-plane thickness (in meters) of the material. Controls the spring's constant and non-local volume parameter. Default: 1 m.

        scale : float
            Scaling of the lengths from PhaseDistribution (since PhaseDistribution can often be in microns). Default : 1e-6 m.
        
        mode : str
            Two possible modes in 2D. Either `plane-stress` or `plain-strain`.
        
        total_time : int
            Total runtime of each simulation.

        courant_number : float
            Fraction of the critical time step `dtc` that must be used for the simulations

        tol : float
            Tolerance for reaching convergence in the quasi-static simulation.

        at_every = bool
            Frequency of recording the fields in time. 

        save_states : bool
            Should the `LatticeParticleMethod` states be saved for post-processing or not. Default : None

        Returns
        -------
        Stiffness_mat : ndarray
            The homogenized stiffness matrix of shape 3x3. 
        Eps_h : ndarray
            Homogenized strain tensor. Each column summarizes the mean [epsxx,epsxy,epsyy] for the three tests.
        Sigma_h : 
            Homogenized stress tensor. Each column summarizes the mean [sigxx,sigxy,sigyy] for the three tests.
        """
        image = phase.image[1:,1:] 
        image[:, 0] = image[:, -1]
        image[0, :] = image[-1, :] # corrects any spurious noise at the boundaries inspite of the periodicity
        phase.image = np.bool(image)
        system = LatticeParticleMethod(Phase=phase,Material=Mat,thickness=thickness,is_periodic=True)
        particle_roi = np.full(system.positions.shape[0],True)
        Force_roi = np.full(system.positions.shape,False) 

        # 1) x loading
        print ('Simulating x-loading')
        affine_matrix = np.array([[1+strain_rate, 0],[0,1]]) 
        results = system.run_sim(particle_roi, cp.asarray(Force_roi ), is_velocity=True,
                                num_steps=total_time,at_every=at_every, dt = courant_number*system.dtc,tolerance=tol, 
                                affine_matrix=affine_matrix,damping_factor=0)
        Eps = system.get_strains()
        epsxx,epsxy,epsyy = np.mean(Eps[0]), np.mean(Eps[1]), np.mean(Eps[2])
        Sigma = system.get_stresses(system.Force_pairs)
        sigmaxx,sigmaxy,sigmayy = np.mean(Sigma[0]),np.mean(Sigma[1]),np.mean(Sigma[2])
        xloading_results = cp.asnumpy(cp.asarray([epsxx,2*epsxy,epsyy,sigmaxx,sigmaxy,sigmayy]))

        if (save_states==True):
            xload_state = system
        else:
            xload_state = None

        # 2) y loading
        print ('Simulating y-loading')
        affine_matrix = np.array([[1, 0],[0,1+strain_rate]])
        system = LatticeParticleMethod(Phase=phase,Material=Mat,thickness=thickness,is_periodic=True)
        results = system.run_sim(particle_roi, cp.asarray(Force_roi ), is_velocity=True,
                                num_steps=total_time,at_every=at_every, dt = courant_number*system.dtc,tolerance=tol, 
                                affine_matrix=affine_matrix,damping_factor=0)
        Eps = system.get_strains()
        epsxx,epsxy,epsyy = np.mean(Eps[0]), np.mean(Eps[1]), np.mean(Eps[2])
        Sigma = system.get_stresses(system.Force_pairs)
        sigmaxx,sigmaxy,sigmayy = np.mean(Sigma[0]),np.mean(Sigma[1]),np.mean(Sigma[2])
        yloading_results = cp.asnumpy(cp.asarray([epsxx,2*epsxy,epsyy,sigmaxx,sigmaxy,sigmayy]))
        
        if (save_states==True):
            yload_state = system
        else:
            yload_state = None

        # 3) xy loading / shear
        print ('Simulating shear loading')
        system = LatticeParticleMethod(Phase=phase,Material=Mat,thickness=thickness,is_periodic=True)
        affine_matrix = np.array([[1, strain_rate],[0,1]])
        results = system.run_sim(particle_roi, cp.asarray(Force_roi ), is_velocity=True,
                                num_steps=total_time,at_every=at_every, dt = courant_number*system.dtc,tolerance=tol, 
                                affine_matrix=affine_matrix,damping_factor=0)
        Eps = system.get_strains()
        epsxx,epsxy,epsyy = np.mean(Eps[0]), np.mean(Eps[1]), np.mean(Eps[2])
        Sigma = system.get_stresses(system.Force_pairs)
        sigmaxx,sigmaxy,sigmayy = np.mean(Sigma[0]),np.mean(Sigma[1]),np.mean(Sigma[2])
        xyloading_results = cp.asnumpy(cp.asarray([epsxx,2*epsxy,epsyy,sigmaxx,sigmaxy,sigmayy]))
        
        if (save_states==True):
            xyload_state = system
        else:
            xyload_state = None

        Eps1,Sigma1,Eps2,Sigma2,Eps3,Sigma3 = xloading_results[:3], xloading_results[3:], yloading_results[:3], yloading_results[3:], xyloading_results[:3], xyloading_results[3:] 
        Eps_h = np.array([[Eps1[0],Eps2[0],Eps3[0]],[Eps1[2],Eps2[2],Eps3[2]], [Eps1[1],Eps2[1],Eps3[1]] ] )
        Sigma_h = np.array([[Sigma1[0],Sigma2[0],Sigma3[0]],[Sigma1[2],Sigma2[2],Sigma3[2]], [Sigma1[1],Sigma2[1],Sigma3[1]] ] )
        Stiffness_mat = (Sigma_h@np.linalg.inv(Eps_h))/Mat.youngs_modulus

        return Stiffness_mat,Eps_h, Sigma_h, xload_state,yload_state,xyload_state