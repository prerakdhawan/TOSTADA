import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tostada.util.materials import Material
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
    NOTE : Currently only implemented for 2D systems. Future release will have fracture mechanics (also in 2D).
    """
    #def __init__(self, Phase, Y=6.9e10, v=0.25, rho = 3900, scale=1e-6,mode='plane-stress'):
    def __init__(self, Phase, Material, scale=1e-6,mode='plane-stress'):
        self.scale = scale #converting to meters
        self.Phase = Phase
        self.image = self.Phase.image
        self.n_particles = np.size(self.image)#positions.shape[0]
        self.dx = Phase.resolution*self.scale  # 9.33e-5
        self.Lx = self.Phase.Lx*self.scale
        self.Ly = self.Phase.Ly*self.scale
        self._positions = self.prepare_geometry() #positions
        self.mode = mode
        self.tree = cKDTree(self._positions)
        self.init_positions = self.prepare_geometry() #init_positions
        self.Y = Material.youngs_modulus  # Young's modulus. default=6.9e10
        self.v = Material.poisson_ratio  # Poisson's ratio. default=0.2
        self.K = self.Y / (3 * (1 - 2 * self.v))  # Bulk Modulus
        self.G = self.Y / (2 * (1 + 2 * self.v))  # Shear Modulus
        self.R = self.dx / 2  # radius of each particle
        self.k1, self.k2, self.T = self.get_strain_parameters()

        # Pairlist using KDTree - only done once
        self.pairlist0 = np.asarray([self.tree.query(self.init_positions[i], k=9)[1] for i in range(self.init_positions.shape[0])])
        self.corner_ind, self.bulk_ind, self.edge_ind = self.boundary_indices()

        # Convert pairlist and initial distances to CuPy arrays for GPU acceleration
        self.pairlist0_cp = cp.asarray(self.pairlist0)
        self.init_positions_cp = cp.asarray(self.init_positions)
        self.rho = Material.density #rho
        self.mass = self.rho * np.power(self.dx,2) #change later when moving to 3D
        self.vp = np.sqrt((self.K + (4*self.G/3))/self.rho) #maximum velocity of the waves
        self.dtc = self.dx/self.vp #critical time step for the simulation. Keep dt smaller than dtc for numerical stability.
        self.exclusions = np.where(Phase.image.flatten()==0)[0] #excluded regions in the material like pores.
        self.inclusions = np.where(np.isin(np.arange(self.init_positions.shape[0]), self.exclusions), False, True) 
        self.D0 = self.get_initial_dist_new()

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, new_positions):
        self._positions = new_positions  # Update the internal positions
        self.tree = cKDTree(self._positions)  # Rebuild the KDTree

    def get_strain_parameters(self):
        self.k1 = 2*self.Y /(1+self.v)
        self.k2 = self.Y /(1+self.v)
        if (self.mode=='plane-strain'):
            #self.T = self.Y * (4*self.v - 1) / ( (2+np.sqrt(2) )**2 * (1+self.v)*(1 - 2*self.v))
            self.T = self.Y * (4*self.v - 1) / ( (2) * (1+self.v)*(1 - 2*self.v))

        else:
            #self.T = (0.5*self.Y * (3*self.v-1) * (1-self.v) ) / ( (1+self.v) * (1-2*self.v)**2) #works with get_Es. 
            self.T = self.Y * (3*self.v - 1) / ( (2+np.sqrt(2) )**2 * (1 - self.v**2))

        return self.k1,self.k2,self.T
    
    def prepare_geometry(self):
        #[_y,_x] = np.meshgrid(np.arange(X.image.shape[0]),np.arange(X.image.shape[1])) 
        [_y,_x] = np.meshgrid(np.arange(self.image.shape[0]),np.arange(self.image.shape[1])) 
        positions = np.c_[_x.ravel(),_y.ravel()]*self.dx
        return positions

    def boundary_indices(self):
        ind = self.pairlist0[:,0]
        corner_ind = np.where( (ind == 0) | (ind == self.image.shape[0] - 1) | (ind == self.image.shape[0] * (self.image.shape[1]-1)) | (ind == self.image.size-1))[0]
        edge_ind = np.where( ( (ind < self.image.shape[0] - 1) & (ind > 0) ) | (ind % self.image.shape[0] == 0) | ((ind - (self.image.shape[0]-1)) % self.image.shape[0] == 0) | ( (ind > (self.image.shape[0] * (self.image.shape[1]-1))) & (ind < self.image.size) ))[0]
        bulk_ind = ind[~np.isin(ind,edge_ind)]
        edge_ind = edge_ind[~np.isin(edge_ind, corner_ind)]
        return corner_ind,bulk_ind,edge_ind
    
    def get_spring_matrix(self):
        # Using CuPy for GPU acceleration if available
        SpringMatrix = cp.zeros((self.init_positions_cp.shape[0], 9), dtype=cp.float32) #64
        corner_ind_cp = cp.asarray(self.corner_ind)
        edge_ind_cp = cp.asarray(self.edge_ind)
        bulk_ind_cp = cp.asarray(self.bulk_ind)

        # Assign spring constants for corner particles
        SpringMatrix[corner_ind_cp, 1:3] = self.k1
        SpringMatrix[corner_ind_cp, 3:4] = self.k2

        # Assign spring constants for edge particles
        SpringMatrix[edge_ind_cp, 1:4] = self.k1
        SpringMatrix[edge_ind_cp, 4:6] = self.k2

        # Assign spring constants for bulk particles
        SpringMatrix[bulk_ind_cp, 1:5] = self.k1
        SpringMatrix[bulk_ind_cp, 5:9] = self.k2

        #return SpringMatrix.get()  # Return as NumPy array if needed
        return asnumpy(SpringMatrix)
    
    def get_initial_dist_new(self):
        _idx = self.pairlist0_cp
        D0_new = self.init_positions_cp[_idx[:, 0]][:, None] - self.init_positions_cp[_idx]
        D0_new = cp.linalg.norm(D0_new, axis=2)
        corner_ind_cp = cp.asarray(self.corner_ind)
        edge_ind_cp = cp.asarray(self.edge_ind)
        D0_new[corner_ind_cp, 4:] = 0
        D0_new[edge_ind_cp, 6:] = 0
        #return D0_new.get()  # Converting back to NumPy if necessary
        return asnumpy(D0_new)#.get()  
    
#    def Global_Energies(self,velocities,exclusions):
    def Global_Energies(self,velocities):
        #StrainE = self.strain_energy()
        #U_strain = np.sum(StrainE)
        #KinE = np.sum(0.5*self.mass*np.linalg.norm(velocities,axis=1)**2)
        KinE = np.sum(0.5*self.mass*np.sum(velocities**2,axis=1))
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
        corner_ind_cp = cp.asarray(self.corner_ind)
        edge_ind_cp = cp.asarray(self.edge_ind)
#        bulk_ind_cp = cp.asarray(self.bulk_ind)
        Dnew[corner_ind_cp, 4:] = 0
        Dnew[edge_ind_cp, 6:] = 0
        UnitVectorMatrix_x[corner_ind_cp, 4:] = 0
        UnitVectorMatrix_y[corner_ind_cp, 4:] = 0
        UnitVectorMatrix_x[edge_ind_cp, 6:] = 0
        UnitVectorMatrix_y[edge_ind_cp, 6:] = 0
        #return Dnew.get(), UnitVectorMatrix_x.get(), UnitVectorMatrix_y.get()
        return asnumpy(Dnew), asnumpy(UnitVectorMatrix_x), asnumpy(UnitVectorMatrix_y)
    
    def get_dists_exc(self):
        _idx = self.pairlist0_cp
        positions_cp = cp.asarray(self.positions)
        
        # **Blue: Apply exclusion mask directly to filter pairlist before calculation**
        mask = ~cp.isin(_idx[:, 0], self.exclusions)  # Exclude based on the first index in pairs
        _idx = _idx[mask]  # Filtered pairlist
        
        _Dnew = positions_cp[_idx[:, 0]][:, None] - positions_cp[_idx]
        
        #__Dnew[_idx] = _Dnew
        #print (_Dnew.shape)
        __Dnew = cp.linalg.norm(_Dnew, axis=2)
        Dnew = cp.asarray(np.zeros([self.positions.shape[0],9]))
        Dnew[_idx[:,0]] = __Dnew
#        Dnew = cp.linalg.norm(__Dnew, axis=2)
        unitvec = cp.nan_to_num(_Dnew / __Dnew[:, :, None])
        #unitvec = cp.nan_to_num(__Dnew / Dnew[:, :, None])
        #print (unitvec.shape)
        _Nx = unitvec[:, :, 0]
        _Ny = unitvec[:, :, 1]
        UnitVectorMatrix_x = cp.asarray(np.zeros([self.positions.shape[0],9]))
        UnitVectorMatrix_y = cp.asarray(np.zeros([self.positions.shape[0],9]))
        UnitVectorMatrix_x[_idx[:,0]] = _Nx
        UnitVectorMatrix_y[_idx[:,0]] = _Ny
        corner_ind_cp = cp.asarray(self.corner_ind)
        edge_ind_cp = cp.asarray(self.edge_ind)
        
        # Zero out elements that are invalid (keeping only valid pairs)
        Dnew[corner_ind_cp, 4:] = 0
        Dnew[edge_ind_cp, 6:] = 0
        UnitVectorMatrix_x[corner_ind_cp, 4:] = 0
        UnitVectorMatrix_y[corner_ind_cp, 4:] = 0
        UnitVectorMatrix_x[edge_ind_cp, 6:] = 0
        UnitVectorMatrix_y[edge_ind_cp, 6:] = 0

        #_Nx = cp.asarray(np.zeros_like(self.D0))
        #_Ny = cp.asarray(np.zeros_like(self.D0))
        #_Nx[_idx[:,0]] = UnitVectorMatrix_x
        #_Ny[_idx[:,0]] = UnitVectorMatrix_y
        #return Dnew.get(), UnitVectorMatrix_x.get(), UnitVectorMatrix_y.get() #_Nx.get(), _Ny.get() 
        return asnumpy(Dnew), asnumpy(UnitVectorMatrix_x), asnumpy(UnitVectorMatrix_y)

#    def compute_local_properties(self, roi, exclusions):
    def compute_local_properties(self, roi):
        outside = np.where(roi == False)[0]
        pairs_list = self.pairlist0
        self.pairlist = pairs_list
        #mask = np.where(np.isin(self.pairlist0, self.exclusions), 0, 1)
        mask = ~np.isin(self.pairlist0, self.exclusions)
        mask = mask.astype(float)
        Dnew, self.UnitVectorMatrix_x, self.UnitVectorMatrix_y = self.get_dists()
#        self.StrainMatrix = np.zeros_like(self.D0)
#        self.StrainMatrix[_mask] = Dnew - self.D0[_mask]
        self.StrainMatrix = Dnew - self.D0
        self.SpringMatrix = self.get_spring_matrix()
        self.StrainMatrix = self.StrainMatrix*mask # New. Forces strains to be zero at exclusion sites
        Force_local = self.SpringMatrix * self.StrainMatrix
        #Force_local[exclusions,:] = 0 #new but not needed
        sum_contrib_dense = np.sum(self.StrainMatrix, axis=1)
        sum_contrib_dense[self.exclusions] = 0 #new 
        #beta_and_alpha = np.unique((self.pairlist0[exclusions]).ravel()) 
        #betas = np.setdiff1d(beta_and_alpha,exclusions)
        #gamma_beta_alpha = np.unique((self.pairlist0[betas]).ravel())
        #gamma = np.setdiff1d(gamma_beta_alpha,beta_and_alpha) 
        Force_nonlocal = sum_contrib_dense[self.pairlist0]
        if (self.mode=='plane-strain'):
            self.Force = Force_local + 0.5* (2*np.sqrt(2)-3)*self.T * sum_contrib_dense[:, None] * mask #np.ones([self.positions.shape[0], 9])
                
            Force_nonlocal = Force_nonlocal#*mask . not needed
            Force_nonlocal[:,0] = 0 #zeroth column is local force
            self.Force = self.Force + 0.5*(2*np.sqrt(2)-3)*self.T*Force_nonlocal
#            self.Force = self.Force + sum_contrib_dense[self.pairlist0]
        else:
            #Es = self.get_Es(cp.asarray(sum_contrib_dense))
            self.Force = Force_local + 0.5 * self.T * sum_contrib_dense[:, None] * mask
            #self.Force = Force_local + 0.5*self.T * Es[:, None] * mask#np.ones([self.positions.shape[0], 9])
            Force_nonlocal = Force_nonlocal#*mask
            Force_nonlocal[:,0] = 0
            self.Force = self.Force + 0.5*self.T*Force_nonlocal
            #self.Force = Force_local + self.T * sum_contrib_dense[:, None] * np.ones([self.positions.shape[0], 9])

        self.Force[outside] = 0
        
        self.Force = self.Force * mask #. Not needed

        Force_x = self.Force * self.UnitVectorMatrix_x
        Force_y = self.Force * self.UnitVectorMatrix_y
        Force_i = np.column_stack((np.sum(Force_x, axis=1), np.sum(Force_y, axis=1)))
        return Force_i
    
    def compute_local_properties_exc(self, roi, exclusions):
        outside = np.where(roi == False)[0]
        pairs_list = self.pairlist0
        
        # **Blue: Filter `pairs_list` based on exclusions before processing**
        self.pairlist = pairs_list[~np.isin(pairs_list[:, 0], exclusions)]
        
        Dnew, self.UnitVectorMatrix_x, self.UnitVectorMatrix_y = self.get_dists_exc(cp.asarray(exclusions))
        #_mask = np.isin(self.pairlist0[:,0], exclusions)
        #_mask = self.pairlist[:,0] #self.pairlist0[~_mask][:,0]
        #self.StrainMatrix = np.zeros_like(self.D0)
        #self.StrainMatrix[_mask] = Dnew - self.D0[_mask]
        self.StrainMatrix = Dnew - self.D0
        self.StrainMatrix[exclusions] = 0 #new
        self.SpringMatrix = self.get_spring_matrix()
        
        # **Blue: Apply mask directly to zero out exclusions' contributions**
#        self.StrainMatrix[np.isin(self.pairlist[:, 0], exclusions)] = 0
        Force_local = self.SpringMatrix * self.StrainMatrix
        
        sum_contrib_dense = np.sum(self.StrainMatrix, axis=1)
        
        # **Blue: Set sum_contrib_dense for exclusions to 0**
        sum_contrib_dense[exclusions] = 0
        
        Force_nonlocal = sum_contrib_dense[self.pairlist0]
        
        if self.mode == 'plane-strain':
            self.Force = Force_local + 0.5 * (2 * np.sqrt(2) - 3) * self.T * sum_contrib_dense[:, None]
            Force_nonlocal[:, 0] = 0
            self.Force += 0.5 * (2 * np.sqrt(2) - 3) * self.T * Force_nonlocal
        else:
            Es = self.get_Es(cp.asarray(sum_contrib_dense))
            self.Force = Force_local + 0.5 * self.T * sum_contrib_dense[:, None]
            Force_nonlocal[:, 0] = 0
            self.Force += 0.5 * self.T * Force_nonlocal

        self.Force[outside] = 0
        Force_x = self.Force * self.UnitVectorMatrix_x
        Force_y = self.Force * self.UnitVectorMatrix_y
        Force_i = np.column_stack((np.sum(Force_x, axis=1), np.sum(Force_y, axis=1)))
        return Force_i


    def get_stresses(self):
        """
        Computes the local stress tensor components. Directly comparable with COMSOL.
        
        Returns
        -------
        Stress tensor components. Since the code is 2D, currently, only gives sigma_xx,sigma_xy and sigma_yy.
        """
        #LF = (1/(2*self.dx*self.dx))*sp.csr_matrix(self.D0).multiply(self.Force)
        LF = (1/(2*self.dx*self.dx))*self.D0*self.Force
        sigmaxx = np.sum(LF*self.UnitVectorMatrix_x*self.UnitVectorMatrix_x,axis=1)
        sigmaxy = np.sum(LF*self.UnitVectorMatrix_x*self.UnitVectorMatrix_y,axis=1)
        sigmayy = np.sum(LF*self.UnitVectorMatrix_y*self.UnitVectorMatrix_y,axis=1)
        von_mises = np.sqrt( np.power(sigmaxx,2) + np.power(sigmayy,2) + 3*np.power(sigmaxy,2) - (sigmaxx*sigmayy) )

        sigma1=(sigmaxx+sigmayy)/2 + np.sqrt( np.power((sigmaxx-sigmayy),2)/4 + np.power(sigmaxy,2) )
        sigma2=(sigmaxx+sigmayy)/2 - np.sqrt( np.power((sigmaxx-sigmayy),2)/4 + np.power(sigmaxy,2) )
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
    
    def get_strains(self):
        """
        Computes the Green-Lagrange components of the strain tensor from the displacements. Directly comparable with COMSOL.
        
        Returns
        -------
        Strain tensor components. Since the code is 2D, currently, only gives Eps_xx,Eps_xy and Eps_yy.
        """
        U,V = self.reshape_fields(self.get_displacements())
        dU_dy, dU_dx = np.gradient(U.T, self.dx, self.dx)
        dV_dy, dV_dx = np.gradient(V.T, self.dx, self.dx) 
        Eps_xx = 0.5*(2*dU_dx + dU_dx**2 + dV_dx**2)
        Eps_yy = 0.5*(2*dV_dy + dU_dy**2 + dV_dy**2)
        Eps_xy = 0.5*(dU_dy + dV_dx + dU_dx*dU_dy + dV_dx*dV_dy)
        return Eps_xx,Eps_xy,Eps_yy

    def strain_energy(self):
        StrainEnergy=np.sum(self.SpringMatrix*self.StrainMatrix**2,axis=1) + 0.5 * self.T * np.sum(self.StrainMatrix,axis=1)*np.sum(self.StrainMatrix,axis=1)
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
    
    def run_sim(self,roi,Force,dt=None,num_steps=3000,tolerance=1.5e-3,at_every=20):
        """
        Runs quasi-static LPM simulation using applied forces at boundaries. For damage simulation, use 'run_sim_damage' (available in next update).
        """
        particle_roi = roi
        num_particles = self.positions.shape
        dt=0.6*self.dtc if dt is None else dt 
        self.dt = dt #save it for further post-process
        Force_ext = Force #np.zeros(positions.shape)
        velocities = (dt/self.mass) * Force_ext
        Force_hist = []
        pos_hist = []
        vel_hist = []
        Force_hist = []
        Damage=[]
        sigma_hist=[]
        Ustrain,KinE = np.zeros(num_steps),np.zeros(num_steps)
        iter_array = [] #to record the steps where quantities are recorded
        KinE0 = self.Global_Energies(velocities)
        for i in range(1, num_steps):        

            r0 = self.positions
            v0 = velocities        
            KinE0=self.Global_Energies(v0)
            half_pos = r0 + (dt/2)*v0

            self.positions = half_pos

            Force_i = self.compute_local_properties(particle_roi)

            velocities = v0 + (dt)*(Force_ext - Force_i)/self.mass
            self.positions = r0 + (dt/2)*(v0 + velocities)/2
            #Ustrain[i],KinE[i] = self.Global_Energies(velocities)
            KinE[i] = self.Global_Energies(velocities)
            #if (KinE[i]<=self.Global_Energies(v0)[1]):
            if (KinE[i]<=KinE0):
                velocities = v0*0 #cease the motion
                print ('current KE=',KinE[i])
                print ('Ratio=',KinE[i]/np.max(KinE))
                print ('velocities set to zero at t=',i)
                if (KinE[i]/np.max(KinE)<tolerance):
                    print ('Simulation converged')
                    #Force_hist.append(Force_i)
                    #Force_hist.append(Force_i)
                    pos_hist.append(self.positions)
                    #vel_hist.append(velocities)
                    stresses = self.get_stresses()
                    sigma_hist.append(stresses)
                    iter_array.append(i)
                    break
            
            if (np.mod(i,at_every)==0):
                #Force_hist.append(Force_i)
                #Force_hist.append(Force_i)
                pos_hist.append(self.positions)
                #vel_hist.append(velocities)
                stresses = self.get_stresses()
                sigma_hist.append(stresses)
                iter_array.append(i)
                print ('current time step={n}'.format(n=i))

            #KinE0 = KinE[i] #save current step's energy for next time step
            self.iter_array = np.copy(np.asarray(iter_array))
            self.sigma_history = np.copy(np.asarray(sigma_hist)) #stresses recorded 
            self.position_history = np.copy(np.asarray(pos_hist)) #positions recorded
        self.KE_history = KinE
        return None

    def boundary_condition(self,edge,value=None):
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
        Arr[self.exclusions] = False #Remove spuriously included regions, if any
        if (value is not None):
            Component = np.zeros(self.positions.shape[0])
            Component[Arr] = value
        else:
            Component = Arr
        return Component
    
    def plot_stress(self,ind=None,ax=None,vmax=None,mode=['sigmaxx','sigmaxy','sigmayy','von-mises','principal-1','principal-2'],cmap='turbo'):
        from matplotlib.ticker import ScalarFormatter
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        if (mode=='sigmaxx'):
            _i = 0
        elif (mode=='sigmaxy'):
            _i = 1
        elif (mode=='sigmayy'):
            _i = 2
        elif (mode=='Von Mises'):
            _i = 3
        elif (mode=='principal-1'):
            _i = 4
        elif (mode=='principal-2'):
            _i = 5
        ind = -1 if ind is None else ind
        stress = self.sigma_history[ind,_i,:]#*self.dx
        stress[self.exclusions] = np.nan
        if not hasattr(self, 'damage_hist'):
            damage = self.exclusions
        else:
            damage = self.damage_hist[ind]
        stress[damage]=np.nan

        _fig = ax.imshow(self.reshape_fields(stress).T,origin='lower',cmap=cmap,
                  vmax=vmax,extent=[0,self.Lx,0,self.Ly])
        cbar=ax.figure.colorbar(_fig,ax=ax)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # Forces scientific notation for all numbers
        cbar.ax.yaxis.set_major_formatter(formatter)
        ax.set_title('Stress: {md}, time={t}s'.format(t=np.around(self.iter_array[ind]*self.dt,10), md=mode ))
        return ax
    
    def surf_integration(self,quantity):
        """
        Integrates a local quantity (like stress/strain components or etc) over the solid phase region (excludes the pores).
        """
        inclusions = np.where(np.isin(np.arange(self.positions.shape[0]), self.exclusions), False, True)
        _surf_int = np.trapz(quantity[inclusions],dx=self.dx**2)
        return _surf_int
    
    def EnergyDensity(self):
        """
        Energy Density from sigma and eps tensor. Essentially does a Frobenius norm: sigma_ij epsilon_ij.
        """
        Stresses = self.get_stresses()
        Strains = self.get_strains()
        return 0.5*(Stresses[0]*Strains[0] + Stresses[1]*Strains[1] + Stresses[2]*Strains[2])

    def effective_quantities(self):
        """
        Homogenized global quantities. 
        """
        eps_xx,eps_xy,eps_yy = self.get_strains()
        sigma_xx,sigma_xy,sigma_yy,sigma_vm,sigma_p1,sigma_p2 = self.get_stresses()
        #dLx,dLy,Lxnew,Lynew = self.total_displacements()
        youngs_eff_xx = np.mean(sigma_xx)/np.mean(eps_xx)
#        poisson_eff = -np.mean(eps_yy.flatten()[self.inclusions])/np.mean(eps_xx.flatten()[self.inclusions])
        poisson_eff = -np.mean(eps_yy.flatten())/np.mean(eps_xx.flatten())
        youngs_eff_xy = np.mean(sigma_xy)/np.mean(eps_xy)
        youngs_eff_yy = np.mean(sigma_yy)/np.mean(eps_yy)
        stress_conc = np.max(sigma_vm)/np.mean(sigma_vm[self.inclusions])
        return stress_conc,poisson_eff,youngs_eff_xx,youngs_eff_xy,youngs_eff_yy