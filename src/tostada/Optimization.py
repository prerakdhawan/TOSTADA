import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from scipy.interpolate import interp1d
import numpy as np
from tostada.PhaseDistribution import PhaseDistribution
from tostada.PointDistribution import PointDistribution
import jax
import jax.numpy as jnp
from jax import grad
from jaxopt import LBFGS,ScipyMinimize

class Optimization: 
    """
    Constructs an optimizer class for a reciprocal-space optimization. Currently only implemented for PointDistribution and in 2D.
    """
    def __init__(self,Distribution,Qmax=100,Target=None):
        """
        Parameters
        ----------
        Distribution : obj
            The PointDistribution object whose positions need to be tuned.
        
        Qmax : float
            Maximum Q for which Sq needs to be computed.
        
        Target : ndarray, optional
            Target to be matched for optimization. If None, uses simply S(q). If 1D, uniformly interpolates to a 2D grid.
        
        """
        self.Distribution = Distribution
        self.init_positions = self.Distribution.positions[:,:-1] #Positions before the optimization. z-coordinate ignore currently
        self.Target = Target
        self.Qmax = Qmax
        self.D0 = self.Distribution.diameter
        self.q_i = 2*jnp.pi/self.Distribution.BoxSize[0]

    def optimize(self, q_f, maxiter=1000, tol=1e-8, D0=None, positions=None, masked_pos=None, mask_mode = 'radial',**kwargs):
        """
        Optimizes the positions of the scatterers to the input Target. 
        
        Parameters
        ----------
        q_f : float
            Maximum q value in the masked region. If Target=None, S( q_i < q < q_f ) = 0 (Stealthy HuD)
        maxiter : int, optional
            Maximum value till which the optimization needs to be run. Default : 1000
        tol : float, optional
            Tolerance for the optimization. Default : 1e-8
        D0 : float, optional
            Distance to which the pair-correlation is required to be zero. Currently used for avoiding overlap. If None, uses self.diameter
        positions : ndarray, optional
            Specific positions that need to be optimized. If None, uses positions from self.Distribution. If None, uses self.positions
        masked_pos : ndarray, optional
            Positions that need to be considered for objective function (and gradients) but that are fixed during optimization. Default : None
        mask_mode : bool, optional
            Shape of the ROI in the reciprocal space. Currently allowed inputs are 'radial' and 'square'. Default : 'radial'

        kwargs
        ------
        q_f2 : float, optional
            q_f along the y direction. 
        
        g_trend : str, optional
            Trends for pair-correlation weights. Possible inputs: 'radial' (1-r/D) and 'exp' (exp(-r/D)^2)
        """
        inputs = self.init_positions.flatten() if positions is None else positions.flatten()
        D0 = self.D0 if D0 is None else D0
        self.current_iter=0
        #ub = jnp.c_[jnp.ones(self.Distribution.totalparticles)*(self.Distribution.BoxSize[0]/2),
        #            jnp.ones(self.Distribution.totalparticles)*(self.Distribution.BoxSize[0]/2)].ravel()
        #lb =jnp.c_[-jnp.ones(self.Distribution.totalparticles)*(self.Distribution.BoxSize[0]/2),
        #           -jnp.ones(self.Distribution.totalparticles)*(self.Distribution.BoxSize[1]/2)].ravel()
        ub = jnp.c_[jnp.ones(inputs.shape[0]//2)*(self.Distribution.BoxSize[0]/2),
                    jnp.ones(inputs.shape[0]//2)*(self.Distribution.BoxSize[1]/2)].ravel()
        lb =jnp.c_[-jnp.ones(inputs.shape[0]//2)*(self.Distribution.BoxSize[0]/2),
                   -jnp.ones(inputs.shape[0]//2)*(self.Distribution.BoxSize[1]/2)].ravel()
        bounds=jnp.vstack([lb,ub])
        wrapped_objective = lambda inputs: self.objective_with_custom_grad(inputs, D0, q_f, masked_pos,mask_mode=mask_mode, **kwargs)
        opt = ScipyMinimize(fun=wrapped_objective,method='L-BFGS-B',value_and_grad=True,maxiter=maxiter,tol=tol,options=dict({'disp':True}) ,jit=False)
        result = opt._run(inputs,bounds=bounds) #for scipyminimize
        optimized = result.params.reshape((-1, 2))
        self.opt_positions = optimized
        self.results = result
        print("Current phi =", result.state)
        return result.state, optimized 

    def objective_with_custom_grad(self,inputs,D0,q_f,masked_pos,mask_mode,**kwargs):
        """
        Returns the value of the objective function along with its gradient. 
        Gradient is computed using a custom function (only implemented for PointDistribution currently)
        """
        self.current_iter = self.current_iter+1
        value = self.Objective(inputs,D0,q_f,masked_pos,mask_mode=mask_mode,**kwargs)
        grad = self.custom_grad(inputs,D0,q_f,masked_pos,mask_mode=mask_mode,**kwargs)
        if (np.mod(self.current_iter,5)==0):
            print ('Current iteration : {i}, Objective = {ob} with gradient = {gr}'.format(i=self.current_iter,ob=value,gr=np.abs(grad)))
        return value, grad

    @staticmethod
    def weights_gterm(_cdist,D0,trend=['linear','exp']):
        if (trend=='linear'):
            weights=jnp.maximum(0,1-_cdist/D0)
        elif (trend=='exp'):
            weights = jnp.exp(-(_cdist/D0)**2/0.5) 
        return weights
    
    @staticmethod
    def weights_sterm(Sq_term):
        weights = jnp.where(Sq_term>0,1,0)
        return weights

    def pair_correlation_objective(self,D0,**kwargs):
        Sq = kwargs.get('Sq',self.Sq)
        Gr = self.Distribution.pair_corr_Sq_jax(Sq) 
        self.Gr = Gr
        _cdist = jnp.hypot(self.Gr[1],self.Gr[0])
        trend = kwargs.get('g_trend','linear')
        weights = self.weights_gterm(_cdist,D0,trend=trend)
        f_term = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(weights/jnp.sum(weights))))
        gterm = 1/(jnp.size(f_term)) * jnp.sum(f_term*(Sq[2]-1))
        return gterm

    def grad_pair_correlation_objective(self,D0,**kwargs):
        Gr = kwargs.get('Gr',self.Gr)
        _cdist = jnp.hypot(Gr[1],Gr[0])
        trend = kwargs.get('g_trend','linear')
        weights = self.weights_gterm(_cdist,D0,trend=trend)
        f_term = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(weights/jnp.sum(weights))))
        dgterm = 1/(jnp.size(f_term))*jnp.ravel((f_term))
        return dgterm

    def mask_Sq(self,Sq,q_f,q_f2=None,mode='radial'):
        
        qdist = jnp.hypot(Sq[0], Sq[1])
        if (mode=='radial'):
            masked_S = jnp.where(jnp.logical_or((qdist >= q_f),
                                                (qdist <= self.q_i)), 0, Sq[2])
        elif (mode=='rect'):
            q_f2 = q_f if q_f2 is None else q_f2
            masked_S = jnp.where(jnp.logical_and(jnp.abs(Sq[0])<=q_f,jnp.abs(Sq[1])<=q_f2 ), Sq[2],0 )
            #masked_S[qdist<=self.q_i] = 0
            masked_S = jnp.where(qdist<=self.q_i,0,masked_S)
        return masked_S

    def Objective(self,pos, D0, q_f,masked_pos=None,mask_mode='radial',**kwargs):
        """
        Objective function for the optimization. Includes overlap condition and a Target. Currently only implemented for particle-distributions. 

        Parameters
        ----------
        pos : ndarray
            Nx2 array of positions. 
        D0 : float
            Distance to which the pair-correlation is required to be zero. Currently used for avoiding overlap.
        Q : float
            Maximum Q value to be used for S(q) computation. 
            Since pair-correlation is evaluated from S(q), higher the Q, better the resolution for g2(r).
        q_f : float
            Maximum q value in the masked region. If Target=None, S( q_i < q < q_f ) = 0 (Stealthy HuD)
        masked_pos : ndarray, optional
            Positions that need to be considered for objective function (and gradients) but that are fixed during optimization. Default : None
        """
        if pos.ndim == 1:
            pos = pos.reshape((-1, 2))

        total_pos = pos if masked_pos is None else jnp.vstack([pos,masked_pos])
        batch_size_fac = kwargs.get('batch_size',100)
        Sq = self.Distribution.Sk_jax_custom(custom_pos=total_pos, kmax=self.Qmax,batch_size_fac=batch_size_fac)
        self.Sq = Sq
        Sq_masked = self.mask_Sq(Sq,q_f,mode=mask_mode)
        _Target = jnp.zeros_like(Sq_masked) if self.Target is None else self.mask_Sq(self.Target,q_f,mode=mask_mode)
        weights2 = self.weights_sterm(Sq_masked) #jnp.where(Sq_masked>0,1,0)
        #_Target = Target[2]*weights2
        sterm = jnp.sum(weights2*(Sq_masked -_Target)**2/jnp.sum(weights2))#  
        gterm = 0 if D0 == 0 else self.pair_correlation_objective(D0=D0,Sq=Sq,**kwargs) 
        phi = sterm + gterm
        return phi

    def custom_grad(self,pos,D0, q_f,masked_pos=None,mask_mode='radial',**kwargs):
        """
        Gradient function for matching reciprocal-space response + avoiding overlap. 
        Since overlap condition uses ifft of S(q), overlap is avoided with periodic boundaries too.
        """
        if pos.ndim == 1:
            pos = pos.reshape((-1, 2))
        total_pos = pos if masked_pos is None else jnp.vstack([pos,masked_pos])
        x = jnp.array(pos[:, 0], dtype=jnp.float32)  # Changed from jnp.cast
        y = jnp.array(pos[:, 1], dtype=jnp.float32)  # Changed from jnp.cast
        xt = jnp.array(total_pos[:, 0], dtype=jnp.float32)  # Changed from jnp.cast
        yt = jnp.array(total_pos[:, 1], dtype=jnp.float32)  # Changed from jnp.cast
        Sq = kwargs.get('Sq',self.Sq)
        Kx=jnp.ravel(Sq[0])
        Ky= jnp.ravel(Sq[1])
        k_shape = Kx.shape[0]
        batch_size = int(k_shape/(140)) #,dtype=jnp.int32) #3D needs smaller batch for GPU-memory limit
        arg_chunk = []
        arg_chunk_t = []
        for i in range(0, k_shape, batch_size):
            KX_chunk = Kx[i:i+batch_size]
            KY_chunk = Ky[i:i+batch_size]
            arg_chunks = jnp.outer(x, KX_chunk) + jnp.outer(y, KY_chunk) 
            arg_chunks_t = jnp.outer(xt, KX_chunk) + jnp.outer(yt, KY_chunk) 
            arg_chunk.append(arg_chunks)
            arg_chunk_t.append(arg_chunks_t)
        argu_t = jnp.hstack(arg_chunk_t)
        argu = jnp.hstack(arg_chunk)#,axis=0)
        term_real = jnp.cos(argu)
        term_imag = jnp.sin(argu)
        term_real_t = jnp.cos(argu_t)
        term_imag_t = jnp.sin(argu_t)
        #term = jnp.exp(1j*(argu)) # do not reshape or transpose! grad_check pass with no gr
        term = term_real + 1j * term_imag
        #term_t = term_real_t - 1j * term_imag_t
        #nk = jnp.sum((term_real_t - 1j * term_imag_t ),axis=0)
        nkreal = jnp.sum(term_real_t,axis=0)
        nkimag = jnp.sum(term_imag_t,axis=0)
        nk = nkreal - 1j*nkimag #jnp.sum(term_real_t,axis=0) - 1j*jnp.sum(term_imag_t,axis=0)

        Sq_masked = self.mask_Sq(Sq,q_f,mode=mask_mode) #!jnp.where(jnp.logical_or((kdist >= q_f),(kdist <= self.q_i)), 0,(Sq[2])) 
        weights1 = self.weights_sterm(Sq_masked) #!jnp.where(Sq_masked>0,1,0)
        _Target = jnp.zeros_like(Sq_masked) if self.Target is None else self.mask_Sq(self.Target,q_f,mode=mask_mode)

        f_term1 = (weights1*((Sq_masked)-_Target) ).flatten()/jnp.sum(weights1) 

        dgterm = 0 if D0==0 else self.grad_pair_correlation_objective(D0=D0,**kwargs)
        fk = 2*f_term1+dgterm
        large_term = np.array(term * nk)
        large_term = large_term * np.array(fk)
        #large_term = jnp.array(large_term)
        #large_term = term*nk*fk 

        gradx = np.sum(large_term*np.ravel((Sq[0])),axis=1) #do not change
        grady = np.sum(large_term*np.ravel((Sq[1])),axis=1) #do not change
        grads = jnp.imag(jnp.column_stack([gradx,grady])) #do not change
        #grads = -(2/self.Distribution.totalparticles)*grads.flatten() 
        grads = -(2/total_pos.shape[0])*grads.flatten() 
        return grads

    def finite_diff_grad(func, pos, D0, q_i, qmax,epsilon=1e-3): #1e-5 is too small.
        grad_approx = jnp.zeros_like(pos)
        for i in range(len(pos)):
            pos_fwd = pos.at[i].add(epsilon)
            pos_bwd = pos.at[i].add(-epsilon)
            f_fwd = func(pos_fwd, D0, q_i, qmax)
            f_bwd = func(pos_bwd, D0, q_i, qmax)
            grad_approx = grad_approx.at[i].set((f_fwd - f_bwd) / (2 * epsilon))
        return grad_approx

    def interpolate_Target1D(self,Sq_target,**kwargs):
        """
        Interpolate the 1D target function to a 2D grid. Assumes isotropy in reciprocal space. The 2D grid has resolution of 2\\pi/ PointDistribution.BoxSize
        
        Parameters
        ----------
        Sq_target : Nx2 array
            1D Target function for the optimization prepared as [q , S_target(q)]. Caution: Sq_target must be pre-computed for Qmax larger than np.sqrt(2) * self.Qmax
        """
        Sq_ = np.vstack([np.array([0,0]),Sq_target]) # Added S(0)=0 for full interpolation range
        Sq_int = interp1d(Sq_[:,0],Sq_[:,1])
        batch_size_fac = kwargs.get('batch_size',100)
        Target_grid = self.Distribution.Sk_jax_custom(kmax=self.Qmax,batch_size_fac=batch_size_fac)
        Sq_target_int = Sq_int(jnp.hypot(Target_grid[0],Target_grid[1]))
        return np.asarray([Target_grid[0], Target_grid[1], Sq_target_int])




