import numpy as np
import jax
import jax.numpy as jnp
from tostada.PhaseDistribution import PhaseDistribution
from jaxopt import LBFGS, ScipyBoundedMinimize
jax.config.update("jax_enable_x64", True)  


class Optimization_phase: 
    """
    Constructs an optimizer class for a reciprocal-space optimization of PhaseDistributions. 
    Uses Topology-Optimization style LBFGS optimization with sigmoid for binarization.
    Implemented for 2D/3D PhaseDistributions. TODO : Test for 3D.
    """
    def __init__(self,Distribution,q_f,Target=None, s_trend='hard',filter_radius=None):
        """
        Parameters
        ----------
        Distribution : PhaseDistribution
            The phase distribution object whose image will be optimized. Can be 2D or 3D

        q_f : float
            Exclusion zone radius.

        Target : array or None
            Target spectral density. If None, target is zero.

        s_trend : str
            Mask type for the weights in reciprocal-space. If 'soft', gives a smooth w(q). Default : 'hard' (w(q)=1 or 0 in the ROI) 
        
        filter_radius : float
            Minimum feature size in the optimization. Larger value gives larger features in the final structure

        """
        self.Distribution = Distribution
        self.init_image = np.asarray(Distribution._image, dtype=np.float64)
        self.Target = Target
        self.q_f = q_f
        self.ndim = Distribution.ndim
        _axes = [Distribution.fftx, Distribution.ffty]
        _fz   = getattr(Distribution, 'fftz', None)
        if _fz is not None:
            _axes.append(_fz)
        self._qgrid = jnp.meshgrid(*[jnp.asarray(a) for a in _axes])
        self._qdist = jnp.sqrt(sum(q**2 for q in self._qgrid))
        self._zone_mask = jnp.where(self._qdist <= q_f, 1.0, 0.0)
        if s_trend == 'soft':
            self._weights = jnp.exp(-((self._qdist/q_f)**2)) * self._zone_mask
        else:
            self._weights = self._zone_mask
        self._weight_sum = jnp.sum(self._weights)
        self.Q_feature = None
        self._filter_gain = 1.0
        if filter_radius is not None:
            _qu = jnp.fft.ifftshift(self._qdist)
            self.Q_feature = jnp.exp(-(_qu*filter_radius)**2/2)  
            _f  = np.asarray(self._density_filter(jnp.asarray(self.init_image), self.Q_feature))
            self._filter_gain = float(np.std(self.init_image)) / (float(np.std(_f)) + 1e-20)
        self.opt_image = None
        self.result = None
        self._beta_state = [10.0]
        self._call_count = [0]

    @staticmethod
    def _spectraldensity_jax(image):
        """
        JAX-native spectral density. Identical to the version in PhaseDistribution.
        """
        volumefraction = jnp.mean(image)

        xq = jnp.fft.fftshift(jnp.fft.fftn(image - volumefraction))
        xq = jnp.abs(xq) ** 2 / image.size

        return xq

    @staticmethod       
    def _weights_2d(Xq_masked, qx=None, qy=None, q_f=None, trend='hard',zone_mask=None):
        """
        Weighting function over the exclusion zone.
        """
        if trend == 'hard':
            #weights = jnp.where(Xq_masked > 0, 1, 0)
            weights = zone_mask if zone_mask is not None else jnp.where(Xq_masked > 0, 1, 0)
        elif trend == 'soft':
            qdist   = jnp.linalg.norm(jnp.array([qx, qy]), axis=0)
            weights = jnp.exp(-((qdist / q_f) ** 2)) 
            if zone_mask is not None:            
                weights = weights * zone_mask
        return weights

    @staticmethod  
    def tanh_binarize(x, beta=10.0,x0=0.5):
        """
        Smooth approximation to a binary step for obtaining binarized features.
        Parameters
        ----------
        x : ndarray
            Dataset to be binarized
        beta : float
            Controls the steepness of the mapping. 
        x0 : float
            Center value.

        """
        return (jnp.tanh(beta * x0) + jnp.tanh(beta * (x - x0))) / (jnp.tanh(beta * x0) + jnp.tanh(beta * (1.0 - x0)))

    def _compute_phi(self, image, gamma=0.0, lam_v=0.0, target_volume=None):
        """
        Evaluate the spectral density objective.

        Parameters
        ----------
        image : (N, M) jnp.ndarray
            Current phase field
        
        gamma : float
            Variance in the values. 
        
        lam_v : float
            Weight for the volume fraction term in the objective
        
        target_volume : float
            Target value of the volume fraction (ratio) to be achieved

        Returns
        -------
        phi : scalar jnp.array

        """
        Xq = self._spectraldensity_jax(image)#, self.Distribution)
        Xq_masked  = Xq * self._zone_mask
        _Target = jnp.zeros_like(Xq_masked) if self.Target is None else self.Target*self._zone_mask
        phi = (jnp.sum(self._weights * (Xq_masked-_Target) ** 2) / (self._weight_sum + 1e-12)) 
        if target_volume is not None:  
            phi = phi / (target_volume * (1.0 - target_volume)) ** 2  
            phi = phi + lam_v * (jnp.mean(image) - target_volume) ** 2 
        phi = phi + gamma * jnp.mean(4.0 * image * (1.0 - image))
        return phi

    @staticmethod
    def _density_filter(image, q0, gain=1.0):
        """Convolve with a radial kernel in Fourier space for mininum length-scale control). """ 
        f = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(image) * q0))
        if (gain == 1.0):
            return f
        m = jnp.mean(f)
        return jnp.clip(m + (f - m) * gain, 0.0, 1.0)

    def optimize(self, maxiter=1000, binarize=False, beta_start=1.0, beta_end=500.0,target_volume=0.5,**kwargs):
        '''
        Optimizer (using L-BFGS-B) for the phase distribution image. Optionally binarizes the image using a sigmoid function with annealing.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations for the optimizer.
        
        binarize : bool
            Whether to binarize the image using a sigmoid function.
        
        beta_start : float
            Initial steepness for the sigmoid binarization.
        
        beta_end : float
            Maximum steepness for the sigmoid binarization (not always reached, depends on convergence).
        
        target_volume : float
            Target volume fraction. Default : 0.5

        Returns
        -------
        phase_opt : PhaseDistribution 

        '''
        self._beta_state = [beta_start]
        self._call_count = [0]
        self._gamma_state = [0.0]
        beta_growth = kwargs.get('beta_growth',2.0)
        gamma_min = kwargs.get('gamma_min', 1e-4)     #!
        gamma_end = kwargs.get('gamma_end', 0.1)    #!
        lam_v = kwargs.get('lam_v', 1e-1) #!
        n_stages = kwargs.get('n_stages', 6) #!
        early_maxiter = kwargs.get('early_maxiter', 60)
        final_maxiter = kwargs.get('final_maxiter', min(maxiter, 300))
        hard_threshold = kwargs.get('hard_threshold', False) #!
        x0 = self.Distribution.volumefraction
        x0 = float(np.quantile(self.init_image, 1.0 - target_volume))
        if self.Q_feature is not None:
            _if = np.asarray(self._density_filter(jnp.asarray(self.init_image),self.Q_feature,self._filter_gain))
            x0 = float(np.quantile(_if, 1.0 - target_volume))
        I_flat = jnp.array(self.init_image.ravel())
        N      = I_flat.shape[0]
        self.history = []
        if binarize:
            betas = [beta_start]
            while betas[-1] < beta_end:
                betas.append(min(betas[-1] * beta_growth, beta_end))
        else:
            betas = [None] * n_stages
        gammas = list(np.geomspace(gamma_min, gamma_end, len(betas)))

        def callback(xk):
            self._call_count[0] += 1

        @jax.jit
        def _vg(I_flat, beta, gamma):
            def _phi(I_flat):
                image = I_flat.reshape(self.init_image.shape)
                if self.Q_feature is not None:
                    image = self._density_filter(image, self.Q_feature,self._filter_gain)
                if binarize:
                    image = self.tanh_binarize(image, beta=beta, x0=x0)
                return self._compute_phi(image, gamma=gamma, lam_v=lam_v, target_volume=target_volume)
            return jax.value_and_grad(_phi)(I_flat)

        def wrapped(I_flat):
            _b = 0.0 if self._beta_state[0] is None else self._beta_state[0] 
            return _vg(I_flat, jnp.float64(_b), jnp.float64(self._gamma_state[0]))
        
        _b0 = 0.0 if betas[0] is None else betas[0]
        _f0, _g0 = _vg(I_flat, jnp.float64(_b0), jnp.float64(gammas[0]))
        f_scale = float(jnp.abs(_f0))
        g_scale = float(jnp.max(jnp.abs(_g0)))

        lb     = jnp.zeros(N)
        ub     = jnp.ones(N)
        bounds = bounds = (lb, ub) #jnp.vstack([lb, ub])
        for k, (beta_k, gamma_k) in enumerate(zip(betas, gammas)):
            _last = (k == len(betas) - 1) 
            _mi   = final_maxiter if _last else early_maxiter
            if binarize:
                self._beta_state[0] = beta_k 
            self._gamma_state[0] = gamma_k
            self._call_count[0] = 0 
            opt = ScipyBoundedMinimize(
                fun            = wrapped,
                method         = 'L-BFGS-B',
                value_and_grad = True,
                maxiter        = _mi,
                tol            = None,
                jit            = False,
                options        = {
                    'disp'  : False,#True,
                    'ftol'  : max((1e-10 if _last else 1e-7) * f_scale, 1e-16),
                    'gtol'  : max((1e-6  if _last else 1e-4) * g_scale, 1e-16),
                    'maxcor' : 5
                },
                callback       = callback,   # called once per iteration by scipy
            )

            self.result  = opt.run(I_flat, bounds=bounds)

            I_flat        = self.result.params
            img_k = I_flat.reshape(self.init_image.shape)
            if self.Q_feature is not None: 
                img_k = self._density_filter(img_k, self.Q_feature,self._filter_gain) 
            if binarize:
                img_k = self.tanh_binarize(img_k, beta=beta_k, x0=x0)

            Xq_k  = self._spectraldensity_jax(img_k) 
            _spec = float(jnp.sum(self._weights * (Xq_k*self._zone_mask)**2)
                          / (self._weight_sum + 1e-12)
                          / (target_volume*(1.0-target_volume))**2)
            self.history.append(dict( current_obj = _spec,
                                     stage=k, beta=beta_k, gamma=gamma_k,
                                     maxiter=self._call_count[0],
                                     M_nd=float(100.0 * jnp.mean(4.0 * img_k * (1.0 - img_k))),
                                     vf=float(jnp.mean(img_k)), vf_drift=float(jnp.mean(img_k)) - target_volume))
            print(self.history[-1])
        final_image = np.array(img_k)
        if hard_threshold:
            thr = np.quantile(final_image, 1.0 - target_volume) 
            final_image = (final_image > thr).astype(np.float32)
        self.opt_image       = final_image
        phase_opt = PhaseDistribution(final_image, self.Distribution.resolution)
        return phase_opt