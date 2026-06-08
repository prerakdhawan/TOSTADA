from tostada.PointDistribution import PointDistribution
import autograd.numpy as np
import scipy as sp
from tostada.Statistics import RSA_distribution_function,Sq_analytic
from tostada.Optimization import Optimization
import os
from scipy.stats import binned_statistic,norm,cauchy,halfnorm

class Pointprocess:
    """
    Create a definition of a point process for a size, point-spacing (pitch or mean interparticle distance) in 2D/3D. 
    This can be used to create common poing distributions (tostada.PointDistribution) like perturbed square
    or hexagonal lattices or Monte-Carlo type processes (Random-Sequential Adsorption or soft- and hard-core Matern type III).
    
    """
    def __init__(self, BoxSize, diameter, ax, ay=None,az=None,is_3D=False):
        self.ax = ax
        self.ay = ay if ay is not None else ax
        self.az = az if az is not None else ax
        self.is_3D = is_3D
        self.BoxSize = BoxSize
        self.ndim = 2*np.logical_not(self.is_3D) + 3*self.is_3D
        if (np.isscalar(self.BoxSize)==True):
            self.BoxSize = [self.BoxSize,self.BoxSize,self.is_3D*self.BoxSize]
        self.diameter = diameter
        self.particle_density = 1/np.power(self.ax,self.ndim) 
        self.estimated_particles = int(self.particle_density * (self.BoxSize[0]*self.BoxSize[1]*self.BoxSize[2]**self.is_3D) )
        #if (np.mod(self.BoxSize/self.ax,2)!=0):
        #    self.BoxSize = int((self.BoxSize/self.ax)+1)*self.ax
    
    def poisson(self,seed=None,is_periodic=True):
        """
        Generate a poisson-type point distribution in 2D/3D with/without periodic boundaries. The points are uncorrelated.

        Parameters
        ----------

        seed : int, optional
            Random seed for reproducibility
        
        is_periodic : bool, optional
            Condition for periodic boundaries. Default : True.
        """
        if seed is not None:
            np.random.seed(seed)
        coords = np.random.uniform(-1,1,[self.estimated_particles,self.ndim+1]) # since numbers are uncorrelated, the pts can be generated in 3D regardless
        coords = coords*np.array(self.BoxSize) # if Lz=0, the z will be forced to 0.
        if (is_periodic==True):
            pointconfig = PointDistribution(np.mod(coords,self.BoxSize[0]) - np.array(self.BoxSize)/2,self.diameter,self.BoxSize) 
        else:
            pointconfig = PointDistribution(coords,self.diameter,self.BoxSize) 
        return pointconfig

    def hyperplane_intersection(self, seed=None):
        """
        Generate HIP (Hyperplane Intersection Process) point distribution. Such a distribution is anti-hyperuniform (S(q) diverges at q->0). 
        First, random uniform lines are generated inside a box. If an intersection point of a pair of lines lies inside the box, the point is selected for the point distribution. 
        The distribution is always wrapped inside the box and thus obeys periodic boundaries.
        TODO : Currently only implemented for 2D.
        ! Note : Due to nature of the process, the distribution may not have a fixed number of particles and statistical stability is not expected.
        
        Parameters
        -----------

        seed : int, optional
            Random seed for reproducibility
        
        Returns
        --------

        tostada.PointDistribution 
            
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random lines in the Poisson line process
        # Line representation: rho = x*cos(theta) + y*sin(theta)
        num_lines = int(2 * self.BoxSize[0] * (1/self.ax) * np.pi)
        num_lines = int ((self.BoxSize[0]/self.ax)) # along each axis
        theta = np.random.uniform(0, np.pi, num_lines)
        max_rho = self.BoxSize[0] * np.sqrt(2) / 2  # Half diagonal of the box
        rho = np.random.uniform(-max_rho, max_rho, num_lines)

        intersections = []
        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                theta1, theta2 = theta[i], theta[j]
                rho1, rho2 = rho[i], rho[j]
                
                # Check if lines are nearly parallel
                sin_diff = np.sin(theta2 - theta1)
                if abs(sin_diff) < 1e-10:
                    continue  # Skip nearly parallel lines
                
                # Calculate intersection point
                x = (rho1 * np.sin(theta2) - rho2 * np.sin(theta1)) / sin_diff
                y = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / -sin_diff
                
                # Keep points within the box
                if 0 <= x <= self.BoxSize[0] and 0 <= y <= self.BoxSize[1]:
                    intersections.append([x, y])
        
        points = np.array(intersections) if intersections else np.array([]).reshape(0, 2)
        points = np.c_[points[:,0],points[:,1],np.zeros(points.shape[0])]
        return PointDistribution(points-np.array(self.BoxSize)/2,diameter=self.diameter,BoxSize=self.BoxSize)

    def rect_lattice(self,noise=0,correlation_length=0,is_periodic=True):
        """
        Pertubed rectangular lattice in 2D or 3D. Perturbations can either be uniform for each lattice site or correlated by a correlation function. 
        Currently correlation function is a simple gaussian with correlation length. Returns a PointDistribution object for further analysis.
        
        Parameters
        ----------
        noise : float
            Un-correlated perturbations. Normalized to lattice constants.

        correlation_length : float 
            Correlation length for the perturbations. Normalized to lattice constants.
        
        is_periodic : bool, optional
            Condition for periodic boundaries. Default : True.
            
        Returns
        -------
        PointDistribution object 
        """
        x = np.arange(self.ax,self.BoxSize[0]+self.ax,step=self.ax)
        y = np.arange(self.ay,self.BoxSize[1]+self.ay,step=self.ay)
        if (self.is_3D==False):
            particle_num = int(len(x)*len(y))
            x,y = np.meshgrid(x,y)
            z = np.zeros(particle_num)
        else:
            z = np.arange(self.az,self.BoxSize[2]+self.az,step=self.az)
            particle_num = int(len(x)*len(y)*len(z))
            x,y,z = np.meshgrid(y,x,z)
        x = x.ravel() 
        y = y.ravel() 
        z = z.ravel()
        dx = np.random.uniform(low=-noise*(self.ax/2),high=noise*(self.ax/2),size=particle_num)
        dy = np.random.uniform(low=-noise*(self.ay/2),high=noise*(self.ay/2),size=particle_num)
        dz = np.random.uniform(low=-noise*(self.az/2),high=noise*(self.az/2),size=particle_num)
        x = x + dx + (-self.BoxSize[0]/2)
        y = y + dy + (-self.BoxSize[1]/2)
        z = z + dz + (-self.BoxSize[2]/2)    
        out_x= np.logical_or((x>self.BoxSize[0]/2) , (x<-self.BoxSize[0]/2)) #indices for outliers on x axis
        out_y= np.logical_or((y>self.BoxSize[1]/2) , (y<-self.BoxSize[1]/2)) # "" for y axis        
        out_z= np.logical_or((z>self.BoxSize[2]/2) , (z<-self.BoxSize[2]/2)) # "" for z axis
        out = out_x+out_y+self.is_3D*out_z
        z = self.is_3D*z 
        coords=np.c_[x,y,z]
        if (correlation_length!=0):
            for i in range(coords.shape[0]):
                for j in range(coords.shape[0]):
                    rij = np.linalg.norm(coords[i] - coords[j])
                    if (i==j):
                        rij = 1e5 #hack to remove i=j from summation
                    coords[i,0] = coords[i,0] + dx[j]*np.exp(-(rij/(2*correlation_length*self.ax+1e-5))**2)
                    coords[i,1] = coords[i,1] + dy[j]*np.exp(-(rij/(2*correlation_length*self.ay+1e-5))**2)
                    coords[i,2] = self.is_3D*(coords[i,2] + dz[j]*np.exp(-(rij/(2*correlation_length*self.az+1e-5))**2))
        if (is_periodic==True):
            pointconfig = PointDistribution(np.mod(coords,self.BoxSize[0]) - np.array(self.BoxSize)/2,self.diameter,self.BoxSize) 

        else:
            pointconfig = PointDistribution(coords[~out],self.diameter,self.BoxSize)
        return pointconfig

    def hex_lattice(self, noise=0,is_periodic=False):
        """
        Pertubed hexagonal lattice in 2D. Currently only uses uncorrelated noise. Returns a PointDistribution object for further analysis.
        
        Parameters
        ----------
        noise : float
            Un-correlated perturbations. Normalized to lattice constants.        

        Returns
        -------
        PointDistribution object 
        """

        # Calculate number of columns and rows based on domain size and spacing
        #spacing = (2 / np.sqrt(3)) * self.ax
        spacing = (1) * self.ax
        cols = int(np.ceil(self.BoxSize[0] / spacing))
        rows = int(np.ceil(self.BoxSize[1] / (spacing * np.sqrt(3) / 2)))

        # Create meshgrid for even and odd rows
        x_even = np.arange(0, cols * spacing, spacing)
        x_odd = x_even + spacing / 2
        y = np.arange(0, rows * spacing * np.sqrt(3) / 2, spacing * np.sqrt(3) / 2)
        #y = y[:-1]
        # Generate grid points
        x_even_grid, y_even_grid = np.meshgrid(x_even, y[::2])
        x_odd_grid, y_odd_grid = np.meshgrid(x_odd, y[1::2])

        # Flatten and concatenate the grids
        x_coords = np.concatenate([x_even_grid.flatten(), x_odd_grid.flatten()]) 
        y_coords = np.concatenate([y_even_grid.flatten(), y_odd_grid.flatten()]) 
        particle_num = int(len(x_coords))
        # Combine x and y coordinates into a single array
        coords = np.column_stack((x_coords + np.random.uniform(low=-noise*(spacing/2),high=noise*(spacing/2),size=particle_num) , 
                                  y_coords + np.random.uniform(low=-noise*(spacing/2),high=noise*(spacing/2),size=particle_num) ))

        # Filter points within the given width and height
        if (is_periodic==True):
            coords = np.c_[coords[:,0],coords[:,1],np.zeros(coords[:,0].shape[0])] #- (self.BoxSize/2)
            pointconfig = PointDistribution(np.mod(coords,self.BoxSize[0]) - np.array(self.BoxSize)/2,self.diameter,self.BoxSize)
        else:
            coords = coords[(coords[:, 0] < self.BoxSize[0]) & (coords[:, 1] < self.BoxSize[1] )]
            coords = np.c_[coords[:,0],coords[:,1],np.zeros(coords[:,0].shape[0])] #- (self.BoxSize/2)
            pointconfig = PointDistribution(coords - np.array(self.BoxSize)/2,self.diameter,[self.BoxSize[0],self.BoxSize[1] ,0 ])
        #pointconfig = PointDistribution(coords,self.diameter,[self.BoxSize, (np.max(coords[:,1]-np.min(coords[:,1]))) * np.sqrt(3) / 2  ])
        return pointconfig
    
    def fourier_dual_ocp(self,q_f=None,init_state=None,**kwargs):
        """
        Create a fourier-dual of a one-component plasma point distribution with periodic boundaries such that its structure factor is given by S(q) = 1 - exp(-(aq)/pi). 
        Such a distribution is class II hyperuniform (alpha=1). For reference, refer https://doi.org/10.1063/5.0189769.

        Returns
        -------

        tostada.PointDistribution object

        kwargs
        ------
        qmax : float
            Maximum value in the reciprocal space until which the structure factor is computed. 
            Note : since pair-correlations are evaluated using structure factor, larger this value, better resolution in pair correlation (dr = 2pi/qmax).
            Default : 2pi / (diameter/2)   

        tol : float
            Tolerance for the optimizer
        maxiter : int
            Maximum number of iterations for the optimizer

        """
        qmin = 2*np.pi/(self.BoxSize[0])
        q_f =  2*np.pi / (self.ax) if q_f is None else q_f 
        D0 = self.diameter if self.diameter != 0 else self.ax/2
        qmax = kwargs.get('qmax', 2*np.pi / (D0/3))
        print ('choosing qmax = {q1} and q_f = {q2}'.format(q1=qmax,q2=q_f)) 
        q = np.linspace(qmin,qmax,2000)
        Sq_1d = Sq_analytic(q,dmean=self.ax,param=None,option='fourier_dual_ocp')
        if (init_state==None):
            print ('Initial state undefined. Choosing random sequential adsorption process.')
            pts = self.RandomSequentialAdsorption(sdev_histo=0.2*self.ax, D0=0.25*self.ax)
        else:
            pts = init_state
        pointdist = self.reciprocal_space_optimization(target = np.c_[q,Sq_1d],Qmax = 0.9*qmax/np.sqrt(2),q_f=q_f,init_state=pts,**kwargs)
        return pointdist        

    def ginibre(self,q_f=None,init_state=None,**kwargs):
        """
        Create a Ginibre point distribution with periodic boundaries such that its structure factor is given by S(q) = 1 - exp(-(aq)^2/4pi). 
        Such a distribution is class I hyperuniform (alpha=2). For reference, refer https://doi.org/10.1063/5.0189769.

        Returns
        -------

        tostada.PointDistribution object

        kwargs
        ------
        qmax : float
            Maximum value in the reciprocal space until which the structure factor is computed. 
            Note : since pair-correlations are evaluated using structure factor, larger this value, better resolution in pair correlation (dr = 2pi/qmax).
            Default : 2pi / (diameter/2)   

        tol : float
            Tolerance for the optimizer
        maxiter : int
            Maximum number of iterations for the optimizer

        """
        qmin = 2*np.pi/(self.BoxSize[0])
        q_f =  2*np.pi / (self.ax) if q_f is None else q_f 
        D0 = self.diameter if self.diameter != 0 else self.ax/2
        qmax = kwargs.get('qmax', 2*np.pi / (D0/3))
        print ('choosing qmax = {q1} and q_f = {q2}'.format(q1=qmax,q2=q_f)) 
        q = np.linspace(qmin,qmax,2000)
        Sq_1d = Sq_analytic(q,dmean=self.ax,param=None,option='ginibre')
        if (init_state==None):
            print ('Initial state undefined. Choosing random sequential adsorption process.')
            pts = self.RandomSequentialAdsorption(sdev_histo=0.2*self.ax, D0=0.25*self.ax)
        else:
            pts = init_state
        pointdist = self.reciprocal_space_optimization(target = np.c_[q,Sq_1d],Qmax = 0.9*qmax/np.sqrt(2),q_f=q_f,init_state=pts,**kwargs)
        return pointdist

    def anti_hyperuniform(self,q_f=None,init_state=None,**kwargs):
        """
        Create a anti-hyperuniform point distribution such that its structure factor is given by S(q) = 1 + 1/q. 
        Such a distribution is anti-hyperuniform (alpha < 0). 

        Returns
        -------

        tostada.PointDistribution object

        kwargs
        ------
        qmax : float
            Maximum value in the reciprocal space until which the structure factor is computed. 
            Note : since pair-correlations are evaluated using structure factor, larger this value, better resolution in pair correlation (dr = 2pi/qmax).
            Default : 2pi / (diameter/2)   

        tol : float
            Tolerance for the optimizer

        maxiter : int
            Maximum number of iterations for the optimizer    
        """
        qmin = 2*np.pi/(self.BoxSize[0])
        q_f =  2*np.pi / (self.ax) if q_f is None else q_f 
        D0 = self.diameter if self.diameter != 0 else self.ax/2
        qmax = kwargs.get('qmax', 2*np.pi / (D0/3))
        print ('choosing qmax = {q1} and q_f = {q2}'.format(q1=qmax,q2=q_f)) 
        q = np.linspace(qmin,qmax,2000)
        Sq_1d = Sq_analytic(q,dmean=self.ax,param=None,option='anti_hud')
        if (init_state==None):
            print ('Initial state undefined. Choosing random sequential adsorption process.')
            pts = self.RandomSequentialAdsorption(sdev_histo=0.2*self.ax, D0=0.25*self.ax)
        else:
            pts = init_state
        pointdist = self.reciprocal_space_optimization(target = np.c_[q,Sq_1d], Qmax = 0.9*qmax/np.sqrt(2),q_f=q_f,init_state=pts,**kwargs)
        return pointdist

    def hermite_gaussian(self,q_f=None,init_state=None,**kwargs):
        """
        Create a hermite gaussian point distribution with periodic boundaries such that its structure factor is given by S(q) = 1 - v(q) * exp(-q^2/2). 
        Such a distribution is non-hyperuniform (alpha = 0). For reference, refer https://doi.org/10.1063/5.0189769.

        Returns
        -------

        tostada.PointDistribution object

        kwargs
        ------
        qmax : float
            Maximum value in the reciprocal space until which the structure factor is computed. 
            Note : since pair-correlations are evaluated using structure factor, larger this value, better resolution in pair correlation (dr = 2pi/qmax).
            Default : 2pi / (diameter/2)   

        tol : float
            Tolerance for the optimizer

        maxiter : int
            Maximum number of iterations for the optimizer    
        """
        qmin = 2*np.pi/(self.BoxSize[0])
        q_f =  2*np.pi / (self.ax) if q_f is None else q_f 
        D0 = self.diameter if self.diameter != 0 else self.ax/2
        qmax = kwargs.get('qmax', 2*np.pi / (D0/3))
        print ('choosing qmax = {q1} and q_f = {q2}'.format(q1=qmax,q2=q_f)) 
        q = np.linspace(qmin,qmax,2000)
        Sq_1d = Sq_analytic(q,dmean=self.ax,param=kwargs.get('lam',1/15),option='hermite_gaussian')
        if (init_state==None):
            print ('Initial state undefined. Choosing random sequential adsorption process.')
            pts = self.RandomSequentialAdsorption(sdev_histo=0.2*self.ax, D0=0.25*self.ax)
        else:
            pts = init_state
        pointdist = self.reciprocal_space_optimization(target = np.c_[q,Sq_1d], Qmax = 0.9*qmax/np.sqrt(2),q_f=q_f,init_state=pts,**kwargs)
        return pointdist
    
    def reciprocal_space_optimization(self,target,Qmax,q_f=None,init_state=None,**kwargs):
        """
        Create a point distribution with periodic boundaries for a prescribed structure factor. Uses tostada.Optimization for gradient-based optimization.
        Such a distribution is class I hyperuniform.

        target : ndarray
            reciprocal space target. Can be angular-averaged [q, Sq] array or [q_x , q_y, Sq] array.

        Qmax : float
            Maximum value in the reciprocal space until which the structure factor is computed. 
            Note : since pair-correlations are evaluated using structure factor, larger this value, better resolution in pair correlation (dr = 2pi/qmax).
            Default : 2pi / (diameter/2)   

        q_f : float
            Maximum reciprocal space vector until which the structure factor is optimized against a target or simply minimized (S=0). 

        kwargs
        ------

        tol : float
            Tolerance for the optimizer
        maxiter : int
            Maximum number of iterations for the optimizer

        """
        q_f =  2*np.pi / (self.ax) if q_f is None else q_f 
        D0 = self.diameter if self.diameter != 0 else self.ax/2
        maxiter = kwargs.get('maxiter',150)
        tol = kwargs.get('tol',1e-8)
        masked_pos = kwargs.get('masked_pos',None)
        if (init_state==None):
            print ('Initial state undefined. Choosing random sequential adsorption process.')
            pts = self.RandomSequentialAdsorption(sdev_histo=0.2*self.ax, D0=0.25*self.ax)
        else:
            pts = init_state
        opt = Optimization(pts,Qmax= Qmax ) 
        if (target.ndim==2):
            opt.Target = opt.interpolate_Target1D(target)
        else:
            opt.Target = target
        opt_dist = opt.optimize(q_f = q_f,D0=D0,maxiter=maxiter,
                                tol=tol,masked_pos=masked_pos,
                                g_trend=kwargs.get('g_trend','exp'),s_trend=kwargs.get('s_trend','soft'))
        return PointDistribution(opt.opt_positions,diameter=self.diameter,BoxSize=self.BoxSize)

    def RandomSequentialAdsorption(self,sdev_histo=0.06,D0=None,limit=int(1e6),substrate=None):
        """
        Random Sequential Adsorption process in 2D/3D. Sequentially deposits particles based on a half-gaussian probabilistic model until a desired particle density is reached. 
        Avoids overlapping particles.

        Parameters
        ----------
        sdev_histo : float, optional
            Width of histogram used in the probability function. Smaller the value, more likely a new particle will be rejected. Default : 0.06 microns
        limit : int, optional
            Maximum limit for the iterations. 
            Typically, the simulation stops well under this limit but if sdev_histo is very small compared to mean interparticle distance, the simulation will reach near this limit.
        
        substrate : tostada.PhaseDistribution object, optional
            Carry out the deposition on a substrate. The substrate must be defined using tostada.PhaseDistribution. The particles are forbidden from sticking in porous regions.
            If None, assumes a flat substrate with no pores/roughness.

        Returns
        -------

        pointconfig : tostada.PointDistribution object
        """
        D0 = self.diameter if D0 is None else D0

        def is_forbidden(pos,substrate):
            pos = pos + np.array(self.BoxSize)/2
            pos = np.int32(pos/substrate.resolution)
            is_forbidden_roi = np.prod(np.isin(pos[:2],np.array(np.where(substrate.image==0))))           
            return np.bool(is_forbidden_roi)
        #SUPPORT FUNCTIONS
        def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
            """
            Call in a loop to create terminal progress bar
            @params:
                iteration   - Required  : current iteration (Int)
                total       - Required  : total iterations (Int)
                prefix      - Optional  : prefix string (Str)
                suffix      - Optional  : suffix string (Str)
                decimals    - Optional  : positive number of decimals in percent complete (Int)
                length      - Optional  : character length of bar (Int)
                fill        - Optional  : bar fill character (Str)
            """
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
            # Print New Line on Complete
            if iteration == total: 
                print()

        areaF = 1/(self.ax)**self.ndim # target particle density
        FF=0
        mean = self.ax - D0
        coords=np.zeros([1,3],dtype=float)   
        printProgressBar(0, limit, prefix = 'Generating Pattern (T='+str(limit)+'):', suffix = '', length = 30)

        i=0
        
        N_estimate = int(np.power(self.BoxSize[0]/self.ax,self.ndim))
        print ('Estimated particles to be deposited={N} with mean inter-particle distance={d}'.format(N=N_estimate,d=self.ax))
        #placed=0
        placedLastRound=True
        posTree=0
        for i in range(limit):
            Particles = PointDistribution(coords,D0,self.BoxSize)
            FF = Particles.particledensity
            #placed=0
            if FF>=areaF:
                break
            printProgressBar(i, limit, prefix = 'Generating Pattern (T='+str(limit)+'):', suffix = 'Current FF='+str("%.3f" % FF)+' of '+str("%.3f" % areaF), length = 30)

            #randomly decide on (x,y)
            x = np.random.uniform(-self.BoxSize[0]/2,self.BoxSize[1]/2)
            y = np.random.uniform(-self.BoxSize[1]/2,self.BoxSize[1]/2)
            z = np.random.uniform(-self.BoxSize[2]/2,self.BoxSize[2]/2)
            newCoord=[x,y,z]
            #adjsph=Particles.adjacent_particles(Lx=self.BoxSize[0],Ly=self.BoxSize[1])
            adjsph = Particles.tessellate()
            #how many neighbors to consider to calculated sticking probability
            nn = (self.ndim-1)*6 
            if len(adjsph)<nn:
                n_neighbors=len(adjsph)
            else:
                n_neighbors=nn

            if placedLastRound or posTree==0:
                placedLastRound=False
                posTree=sp.spatial.cKDTree(adjsph)
                
            #"""
            Result=posTree.query(newCoord,k=1)
            NNdist=Result[0]
            
            forbidden_region = False if substrate is None else is_forbidden(newCoord,substrate)
            if NNdist<D0:
                continue
            #if NNdist>mean+self.diameter:
            if np.logical_and(NNdist > (mean+D0),forbidden_region==False):
                coords=np.vstack((coords,newCoord))
                #placed+=1
                placedLastRound=True
                continue
            #"""
            randn=np.random.rand()

            Result=posTree.query(newCoord,k=n_neighbors)
                    
            NNdists=Result[0]
            probStick=1
            for neighbor in range(n_neighbors):
                probStick*=RSA_distribution_function(NNdists[neighbor]-D0,mean,sdev_histo)

            #if probStick>=randn:
            if np.logical_and(probStick>=randn,forbidden_region==False):
                coords=np.vstack((coords,newCoord))
                #placed+=1
                placedLastRound=True

        pointconfig = PointDistribution(coords,D0,self.BoxSize)
        print("number of experiments:"+str(i))
        return pointconfig
