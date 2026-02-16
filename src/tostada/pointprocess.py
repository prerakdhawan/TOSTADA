from tostada.PointDistribution import PointDistribution
import autograd.numpy as np
import scipy as sp
import tostada.util.Utility
from tostada.Statistics import RSA_distribution_function, MaternIII_distribution_function
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
        #if (np.mod(self.BoxSize/self.ax,2)!=0):
        #    self.BoxSize = int((self.BoxSize/self.ax)+1)*self.ax
    
    def rect_lattice(self,noise=0,correlation_length=0,is_periodic=False):
        """
        Pertubed rectangular lattice in 2D or 3D. Perturbations can either be uniform for each lattice site or correlated by a correlation function. 
        Currently correlation function is a simple gaussian with correlation length. Returns a PointDistribution object for further analysis.
        
        Parameters
        ----------
        noise : float
            Un-correlated perturbations. Normalized to lattice constants.
        correlation_length : float 
            Correlation length for the perturbations. Normalized to lattice constants.
        
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
            pointconfig = PointDistribution(np.mod(coords,self.BoxSize[0]),self.diameter,self.BoxSize) 
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
        coords = coords[(coords[:, 0] < self.BoxSize[0]) & (coords[:, 1] < self.BoxSize[1] )]
        coords = np.c_[coords[:,0],coords[:,1],np.zeros(coords[:,0].shape[0])] #- (self.BoxSize/2)
        pointconfig = PointDistribution(coords - np.array(self.BoxSize)/2,self.diameter,[self.BoxSize[0],self.BoxSize[1] ,0 ])
        #pointconfig = PointDistribution(coords,self.diameter,[self.BoxSize, (np.max(coords[:,1]-np.min(coords[:,1]))) * np.sqrt(3) / 2  ])
        return pointconfig
    
    def RandomSequentialAdsorption(self,sdev_histo=0.06,limit=int(1e6),substrate=None):
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
        mean = self.ax - self.diameter
        coords=np.zeros([1,3],dtype=float)   
        printProgressBar(0, limit, prefix = 'Generating Pattern (T='+str(limit)+'):', suffix = '', length = 30)

        i=0
        
        N_estimate = int(np.power(self.BoxSize[0]/self.ax,self.ndim))
        print ('Estimated particles to be deposited={N} with mean inter-particle distance={d}'.format(N=N_estimate,d=self.ax))
        #placed=0
        placedLastRound=True
        posTree=0
        for i in range(limit):
            Particles = PointDistribution(coords,self.diameter,self.BoxSize)
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
            nn = (self.ndim-1)*6 #6 neighbours for interaction in a 2D hexagonal lattice. 12 for cubic lattice. 
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
            if NNdist<self.diameter:
                continue
            #if NNdist>mean+self.diameter:
            if np.logical_and(NNdist > (mean+self.diameter),forbidden_region==False):
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
                #probStick*=stickingProb(NNdists[neighbor]-self.diameter,mean,sdev_histo)
                probStick*=RSA_distribution_function(NNdists[neighbor]-self.diameter,mean,sdev_histo)

            #if probStick>=randn:
            if np.logical_and(probStick>=randn,forbidden_region==False):
                coords=np.vstack((coords,newCoord))
                #placed+=1
                placedLastRound=True

        pointconfig = PointDistribution(coords,self.diameter,self.BoxSize)
        print("number of experiments:"+str(i))
        return pointconfig