import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tostada.PhaseDistribution import PhaseDistribution
from tostada.PointDistribution import PointDistribution
import tostada.Statistics as stats
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

class Visualize:
    def __init__(self, distribution):
        self.distribution = distribution

    def plot_distribution(self,ax=None,cmap='viridis',facecolor='tab:olive'):
        """
        Plots phase distribution or point-distribution (with fixed diameter). For 3D phase data, currently only plots a 2D slice. 
        For 3D point distribution, plot the 3D scattering data using pyvista or matplotlib's 3D tools.
        
        Parameters
        ----------
        ax : matplotlib.axes, optional
            Matplotlib axes. Can be used to combine several plots. If not provided, creates a new figure.
        
        cmap : matplotlib.colormap, optional
            Colormap for the phase distribution. Default: 'viridis'
        
        facecolor : matplotlib.colors, optional
            Color of the discrete particles in point-distribution
        
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            #self.fig, self.ax = plt.subplots(figsize=self.figsize)
        if isinstance(self.distribution, PointDistribution):
            return self._plot_pointdistribution(ax,facecolor)
        elif isinstance(self.distribution, PhaseDistribution):
            return self._plot_phasedistribution(ax,cmap=cmap)
        else:
            raise ValueError("Unsupported class type. Please provide an instance of PointDistribution or PhaseDistribution.")
    
    def plot_reciprocal_space(self,ax=None,kmax=60,vmax=20,cmap='cividis'):
        """
        Plots reciprocal space data of the distribution and its angular-average.
        If PointDistribution, plots the structure factor S(q), If PhaseDistribution, plots the spectral density X(q).
        If 3D, plots the central 2D slice of X(q) or S(q).
        """
        if ax is None:
            #fig, ax = plt.subplots(figsize=(7, 6))
            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
        if isinstance(self.distribution, PointDistribution):
            #return self._plot_structurefactor(ax,kmax,vmax,cmap)
            return self._plot_structurefactor([ax1,ax2],kmax,vmax,cmap)
        elif isinstance(self.distribution, PhaseDistribution):
            #return self._plot_spectraldensity(ax,kmax,vmax,cmap)
            return self._plot_spectraldensity([ax1,ax2],kmax,vmax,cmap)
        else:
            raise ValueError("Unsupported class type. Please provide an instance of PointDistribution or PhaseDistribution.")
        
    def plot_real_space_correlation(self,ax=None,dr=None,Rmax=10,vmax=20,cmap='cividis'):
        """
        Plots two-point statistics of the dataset. 
        If PointDistribution, plots the two-point correlation function g_2(r). If PhaseDistribution, plots the auto-covariance ACF(r).
        """
        if ax is None:
            #fig, ax = plt.subplots(figsize=(7, 6))
            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
        if isinstance(self.distribution, PointDistribution):
            #return self._plot_structurefactor(ax,kmax,vmax,cmap)
            return self._plot_g2r([ax1,ax2],dr,Rmax,vmax,cmap)
        elif isinstance(self.distribution, PhaseDistribution):
            #return self._plot_spectraldensity(ax,kmax,vmax,cmap)
            return self._plot_ACF([ax1,ax2],dr,Rmax,vmax,cmap)
        else:
            raise ValueError("Unsupported class type. Please provide an instance of PointDistribution or PhaseDistribution.")

    def _plot_pointdistribution(self,ax,facecolor):
        #fig, ax = plt.subplots()
        #if (self.distribution.ndim==2):
        ax.scatter(self.distribution.positions[:,0], self.distribution.positions[:,1])

        patches = [Circle((xi, yi), radius=self.distribution.diameter/2) for xi, yi in zip(self.distribution.positions[:,0], self.distribution.positions[:,1])]
        collection = PatchCollection(patches,
                                    facecolor=facecolor,
                                    edgecolor='k',
                                    alpha=1)
        ax.add_collection(collection)
        ax.set_title('Point Distribution, mean interparticle distance = {D}'.format(D=self.distribution.dmean))
        ax.set_xlabel('X ($\mu$m)',fontsize=15)
        ax.set_ylabel('Y ($\mu$m)',fontsize=15)
        return ax

    def _plot_phasedistribution(self,ax,cmap):
        "Check for 3D"
        if (self.distribution.ndim==2):
            cax = ax.imshow(self.distribution.image.T/np.max(self.distribution.image), origin='lower',
                            extent=[0,self.distribution.Lx,0,self.distribution.Ly],cmap = cmap)
        else:
            print ('3D Data. Plotting for y={y0} slice'.format(y0=self.distribution.Ly//2))
            cax = ax.imshow(self.distribution.image[:,self.distribution.BoxSize[1]//2,:].T/np.max(self.distribution.image),
                            aspect='auto', origin='lower',
                            extent=[0,self.distribution.Lx,0,self.distribution.Lz], cmap=cmap)
        ax.figure.colorbar(cax,ax=ax)
        ax.set_title('Phase Distribution')
        ax.set_xlabel('X ($\mu$m)',fontsize=15)
        ax.set_ylabel('Z ($\mu$m)',fontsize=15)
        return ax
    
    def _plot_spectraldensity(self,ax,kmax,
                              vmax,cmap):
        #Xq = self.distribution.Spectraldensity()
        Xq_ = self.distribution.ReciprocalSpace()
        Xq = Xq_[0]
        if (self.distribution.ndim==2):
            cax = ax[0].pcolormesh(Xq[0],Xq[1],Xq[2],vmax=vmax,cmap=cmap)
        else:
            print ('3D data. Plotting for qy=0 slice')
            cax = ax[0].pcolormesh(Xq[0][:,:,int(Xq[2].shape[2]/2)],
                                   Xq[1][:,:,int(Xq[2].shape[2]/2)],
                                   Xq[3][:,:,int(Xq[3].shape[2]/2)],vmax=vmax,cmap=cmap)
            
        ax[0].figure.colorbar(cax,ax=ax[0])
        ax[0].set_title('Reciprocal Space, Spectral density $\\tilde{X}(\mathbf{q})$')#'Point Distribution, mean interparticle distance = {D}'.format(self.distribution.dmean))
        ax[0].set_xlabel('$q_{x}$ ($\mu$m$^{-1}$)',fontsize=15)
        ax[0].set_ylabel('$q_{y}$ ($\mu$m$^{-1}$)',fontsize=15)
        ax[0].set_xlim(-kmax,kmax)
        ax[0].set_ylim(-kmax,kmax)

        Xq_averaged = Xq_[1]
        ax[1].plot(Xq_averaged[:,0],Xq_averaged[:,1])
        ax[1].set_xlim(0,kmax)
        ax[1].set_xlabel('|q| ($\mu$m$^{-1}$)',fontsize=14)
        ax[1].set_ylabel('$\\tilde{X}(|\mathbf{q}|)$',fontsize=14)
        return ax
    
    def _plot_structurefactor(self,ax,kmax,
                              vmax,cmap):
        if not hasattr(self.distribution, 'Sq_averaged'):
            Sq = self.distribution.ReciprocalSpace(kmax=kmax)
        
        if (self.distribution.ndim==2):
            cax = ax[0].pcolormesh(self.distribution.Sq[0],self.distribution.Sq[1],self.distribution.Sq[2],vmax=vmax,cmap=cmap)
        else:
            print ('3D data. Plotting for qy=0 slice')
            cax = ax[0].pcolormesh(self.distribution.Sq[0][:,:,self.distribution.Sq[2].shape[2]//2],
                                   self.distribution.Sq[1][:,:,self.distribution.Sq[2].shape[2]//2],
                                   self.distribution.Sq[3][:,:,self.distribution.Sq[3].shape[2]//2],
                                   vmax=vmax,cmap=cmap)

        ax[0].figure.colorbar(cax,ax=ax[0])
        ax[0].set_title('Reciprocal Space, Structure Factor $S(\mathbf{q})$')
        ax[0].set_xlabel('$q_{x}$ ($\mu$m$^{-1}$)',fontsize=15)
        ax[0].set_ylabel('$q_{y}$ ($\mu$m$^{-1}$)',fontsize=15)
        ax[0].set_xlim(-kmax,kmax)
        ax[0].set_ylim(-kmax,kmax)

        ax[1].plot(self.distribution.Sq_averaged[:,0],self.distribution.Sq_averaged[:,1])
        ax[1].set_xlim(0,kmax)
        ax[1].set_xlabel('|q| ($\mu$m$^{-1}$)',fontsize=14)
        ax[1].set_ylabel('$\\tilde{S}(|\mathbf{q}|)$',fontsize=14)
        return ax
    
    def _plot_g2r(self,ax, dr, Rmax,vmax,cmap):
        #if not hasattr(self.distribution, 'Sq_averaged'):
        G2r = self.distribution.RealSpaceCorrelations(dr=dr)
        cax = ax[0].pcolormesh(self.distribution.G2r[0],self.distribution.G2r[1],self.distribution.G2r[2],vmax=vmax,cmap=cmap)
        ax[0].figure.colorbar(cax,ax=ax[0])
        ax[0].set_title('Pair correlation $G_{2}(\mathbf{r})$')
        ax[0].set_xlabel('$x_{x}$ ($\mu$m)',fontsize=15)
        ax[0].set_ylabel('$y_{x}$ ($\mu$m)',fontsize=15)
        ax[0].set_xlim(-Rmax,Rmax)
        ax[0].set_ylim(-Rmax,Rmax)

        ax[1].plot(self.distribution.G2r_averaged[:,0],self.distribution.G2r_averaged[:,1])
        ax[1].set_xlim(0,Rmax)
        ax[1].set_xlabel('|r| ($\mu$m)',fontsize=14)
        ax[1].set_ylabel('$G_{2}(|\mathbf{r}|)$',fontsize=14)
        return ax
    
    def _plot_ACF(self,ax, dr, Rmax,vmax,cmap):
        #if not hasattr(self.distribution, 'Sq_averaged'):
        ACF_ = self.distribution.Autocovariance()
        
        if (self.distribution.ndim==2):
            cax = ax[0].pcolormesh(self.distribution.ACF[0],self.distribution.ACF[1],self.distribution.ACF[2],vmax=vmax,cmap=cmap)
        else:
            print ('3D data. Plotting for ry=0 slice')
            cax = ax[0].pcolormesh(self.distribution.ACF[0][:,:,self.distribution.ACF[2].shape[2]//2],
                                   self.distribution.ACF[1][:,:,self.distribution.ACF[2].shape[2]//2],
                                   self.distribution.ACF[3][:,:,self.distribution.ACF[3].shape[2]//2],
                                   vmax=vmax,cmap=cmap)
        ax[0].figure.colorbar(cax,ax=ax[0])
        ax[0].set_title('ACF $(\mathbf{r})$')
        ax[0].set_xlabel('$x_{x}$ ($\mu$m)',fontsize=15)
        ax[0].set_ylabel('$y_{x}$ ($\mu$m)',fontsize=15)
        ax[0].set_xlim(-Rmax,Rmax)
        ax[0].set_ylim(-Rmax,Rmax)
        ACF_averaged = stats.angular_average(self.distribution.ACF,dr)
        ax[1].plot(ACF_averaged[:,0],ACF_averaged[:,1])
        ax[1].set_xlim(0,Rmax)
        ax[1].set_xlabel('|r| ($\mu$m)',fontsize=14)
        ax[1].set_ylabel('ACF $(|\mathbf{r}|)$',fontsize=14)
        return ax#self.fig, self.ax
    
    def plot_principal_ACF(self,ax=None,fontsize=13):
        """
        Plots the principal auto-correlation function values (along each perpendicular axes). 
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        ACF = self.distribution.compute_principal_ACF()
        for i in range(ACF.shape[0]):
            ax.plot(ACF[i][:,0],ACF[i][:,1],label='$C_{i}$'.format(i=i))
        ax.set_xlabel('R ($\mu$m)',fontsize=fontsize)
        ax.set_ylabel('ACF',fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        return ax

    @staticmethod
    def plot_colors(color_array, ax=None,transpose=True,extent=None):
        """
        Plot colors from given R,G,B values. Can be used for plotting:
            i) A single R,G,B (essentially color for a single scattering angle).
            ii) A set of R,G,B values for a set of scattering-angles (theta,phi). 
        TO OBTAIN THE DATASET FOR (ii), USE `colorize_spectra()` in `color_systems`.
        
        Parameters
        ----------
        color_array : ndarray
            1D data of size 1 x 3 or 3D data of size N_theta x N_phi x 3 (3 for 3 rgb channels)
        
        ax : matplotlib.axes, optional
            Matplotlib axes. Can be used to combine several plots. If not provided, creates a new figure.
        
        transpose : bool, optional
            Transposing of color_array. 
        

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        if (np.size(color_array) > 3):
            plot_array = np.transpose(color_array, (1, 2, 0)) if transpose is True else color_array
            ax.imshow(plot_array,extent=extent)
        else:
            ax.imshow([[np.float32(color_array)]])

    @staticmethod
    def plot_polar(I_array, wvl_0,wvl_f,ax=None,vmax=None,cmap=None):
        """
        Plot polar scattering data I(\theta,\lambda). Can be used for plotting:
            i) Angular-averaged Structure Factor as a function of wavelength and angle.
            ii) Angular-averaged form-factor (single object scattering) as a function of wavelength and angle.
            iii) Angle-resolved scattering from simulations or experiments
            iv) Colors (R,G,B) values as a function of wavelength and angle (irridescence)
        
        Parameters
        ----------
        I_array : ndarray
            2D scattering data of size N_wvl x N_angles or 3D color data of size N_wvl x N_angle x 3 (3 for 3 rgb channels)
        
        wvl_0 : float
            Starting wavelength (in microns)
        
        wvl_f : float
            Ending wavelength (in microns)
        
        ax : matplotlib.axes, optional
            Matplotlib axes. Can be used to combine several plots. If not provided, creates a new figure.
        
        vmax : float
            Maximum value for the heatmap
        """
        if ax is None:
            fig,ax=plt.subplots(figsize=(7,7),subplot_kw={'projection': 'polar'})
        _img = ax.imshow(I_array,origin='lower',extent=[0,np.pi/2,wvl_0,wvl_f],vmax=vmax,cmap=cmap)
        fig.colorbar(_img,ax=ax)
        ax.set_xlim(0,np.pi/2)
        ax.set_theta_zero_location('N')  # Set 0° (North) at the top
        ax.set_theta_direction(-1) 
        return ax