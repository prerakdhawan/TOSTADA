import cv2
import autograd.numpy as np
import matplotlib.pyplot as plt
import tostada.Statistics as stats
import pickle

class PhaseDistribution:
    """
    Create a phase distribution for a given image in 2D/3D. The phase distribution is assume to be between 0 to 1. 
    Once a resolution (in microns) is defined, the overall size along each axes is automatically computed along with its spatial fourier frequencies.
    This class can be used to access various real-space and reciprocal-space properties of the phase distribution.
    """

    def __init__(self, image,resolution=1):
        self._image = image
        #self.BoxSize = image.shape
        self._resolution = resolution
        self.ndim = image.ndim
        self._compute_fft_axes()
        self.volumefraction = np.mean(image)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image):
        self._image = new_image
        self._compute_fft_axes()
    
    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._resolution = value
        self._compute_fft_axes()
    @property
    def Lx(self):
        return self.BoxSize[1]*self.resolution

    @property
    def Ly(self):
        return self.BoxSize[0]*self.resolution
    
    @property
    def Lz(self):
        if self.ndim < 3:
            return None
        return self.BoxSize[2] * self.resolution
    
    @property
    def BoxSize(self):
        return self.image.shape
    
    def _compute_fft_axes(self):
        # Frequencies depend on shape and resolution
        nx, ny = self.image.shape[:2]
        self.fftx = (2 * np.pi / self.Lx) * np.fft.fftshift(np.fft.fftfreq(nx, d=1/nx))
        self.ffty = (2 * np.pi / self.Ly) * np.fft.fftshift(np.fft.fftfreq(ny, d=1/ny))

        if self.ndim == 3:
            nz = self.image.shape[2]
            self.fftz = (2 * np.pi / self.Lz) * np.fft.fftshift(np.fft.fftfreq(nz, d=1/nz))
        else:
            self.fftz = None

    def save(self, filename):
        """
        Save the all the properties of the PhaseDistribution in a pickled state. Similar to `save` method in PointDistribution.

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
        Load an instance of the saved PhaseDistribution created from PointDistribution.save(). Similar to `save` method in PointDistribution.

        Parameters
        ----------
        filename : str
            Name of the file to be loaded. 
        """
        with open(filename+'.pkl', 'rb') as f:
            A = pickle.load(f)
        print ('File loaded with filename {f}'.format(f=filename+'.pkl'))
        return A 
    
    def compute_principal_ACF(self):
        """
        Computes the autocorrelation along x,y,z axes. 
        """
        if not hasattr(self, 'ACF'):
            C = self.Autocovariance()
        if self.ndim==3:
            czz = self.ACF[-1][self.BoxSize[0]//2,self.BoxSize[1]//2,self.BoxSize[2]//2 : ]
            cxx = self.ACF[-1][self.BoxSize[0]//2:,self.BoxSize[1]//2,self.BoxSize[2]//2  ]
            cyy = self.ACF[-1][self.BoxSize[0]//2,self.BoxSize[1]//2:,self.BoxSize[2]//2  ]
            hxx=np.arange(cxx.shape[0])*self.resolution
            hyy=np.arange(cyy.shape[0])*self.resolution
            hzz=np.arange(czz.shape[0])*self.resolution
            return np.asarray([np.c_[hxx,cxx],np.c_[hyy,cyy],np.c_[hzz,czz]])
        else:
            cxx = self.ACF[-1][self.BoxSize[0]//2,self.BoxSize[1]//2 : ]
            cyy = self.ACF[-1][self.BoxSize[0]//2:,self.BoxSize[1]//2  ]
            hxx=np.arange(cxx.shape[0])*self.resolution
            hyy=np.arange(cyy.shape[0])*self.resolution
            return np.asarray([np.c_[hxx,cxx],np.c_[hyy,cyy]])
    
    def invert_phases(self):
        """
        Converts pores to inclusions and vice versa.
        """
        self.image = np.logical_not(self.image)
        return None

    def ReciprocalSpace(self):
        """
        Computes the reciprocal space response of the given phase distribution. Identical to ReciprocalSpace in PointDistribution. 
        First object is the SpectralDensity (2D or 3D) along with it's reciprocal space vectors.
        Second object is the angular-averaged of the SpectralDensity computed in Statistics.py.
        """
        self.Xq = self.Spectraldensity()
        self.Xq_averaged = stats.angular_average(self.Xq,dkx=2*np.pi/self.Lx)
        return self.Xq,self.Xq_averaged
    
    def get_resolution(self,scale_sem=1,test=False,binarized=True,vmin=100,vmax=160): 
        """
        Extract scale-bar from an SEM image. Once extracted, self.resolution is re-scaled accordingly. 
        ! Caution: Still needs to be tested on more images.
        scale_sem : length provided in the SEM image (convert to microns)
        """
        if (binarized==False):
            contrast_enhanced_image = cv2.equalizeHist(self.image)
            # Apply a fixed threshold
            _, self.image = cv2.threshold(contrast_enhanced_image, vmin, vmax, cv2.THRESH_BINARY)

        template_image_path = 'SEM_scalebar.PNG'
        template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        contrast_enhanced_template = cv2.equalizeHist(template_image)
        _, binary_template_image = cv2.threshold(contrast_enhanced_template, 100, 160, cv2.THRESH_BINARY)
        # Perform template matching
        result = cv2.matchTemplate(self.image, binary_template_image, cv2.TM_CCOEFF_NORMED)
        # Get the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Get the size of the template
        template_height, template_width = template_image.shape
        # Draw a bounding box around the detected scale bar
        top_left = max_loc
        #detected_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
#        edges = cv2.Canny(self.image[top_left[1]:top_left[1]+template_height,top_left[0]:top_left[0]+template_width],50,150,apertureSize=3) 
        edges = cv2.Canny(self.image[-70:-10,10:200],50,150,apertureSize=3) #only views bottom-left of the image
        # Apply HoughLinesP method to 
        # to directly obtain line end points

        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=20, # Min number of votes for valid line
                    minLineLength=7, # Min allowed length of line
                    maxLineGap=10 # Max allowed gap between line for joining them
                    )
        for line in lines:
            _line = line.ravel()
            dx = np.abs(_line[0]-_line[2])
            dy = np.abs(_line[1]-_line[3])
            if (dy==0):
                template_width=dx+2
                #print (dx)
                x1,y1,x2,y2 = _line
        if (test==True):
            plt.imshow(self.image)
            buffer=20
            #plt.plot([top_left[0]+x1,top_left[0]+x2],[self.BoxSize[0]-y1,self.BoxSize[0]-y2],color='tab:red',linewidth=2)
            plt.plot([10+x1,10+x2],[self.BoxSize[0]-y1+buffer,self.BoxSize[0]-y2+buffer],color='tab:red',linewidth=2)
            plt.title('Scalebar length = {m} pixels'.format(m=template_width))
            print ('topleft=',top_left)

        self.resolution = scale_sem/template_width

    
    def Spectraldensity(self):
        """
        Spectral Density X(q) for the given 2D or 3D image along with its spatial frequencies.

        Returns
        -------
        Xqdata : Ndarray
            Xqdata[0],Xqdata[1] ... are spatial frequencies and Xqdata[-1] is the spectral density
        """
        xq = np.fft.fftshift(np.fft.fftn(self.image - self.volumefraction )) 
        xq = np.abs(xq)**2 /np.size(self.image)
        if (self.ndim==3):
            [_fftx,_ffty,_fftz] = np.meshgrid(self.ffty,self.fftx,self.fftz)
            data = np.asarray([_fftx,_ffty,_fftz])
        else:
            [_fftx,_ffty] = np.meshgrid(self.ffty,self.fftx)
            print (_fftx.shape,_ffty.shape)
            data = np.asarray([_fftx,_ffty])
        Xqdata = np.concatenate((data, xq[None,:,:]), axis=0) 
        return Xqdata

    def Autocovariance(self):
        """
        Computes the autocovariance function for the given 2D or 3D image along with real-space distance.
        
        Returns
        -------
        ACFdata (numpy.Ndarray): ACFdata[0],ACFdata[1] ... are distances and ACFdata[-1] is the autocovariance
        """
        
        F = np.fft.fftn(self.image)/np.size(self.image)
        power_spectrum = np.abs(F) ** 2
        autocov = np.fft.ifftn(power_spectrum).real*np.size(self.image) 
        autocov = np.fft.fftshift(autocov)     # Center the result
        if (self.ndim==3):
            [X,Y,Z] = np.meshgrid(np.linspace(-self.Ly/2,self.Ly/2,self.BoxSize[1]),np.linspace(-self.Lx/2,self.Lx/2,self.BoxSize[0]),np.linspace(-self.Lz/2,self.Lz/2,self.BoxSize[2]))
            data = np.asarray([X,Y,Z])
        else:
            [X,Y] = np.meshgrid(np.linspace(-self.Ly/2,self.Ly/2,self.BoxSize[0]),np.linspace(-self.Lx/2,self.Lx/2,self.BoxSize[1]))
            data = np.asarray([X,Y])
        ACFdata = np.concatenate((data, autocov[None,:,:]), axis=0)
        self.ACF = ACFdata
        return self.ACF

    def centerofmass_pores(self,dr=0.02):
            #contours,_=cv2.findContours(binary_image[100:500,200:600], cv2.RETR_TREE, 
        #                            cv2.CHAIN_APPROX_SIMPLE) # Detecting shapes in image by selecting region with same colors or intensity. 
        contours,_=cv2.findContours(self.image, cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE) # Detecting shapes in image by selecting region with same colors or intensity. 
        centers = []

        for contour in contours:
            # Approximate contour to polygon
            epsilon = dr * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get the moments to calculate the center
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
            else:
                cX, cY = 0, 0
        return centers

    def tesselate(self,copies=1):
        """
        Tesselate N copies of the image in 2D/3D. For periodicity. Similar to tesselated in PointDistribution.
        
        Returns
        -------
        tesselated (numpy.Ndarray): (2*N+1)Boxsize[0] x (2*N+1)Boxsize[1] array if 2D else (2*N+1)Boxsize[0] x (2*N+1)Boxsize[1] x (2*N+1)Boxsize[2].
        """
        ranges = [np.arange(-copies, copies + 1) for _ in range(self.ndim)]
        grid = np.array(np.meshgrid(*ranges))
        tesselated = np.kron(np.ones_like(grid[0]),self.image)
        return tesselated
    
    def Hyperuniformity_data(self):
        """
        Spectral width (FWHM) and Hyperuniformity index of the structures. See Statistics.py for further details.
        """
        if not hasattr(self, 'Xq_averaged'):
            Xq = self.ReciprocalSpace()
        Hdata = stats.fwhm_and_H(self.Xq_averaged)
        print ('Spectral Width of the structural peak={p}'.format(p=Hdata[0]))
        print ('Hyperuniformity of the structure = {h}'.format(h=Hdata[1]))
        return Hdata
    
    def Dmean_from_q(self,factor=1,kmax=100):
        """
        Calculate mean center-to-center distance of objects from Spectral density X(q).
        """
        if not hasattr(self, 'Xq_averaged'):
            Xq = self.ReciprocalSpace()
        Dmean = stats.dmean_from_qpeak(self.Xq_averaged,factor=factor)
        print ('Mean center-to-center distance = {d}'.format(d=Dmean))
        return np.float32(Dmean)
    
    def SphericalContactDistribution(self):
        """
        Computes Spherical Contact Distribution for a porous object (2D or 3D). See Statistics.py for further details.
        """
        scd = stats.SphericalContactDistribution(self.image,self.resolution)
        return scd
 