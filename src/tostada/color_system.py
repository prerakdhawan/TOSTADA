import numpy as np
from scipy.interpolate import interp1d
from importlib.resources import files
with files('tostada.util').joinpath('cie_color_matching_functions.txt').open('r') as f:
    cmf= np.loadtxt(f)

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array([x, y, 1-x-y])

class ColourSystem:
    """
    Colorize a spectrum using CIE standard illuminations and color references.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    """

    def __init__(self, red, green, blue, white):
        """

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]/3
        self.cmf = cmf
    def gamma_correct(self, rgb):
        """Apply sRGB gamma correction to linear RGB values."""
        rgb = np.clip(rgb, 0.0, 1.0)  # Ensure values are in [0,1]
        a = 0.055
        threshold = 0.0031308
        corrected = np.where(rgb <= threshold,
                            12.92 * rgb,
                            (1 + a) * np.power(rgb, 1/2.4) - a)
        return corrected
    
    def colorize_spectra(self,wavelength,angular_spectrum,gamma_correction=True): 
        """
        Convert an angular-spectrum (I(qx,qy,lambda)) to gamma-corrected R,G,B values at each (qx,qy) pair. 

        Parameters
        ----------
        wavelength : [1 x N_wvl] array 
            Wavelength array in microns
        angular_spectrum : N_wvl x Nx x Ny array
            Angular spectrum as a function of qx,qy. If a 1D array, assumes a fixed angle (qx=qx0,qy=qy0).
        gamma_correction : bool
            Apply gamma-correction to the r,g,b array.

        Returns
        -------
        Color_array : [3 x Nx x Ny] or [3 x 1] array
            R,G,B values for each (qx,qy) pair.

        """
        if (angular_spectrum.ndim == 3):
            I_array = angular_spectrum
            Color_array = np.zeros([3,I_array.shape[1],I_array.shape[2]])
            for i in range(I_array.shape[1]):
                for j in range(I_array.shape[2]):
                    spectrum = np.c_[wavelength,np.clip(I_array[:,i,j],0,1)] #normalizing by maximum for each wavelength
                    rgb = self.spec_to_rgb(spectrum,gamma_correction=gamma_correction)
                    Color_array[:,i,j] = rgb
        elif (angular_spectrum.ndim == 2):
            #Color_array = np.zeros([3,angular_spectrum.shape[0],angular_spectrum.shape[1]])
            Color_array = np.zeros([angular_spectrum.shape[0],angular_spectrum.shape[1],3] )
            max_vals_ars = np.max(angular_spectrum,axis=1)
            for i in range(angular_spectrum.shape[1]):
                spectrum = np.c_[wavelength,angular_spectrum[:,i]/np.max(angular_spectrum)]
                rgb = cs_srgb.spec_to_rgb(spectrum,gamma_correction=gamma_correction)
                Color_array[:,i,:] = rgb#a
        else:
            Color_array = self.spec_to_rgb(np.c_[wavelength,angular_spectrum],gamma_correction=gamma_correction)
        return Color_array

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """
        Transformation from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb
    
    def xyz_to_rgb_with_luminance(self, xyz, out_fmt=None):
        """
        Transform from xyz to rgb with preserved luminance.
        """
        # Extract luminance (Y) before normalization
        luminance = xyz[1]  # Y component contains luminance
        
        # Handle the case where luminance is zero (black)
        if luminance < 1e-6:
            if out_fmt == 'html':
                return "#00000084"
            return np.zeros(3)
        
        # Calculate chromaticity coordinates
        sum_xyz = np.sum(xyz)
        if sum_xyz > 0:
            chromaticity_xyz = xyz / sum_xyz
        else:
            chromaticity_xyz = xyz
        
        # Convert chromaticity to RGB (this will give us the color without brightness)
        rgb_chromaticity = self.T.dot(chromaticity_xyz)
        
        # Handle out-of-gamut colors
        if np.any(rgb_chromaticity < 0):
            # Desaturate until in gamut
            w = -np.min(rgb_chromaticity)
            rgb_chromaticity += w
        
        if not np.all(rgb_chromaticity == 0):
            # Normalize the chromaticity
            rgb_chromaticity /= np.max(rgb_chromaticity)
        
        # Apply luminance scaling (mapping from your luminance range to [0,1])
        # You may need to adjust this based on your luminance range
        luminance_scale = min(luminance / self.white[1], 1.0)
        
        # Scale RGB by luminance
        rgb = rgb_chromaticity * luminance_scale
        rgb = self.gamma_correct(rgb)
        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb
    
    def rgb_to_hex(self, rgb):
        """
        Convert from fractional rgb values to HTML-style hex string.
        """

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """
        -------------- Redundant ---------------
        Convert a spectrum to xyz point. 
        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """
        spec_int = interp1d(spec[:,0],spec[:,1])
        #_xint = interp1d(self.cmf[:,0],self.cmf[:,1])
        #_yint = interp1d(self.cmf[:,0],self.cmf[:,2])
        #_zint = interp1d(self.cmf[:,0],self.cmf[:,3])
        #x0 = _xint(spec[:,0])
        #y0 = _yint(spec[:,0])
        #z0 = _zint(spec[:,0])
        #cmf0 = np.c_[x0,y0,z0]
        spectrum = spec_int(self.cmf[:,0])
        XYZ = np.sum(spectrum[:,np.newaxis]*self.cmf[:,1:],axis=0 )
        #XYZ = np.sum(spec[:,1][:,np.newaxis]*cmf0,axis=0 )
        #XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        #print(XYZ.shape,den.shape)
        return XYZ
        #if den == 0.:
        #    return XYZ
        #return XYZ / den

    def spectrum_to_XYZ(self,spectrum,illumination=None,microns=True):
        """
        Convert a spectrum to the tristimulus values XYZ.
        The spectrum is interpolated on the same grid as the color matching functions.
        
        Parameters
        ----------

        spectrum : [Nx2] Ndarray
            Spectrum with zeroth column as the wavelength and first column as the spectra. 
            NOTE : Wavelength must be in microns by default

        Illumination : [Nx2] Ndarray
            Illumination of the source with zeroth column as the wavelength and first column as the spectra.
        
        microns : Bool
            Whether the wavelength is in microns or nanometers.

        Returns
        -------
        XYZ : [1x3] Ndarray
            Tristimulus values
        """
        if (microns):
            factor=1e3 
        else:
            factor=1

        spec_int = interp1d(spectrum[:,0],spectrum[:,1])
        spec = spec_int(self.cmf[:,0]/factor)
        if (illumination is None):
            illum = np.ones_like(spec)
        else:
            illumination_int = interp1d(illumination[:,0],illumination[:,1])
            illum = illumination_int(self.cmf[:,0])
        
        XYZ = np.sum(spec[:,np.newaxis]*illum[:,np.newaxis]*self.cmf[:,1:]*5/factor,axis=0)
        N = np.sum(self.cmf[:,2]*illum*5/factor,axis=0) #for normalization of XYZ
        return XYZ/N

    def XYZ_to_xyz(self,XYZ):
        """
        Convert the tristimulus values (XYZ) to chromaticity coordinates (xyz).
        """
        denom = np.sum(XYZ)
        return XYZ/denom

    def spec_to_rgb(self, spec, illumination=None,
                    out_fmt=None,microns=True, gamma_correction=True):
        """
        
        Convert a spectrum to an rgb value. The resulting values can be visualized into a color using `tostada.Visualize` module.

        """

        #xyz = self.spec_to_xyz(spec)
        XYZ = self.spectrum_to_XYZ(spec,illumination,microns=microns)
        rgb = self.XYZ_to_RGB(XYZ,gamma_correction=gamma_correction)
        return rgb #self.gamma_correct(self.T@XYZ)
        #return self.xyz_to_rgb_with_luminance(xyz, out_fmt)
    
    def XYZ_to_RGB(self,XYZ,gamma_correction=False):
        """
        Converts the tristimulus values to rgb values. To obtain the XYZ from a spectrum, use `spectrum_to_XYZ`.
        """
        rgb = self.T @ XYZ #rgb without gamma-correction values
        if (gamma_correction==True):
            rgb = self.gamma_correct(rgb) #RGB values
        return rgb

        
illuminant_D65 = xyz_from_xy(0.3127, 0.3291)

cs_hdtv = ColourSystem(red=xyz_from_xy(0.67, 0.33),
                       green=xyz_from_xy(0.21, 0.71),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

cs_smpte = ColourSystem(red=xyz_from_xy(0.63, 0.34),
                        green=xyz_from_xy(0.31, 0.595),
                        blue=xyz_from_xy(0.155, 0.070),
                        white=illuminant_D65)

cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)
