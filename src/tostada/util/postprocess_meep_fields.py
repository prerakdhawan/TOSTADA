import h5py
import numpy as np
from scipy.stats import binned_statistic
from scipy.ndimage import zoom
class Postprocess_fields:
    def __init__(self, file, is_3D=True):
        """
        Post-processes the meep DFT fields for diffraction efficiencies. 
        Uses electric field components for source probe and detector probe (either reflected or transmitted).
    
        Parameters:
        file:
            File containing output from meep simulation
        wavelength: 
            Wavelength array in microns
        resolution:
            Meep resolution in 1/microns
        """
        self.file = file
        self.wavelength = file.attrs['wvl']
        #self.resolution = file.attrs['res']
        #self.is_3D = int(file['fields/source/ex'].ndim - 1)==2 #-1 because fields are shaped as [wvl,Nx,Ny] for 3d and [wvl,Nx] for 2d
        self.Lx = file.attrs['Lx']
        self.Ly = file.attrs['Ly']
        if (self.Ly is None):
            self.is_3D = False
        else:
            self.is_3D = True

        self.fields_shape = file['fields/source/ex'].shape[1:]
        self.resolution = np.mean([self.fields_shape[0]/self.Lx,self.fields_shape[1]/self.Ly])
        self.k_vac = 2 * np.pi / (self.wavelength * self.resolution)
        self.omega = (2 * np.pi * 3e8) / (self.wavelength * 1e-6) #only necessary for absorption
        
        self.kparallel = self.create_kparallel()
        self.source = self.file['fields/source']
        self.backscattered = self.file['fields/backscattered']
        self.forwardscattered = self.file['fields/forwardscattered']
        self.total_reflectance = file.attrs['Reflectance']
        self.total_transmittance = file.attrs['Transmittance']
        self.T0 = file.attrs['Source_flux']
        
        #self.z_pos = file.attrs['z_pos']

    def create_kparallel(self):
        """
        Create the parallel k-vectors in 2D/3D permitted for the given simulation domain.
        """
        if (self.is_3D):
            self.Nx,self.Ny = self.fields_shape
            #self.Lx,self.Ly = self.Nx/self.resolution,self.Ny/self.resolution
            kxf = np.arange(-self.Nx // 2, self.Nx // 2) * 2 * np.pi / (self.Nx)
            kyf = np.arange(-self.Ny // 2, self.Ny // 2) * 2 * np.pi / (self.Ny)
            kparallel = np.meshgrid(kxf, kyf)
        else:
            self.Nx = self.fields_shape[1]
            self.Lx = self.Nx/self.resolution
            kxf = np.arange(-self.Nx // 2, self.Nx // 2) * 2 * np.pi / (self.Nx)
            kparallel = kxf
        return kparallel

    def get_angles(self,k,kz=None):
        """
        Angles decided by the permitted diffraction orders for the problem.
        """
        if (self.is_3D):
            #kparallel = self.kparallel if np.logical_and(kx)
            if (kz is None):
                kz = np.lib.scimath.sqrt(k**2 - np.sum(np.square(self.kparallel),axis=0))
            phi = np.arctan2(self.kparallel[1],self.kparallel[0])
            phi = np.where((phi < 0), 2 * np.pi + phi, phi)#.T
            theta = np.arccos(np.real(kz/k))
            angles = [theta,phi]
        else:
            kz = np.lib.scimath.sqrt( k ** 2 - np.square(self.kparallel))
            theta = np.arccos(np.real(kz / k ))
            angles = [theta]
        return angles
    
    
    def helicity_basis_vectors(self,theta, phi, conj=True):
        """
        Returns the unit vectors of helicity basis to transform the fields.
        """
        sx = 1j * np.sin(phi)
        sy = -1j * np.cos(phi)
        if conj:
            sx = -sx
            sy = -sy
        px = -np.cos(theta) * np.cos(phi)
        py = -np.cos(theta) * np.sin(phi)
        pz = np.sin(theta)

        epx = sx + px
        epy = sy + py
        epz = pz

        enx = sx - px
        eny = sy - py

        return (
            epx / np.sqrt(2),
            epy / np.sqrt(2),
            epz / np.sqrt(2),
            enx / np.sqrt(2),
            eny / np.sqrt(2),
        )
    
    def theta_mask(self,theta_array,theta_lim):
        """
        Mask for extracting diffraction efficiencies for particular angles.
        theta_lim : list
            list of upper and lower limit for angles *in radians*
        """
        theta_filtered = theta_array
        theta_mask = np.where((theta_filtered >= theta_lim[0]) & (theta_filtered <= theta_lim[1]), 1, 0)
        return theta_mask

    def Helicitytransform(self,ex,ey,ez,theta,phi,conj=True):
        epx, epy, epz, enx, eny = self.helicity_basis_vectors(theta, phi, conj)
        E_plus = epx * ex + epy * ey + epz * ez  # change to helicity basis
        E_minus = enx * ex + eny * ey - epz * ez
        return E_plus,E_minus
    

    def backscattered_efficiencies(self,n_in,n_out,theta_in = 0,pol=0,helicity=False,theta_range=[0,0.1]):
        """
        Returns the back-scattered diffraction efficiencies in specular,diffused,helicity decomposed components for N wavelengths.
        Inputs:
            n_in : float or nd.array
                Refractive index of the source medium. Can be constant or a dispersive media
            n_out : float or nd.array
                Refractive index of the detector medium. Can be constant or a dispersive media
            theta_in : float
                Angle of incidence
            pol : int
                Polarization of the source. If helicity=True, these are left- and right-handed modes. If helicity=False, these are TE and TM modes.
            theta_range   : list : 
                Minimum and maximum angle range of angle for which efficiency needs to be summed.
        
        Returns:
            spec : array of size N x 1 : specular reflectance
            diff : array of size N x 1 : diffused reflectance
            refl : array of size N x 1 : total reflectance (sum of specular and diffused [also equals sum of both helicity components])
            Rp   : array of size N x 1 : Reflected power in positive helicity component
            Rm   : array of size N x 1 : Reflected power in negative helicity component
        """
        ex_src = np.asarray(self.source['ex']) 
        ey_src = np.asarray(self.source['ey'])
        ez_src = np.asarray(self.source['ez'])
        
        ex_b = np.asarray(self.backscattered['ex']) 
        ey_b = np.asarray(self.backscattered['ey'])
        ez_b = np.asarray(self.backscattered['ez'])
        
        if (np.isscalar(n_in)):
            n_in = np.ones_like(self.wavelength)*n_in
        if (np.isscalar(n_out)):
            n_out = np.ones_like(self.wavelength)*n_out

        if (helicity==True):
            E_inc = - -1 / np.sqrt(2) * (ex_src (-1)**(pol)* 1j * ey_src)
        else:
            E_inc = (pol)*ey_src + (1-pol)*ex_src + ey_src + ez_src
        # Subtract incoming fields from the total fields to get only scattered fields
        Exr = ex_b 
        Eyr = ey_b 
        Ezr = ez_b
        rk_space=[]
        thetamask=[]
        theta_array=[]
        spec = np.zeros(len(self.wavelength))
        diff = np.zeros_like(spec)
        refl = np.zeros_like(spec)
        Rp = np.zeros_like(spec)
        Rm = np.zeros_like(spec)
        R_smallangle=np.zeros_like(spec)
        ARS_fdtd=[]
        Y,X = np.meshgrid(np.linspace(-self.Nx/2,self.Nx/2,self.Nx),np.linspace(-self.Ny/2,self.Ny/2,self.Ny))
        for i in range(len(self.wavelength)):

            kz_in = n_in[i]*self.k_vac[i]*np.cos(theta_in) 
            kx_in = n_in[i]*self.k_vac[i]*np.sin(theta_in) 
            ky_in = 0
            phase_term = np.exp(1j * kx_in * (X)  ) 
            Exr_ = (ex_b[i] - ex_src[i]) / phase_term
            Eyr_ = (ey_b[i] - ey_src[i]) / phase_term
            Ezr_ = (ez_b[i] - ez_src[i]) / phase_term

            KX= 1*(n_in[i]*self.k_vac[i]*np.sin(theta_in))+ self.kparallel[1]
            KY = self.kparallel[0]
            kz_out = np.lib.scimath.sqrt((n_out[i]*self.k_vac[i])**2  - KX**2 - KY**2 )
            kz_in = n_in[i]*self.k_vac[i]*np.cos(theta_in) 
            theta, phi = self.get_angles(n_out[i]*self.k_vac[i]*np.cos(theta_in))
            ####### transform the fields for each wavelength to reciprocal space (spatial frequency) for computing power in each diffraction mode

            ex_ = np.fft.fftshift(np.fft.fft2(Exr_)) / Exr[0].size
            ey_ = np.fft.fftshift(np.fft.fft2(Eyr_)) / Eyr[0].size
            ez_ = np.fft.fftshift(np.fft.fft2(Ezr_)) / Ezr[0].size

            exs_ = np.fft.fftshift(np.fft.fft2(ex_src[i]/phase_term)) / Exr[0].size
            eys_ = np.fft.fftshift(np.fft.fft2(ey_src[i]/phase_term)) / Exr[0].size
            ezs_ = np.fft.fftshift(np.fft.fft2(ez_src[i]/phase_term)) / Exr[0].size

            E_plus,E_minus = self.Helicitytransform(ex_,ey_,ez_,theta,phi)#,conj=False)
            E_plus_in,E_minus_in = self.Helicitytransform(exs_,eys_,ezs_,theta,phi)#,conj=False)
            
            kz_ratio = np.nan_to_num(np.real(kz_out)/np.real(kz_in))
            
            Rplus = np.abs(E_plus) ** 2 * kz_ratio #np.cos(theta)
            Rminus = np.abs(E_minus) ** 2 * kz_ratio#np.cos(theta)
            
            Rp[i] = np.sum(Rplus)  / np.max(np.abs(E_plus_in)) ** 2
            Rm[i] = np.sum(Rminus) / np.max(np.abs(E_minus_in)) ** 2
            
            #kz_ratio = np.nan_to_num(np.real(kz_out)/np.real(kz_in))
            spec_ind = np.where(kz_ratio==np.max(kz_ratio))

            p_num = (
                        np.sum(np.abs(ex_) ** 2 * np.real(kz_out) )
                        + np.sum(np.abs(ey_) ** 2 * np.real(kz_out) )
                        - (np.sum(np.real(np.conj(ez_)*(ex_ * (KX) + ey_ * KY )))  ) 
                        #+ np.sum(np.abs(ez_) ** 2 * kz_ratio )
            )
            p_denom = (
                        np.sum(np.abs(exs_) ** 2 * kz_in )
                        + np.sum(np.abs(eys_) ** 2 * kz_in )
                        - (np.sum(np.real(np.conj(ezs_)*(exs_ * (-kx_in) + eys_ * ky_in )))  ) 
                        #+ np.sum(np.abs(ez_) ** 2 * kz_ratio )
            )

            p_refl = np.nan_to_num(p_num / p_denom)
            #p_refl = p_refl / (np.max(np.abs(ex_src[i])) ** 2 + (np.max(np.abs(ez_src[i])) ** 2))
            
            refl[i] = p_refl

            r_kspace = (
                        (np.abs(ex_) ** 2 * np.real(kz_out) )
                        + (np.abs(ey_) ** 2 * np.real(kz_out) )
                        - ((np.real(np.conj(ez_)*(ex_ * (KX) + ey_ * KY )))  ) 


            ) / p_denom #np.max(np.abs(E_inc[i]  )) ** 2

            r_kspace_ = zoom(r_kspace,2,order=1)
            theta_ = zoom(theta,2,order=1)
            theta_smallangle = self.theta_mask(theta_,theta_range)

            rk_space.append(r_kspace)
            
            thetamask.append(theta_smallangle)
            theta_array.append(theta)
            r_spec = np.sum(np.where(r_kspace < np.max(r_kspace),0,r_kspace))
            spec[i] = r_spec
            diff[i] = p_refl - r_spec
            R_smallangle[i] = np.sum(r_kspace_*theta_smallangle) 
            theta_flat = theta_.flatten()
            Rk_flat = r_kspace_.flatten()
            theta_bins = np.linspace(0, np.pi/2, 60)
            R_theta, _, _ = binned_statistic(theta_flat, Rk_flat, statistic='mean', bins=theta_bins)
            ARS_fdtd.append(np.c_[theta_bins[:-1],np.nan_to_num(R_theta)])
            #ARS_fdtd.append(kz_ratio)
            #ARS_fdtd.append(Rplus)
        return spec, diff, refl, Rp, Rm,R_smallangle,np.asarray(rk_space),np.asarray(ARS_fdtd),np.asarray(theta_array)


    def transmitted_efficiencies(self,n_in,n_out,theta_in = 0,pol=0,helicity=False,theta_range=[0,0.1]):
        """
        Returns the transmitted diffraction in specular,diffused,helicity decomposed components for N wavelengths.
        Also extrapolates absorbance in a finite Absorber of thickness 'th' micrometers.
        Inputs:
            n_in : float or nd.array
                Refractive index of the source medium. Can be constant or a dispersive media
            n_out : float or nd.array
                Refractive index of the detector medium. Can be constant or a dispersive media
            theta_in : float
                Angle of incidence
            pol : int
                Polarization of the source. If helicity=True, these are left- and right- handed modes. If helicity=False, these are TE and TM modes.
            th   : scalar : 
                thickness (in micrometer) of the cSi for which the absorbance is needed.
        
        Returns:
            spec : array of size N x 1 : specular reflectance
            diff : array of size N x 1 : diffused reflectance
            refl : array of size N x 1 : total reflectance (sum of specular and diffused [also equals sum of both helicity components])
            Rp   : array of size N x 1 : Reflected power in positive helicity component
            Rm   : array of size N x 1 : Reflected power in negative helicity component
            Abs  : array of size N x 1 : Absorbance calculated from transmitted diffraction orders for a finite cSi thickness.
        """
        #n_glass = np.loadtxt('glass_substrate_fdtd_fit.txt')
        ex_src = np.asarray(self.source['ex']) 
        ey_src = np.asarray(self.source['ey'])
        ez_src = np.asarray(self.source['ez'])
        
        ex_f = np.asarray(self.forwardscattered["ex"])
        ey_f = np.asarray(self.forwardscattered["ey"])
        ez_f = np.asarray(self.forwardscattered["ez"])

        if (np.isscalar(n_in)):
            n_in = np.ones_like(self.wavelength)*n_in
        if (np.isscalar(n_out)):
            n_out = np.ones_like(self.wavelength)*n_out

        if (helicity==True):
            E_inc = - -1 / np.sqrt(2) * (ex_src + (-1)**(pol)* 1j * ey_src)
        else:
            E_inc = (pol)*ey_src + (1-pol)*ex_src #+ ez_src
        # Subtract incoming fields from the total fields to get only scattered fields
        Ext = ex_f
        Eyt = ey_f
        Ezt = ez_f
        tk_space=[]
        thetamask=[]
        theta_array=[]
        spec = np.zeros(len(self.wavelength))
        diff = np.zeros_like(spec)
        transmitted = np.zeros_like(spec)
        trn = np.zeros_like(spec)
        Tp = np.zeros_like(spec)
        Tm = np.zeros_like(spec)
        T_smallangle=np.zeros_like(spec)
        ARS_fdtd=[]
        for i in range(len(self.wavelength)):
            #kz_out = np.lib.scimath.sqrt((n_out*self.k_vac[i])**2 - self.kx**2 - self.ky**2)
            kz_out = np.lib.scimath.sqrt((n_out[i]*self.k_vac[i])**2 - np.sum(np.square(self.kparallel),axis=0))
            kz_in = n_in[i]*self.k_vac[i]*np.cos(theta_in) 
            theta, phi = self.get_angles(n_out[i]*self.k_vac[i])

            ex_ = np.fft.fftshift(np.fft.fft2(Ext[i])) / Ext[0].size
            ey_ = np.fft.fftshift(np.fft.fft2(Eyt[i])) / Eyt[0].size
            ez_ = np.fft.fftshift(np.fft.fft2(Ezt[i])) / Ezt[0].size

            #phase = np.exp(1j*k_inc[i]*1e3*np.cos(theta))/1e3
            E_plus,E_minus = self.Helicitytransform(ex_,ey_,ez_,theta,phi,conj=False)

            Rplus = np.abs(E_plus) ** 2 * np.cos(theta)
            Rminus = np.abs(E_minus) ** 2 * np.cos(theta)

            Tp[i] = np.sum(Rplus) / np.max(np.abs(E_inc[i])) ** 2
            Tm[i] = np.sum(Rminus) / np.max(np.abs(E_inc[i])) ** 2

            kz_ratio = np.nan_to_num(np.real(kz_out)/np.real(kz_in))
            spec_ind = np.where(kz_ratio==np.max(kz_ratio))

            p_trn = (np.sum(np.abs(ex_) ** 2 * kz_ratio )
                        + np.sum(np.abs(ey_) ** 2 * kz_ratio )
                        + np.sum(np.abs(ez_) ** 2 * kz_ratio )
                    ) 

            p_trn = p_trn / np.max(np.abs(E_inc[i])) ** 2

            transmitted[i] = p_trn

            t_kspace = (
                        (np.abs(ex_) ** 2 * kz_ratio )
                        + (np.abs(ey_) ** 2 * kz_ratio )
                        + (np.abs(ez_) ** 2 * kz_ratio )
            ) / np.max(np.abs(E_inc[i])) ** 2
            
            t_kspace_ = zoom(t_kspace,2,order=1)
            theta_ = zoom(theta,2,order=1)
            theta_smallangle = self.theta_mask(theta_,theta_range)

            tk_space.append(t_kspace)
            thetamask.append(theta_smallangle)
            theta_array.append(theta)
            spec[i] = np.sum(t_kspace[spec_ind[0][0],spec_ind[1][0]])
            diff[i] = np.sum(t_kspace) - spec[i]  # scalar
            t_masked = t_kspace_*theta_smallangle
            #r_nonzero = r_masked[r_masked>0]       
            T_smallangle[i] = np.sum(t_masked) 
            #if r_nonzero.size > 0:
            #    R_smallangle[i] = np.sum(r_nonzero)
            #else:
            #    R_smallangle[i] = 0
            theta_flat = theta_.flatten()
            Tk_flat = t_kspace_.flatten()
            theta_bins = np.linspace(0, np.pi/2, 60)
            T_theta, _, _ = binned_statistic(theta_flat, Tk_flat, statistic='mean', bins=theta_bins)
            ARS_fdtd.append(np.c_[theta_bins[:-1],np.nan_to_num(T_theta)])
        return spec, diff, transmitted, Tp, Tm,T_smallangle,np.asarray(tk_space),np.asarray(ARS_fdtd),np.asarray(thetamask),np.asarray(theta_array)


class Save_meepfields:
    def __init__(self, filename,wvl,res,Lx,Ly=None):
        """
        Creates the dataset with .h5 format for postprocessing meep dft fields.

        Inputs:
        filename : string
            Name of the file
        wvl : ndarray
            Wavelength array (in microns)
        res : int or float
            Meep resolution in 1/microns
        Lx : float
            Length of the simulation domain (in microns) along x-direction
        Ly : float
            Length of the simulation domain (in microns) along y-direction. If None, assumes 2D simulation
        """
        self.file = filename
        self.wavelength = wvl
        self.resolution = res
        self.Lx = Lx
        self.Ly = Ly

    def create_dataset(self,ex_src,ey_src,ez_src,T0=None,R=None,T=None,ex_f=None,ey_f=None,ez_f=None,ex_b=None,ey_b=None,ez_b=None):
        """
        Creates the dataset with .h5 format for postprocessing meep dft fields.

        Inputs:
        filename : string
            Name of the file
        ex_src : ndarray
            Ex fields of the source
        ey_src : ndarray
            Ey fields of the source
        ez_src : ndarray
            Ez fields of the source
        ex_f : ndarray
            Ex fields forward-scattered
        ey_f : ndarray
            Ey fields forward-scattered
        ez_f : ndarray
            Ez fields forward-scattered
        """
        with h5py.File('{f}.h5'.format(f=self.file), 'w') as f:
            grp_b = f.create_group('fields/backscattered')
            grp_f = f.create_group('fields/forwardscattered')
            grp_src = f.create_group('fields/source')

            #backscattered
            if ex_b is not None:
                grp_b.create_dataset("ex", data=ex_b)
            if ey_b is not None:
                grp_b.create_dataset("ey", data=ey_b)
            if ez_b is not None:
                grp_b.create_dataset("ez", data=ez_b)

            #forward scattered
            if ex_f is not None:
                grp_f.create_dataset("ex", data=ex_f)
            if ey_f is not None:
                grp_f.create_dataset("ey", data=ey_f)
            if ez_f is not None:
                grp_f.create_dataset("ez", data=ez_f)

            #source
            if ex_src is not None:
                grp_src.create_dataset("ex", data=ex_src)
            if ey_src is not None:
                grp_src.create_dataset("ey", data=ey_src)
            if ez_src is not None:
                grp_src.create_dataset("ez", data=ez_src)

            f.attrs['wvl'] = self.wavelength
            f.attrs['res'] = self.resolution
            f.attrs['Source_flux'] = T0
            f.attrs['Reflectance'] = R
            f.attrs['Transmittance'] = T
            f.attrs['Lx'] = self.Lx
            f.attrs['Ly'] = self.Ly
            #f.attrs['probe_zpos'] = z_pos
    def save_fluxes(self,T0,R,T):
        with h5py.File('{f}.h5'.format(f=self.file), 'w') as f:
            f.attrs['wvl'] = self.wavelength
            f.attrs['res'] = self.resolution
            f.attrs['Source_flux'] = T0
            f.attrs['Reflectance'] = R
            f.attrs['Transmittance'] = T
