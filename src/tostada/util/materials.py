import numpy as np
import meep as mp
import os
import inspect

current_file = inspect.getfile(inspect.currentframe())
this_dir = os.path.dirname(os.path.abspath(current_file))

##################################################### STILL IN DEVELOPMENT ###############################################
class Material: 
    """
    Materials library of commonly used materials in optical and mechanical simulations. 
    For optical simulations, the materials are saved using Lorentzian and Drude fitted oscillator parameters.

    Parameters
    ----------
    name : str
        Name of the material

    Description : str, optional
        Description of experimental details of the measurement or any other details to be noted before using the dataset.

    refractive index : float, optional
        Wavelength-independent loss-less refractive index.
    
    optical_dispersions : [ N_oscillators x 3 ] ndarray
            Optimal oscillator parameters fitted to experimental curves. 
            For each of N oscillators, the 3 parameters are the ones parameterizing a Lorentzian oscillator. 
            These parameters can be used directly to create a MEEP dispersive material using `self.meep_dispersion()`. 
            If any frequency is zero, uses Drude Dispersion model instead. To get a wavelength-dependent dispersion from this, use `self.create_dispersion()`

    youngs_modulus : float, optional 
        Young's modulus of the material in N/m^2.

    poisson_ratio : float, optional
        Poisson's ratio of the material.
    
    density : float, optional
        Mass density of the material in kg/m^3.
    """

    def __init__(self, name, Description=None, refractive_index=1.0,
                 optical_measurements=None, optical_dispersions=None, youngs_modulus=None, 
                 poisson_ratio=None, density=None):
        
        self.name = name
        self.Description = Description
        self.refractive_index = refractive_index
        self.optical_measurements=optical_measurements
        self.optical_dispersions = optical_dispersions
        self.youngs_modulus = youngs_modulus 
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.eps_inf = self.refractive_index**2 # epsilon in the infinite frequency limit
        self.meep_medium = self.Meep_medium() #self.meep_dispersion() if self.optical_dispersions is not None else None


    def Meep_medium(self):
        """
        Create a `meep.Medium` object with or without dispersion.
        """
        if (self.optical_dispersions is None):
            return mp.Medium(index=self.refractive_index)
        else:
            return self.meep_dispersion()

    def meep_dispersion(self):
        """
        Returns a `meep.Medium()` object using lorentzian dispersions. 
        The infinite frequency epsilon is taken from refractive index specified in the class definition. 

        Parameters
        ----------

        optical_dispersion : [ N_oscillators x 3 ] ndarray
            Optimal oscillator parameters fitted to experimental curves. 
            For each of N oscillators, the 3 parameters are the ones parameterizing a Lorentzian oscillator ordered as [sigma_n , frequency_n, gamma_n]. 
            These parameters can be used directly to create a MEEP dispersive material using `self.meep_dispersion()`. 
            If any frequency is zero, uses Drude Dispersion model instead. To get a wavelength-dependent dispersion from this, use `self.create_dispersion()`

        """
        num_lorentzians = self.optical_dispersions.shape[0]
        E_susceptibilities = []
        for n in range(num_lorentzians):
            mymaterial_freq = self.optical_dispersions[n][1]
            mymaterial_gamma = self.optical_dispersions[n][2]

            if mymaterial_freq == 0:
                mymaterial_sigma = self.optical_dispersions[n][0]
                E_susceptibilities.append(
                    mp.DrudeSusceptibility(
                        frequency=1.0, gamma=mymaterial_gamma, sigma=mymaterial_sigma
                    )
                )
            else:
                mymaterial_sigma = self.optical_dispersions[n][0]   #/ mymaterial_freq**2
                E_susceptibilities.append(
                    mp.LorentzianSusceptibility(
                        frequency=mymaterial_freq,
                        gamma=mymaterial_gamma,
                        sigma=mymaterial_sigma,
                    )
                )

        meep_material = mp.Medium(epsilon=self.eps_inf, E_susceptibilities=E_susceptibilities)
        return meep_material

    def create_dispersion(self,wavelength,Meep_medium=None):
        """
        Retrieves the complex refractive index of a Meep medium as a function of input wavelengths.

        Parameters
        ----------
        wavelength : ndarray
            Wavelength array (in microns). 
        Meep_medium : meep.Medium, optional
            Meep medium whose refractive index needs to be interpolated to the desired wavelengths. If None, uses self.meep_medium().
        """
        Meep_medium = self.meep_medium if Meep_medium is None else Meep_medium
        ref_index = []
        for i in range(wavelength.shape[0]):
            ref_index.append(Meep_medium.epsilon(1/wavelength[i])[0,0]**0.5)
        ref_index = np.array(ref_index)
        nk_data = np.c_[wavelength,np.real(ref_index),np.imag(ref_index)]
        return nk_data


optical_dispersion_aSi = np.array([
    [7.540660445872102,3.1820013945145043,0],
    [0.898704482262214,2.179867386716985,0.4223978381105804],
    [1.667485230857076,2.49834208418389,0.3582448627183054]
])#eps=1.15


aSi_experiment = np.loadtxt(os.path.join(this_dir, 'aSi_new.txt'))
ITO_experiment = np.loadtxt(os.path.join(this_dir, 'ITO_rear_TCO.txt'))
ITO_front_experiment = np.loadtxt(os.path.join(this_dir, 'ITO_front_TCO.txt'))
glass_experiment = np.loadtxt(os.path.join(this_dir, 'glass-nabs.txt'))
AAO_experiment = np.loadtxt(os.path.join(this_dir, 'Al2O3_mikroMD.txt'))
Au_experiment = np.loadtxt(os.path.join(this_dir, 'Au_opticalConstants.txt'),skiprows=1)

aSi_experiment[:,0] = aSi_experiment[:,0]/1e3
ITO_experiment_ = (ITO_experiment[:,1] + 1j*ITO_experiment[:,2])**0.5
ITO_experiment = np.c_[ITO_experiment[:,0]/1e3, np.real(ITO_experiment_), np.imag(ITO_experiment_)]
glass_experiment[:,0] = glass_experiment[:,0]/1e3
AAO_experiment[:,0] = AAO_experiment[:,0]/1e3
Au_experiment[:,0] = Au_experiment[:,0]/1e3

optical_dispersion_Au = np.array([
    [2.1106141181173115,3.18268130165899,1.3298582695423644],
    [1.0916364599001387,2.345315972968078,0.666281550009676],
    [3.177570630766605,10.760165429705978,0],
    [50.339304923660606,0,0.022712419439515623] # Drude dispersion
]) #eps=1

optical_dispersion_ITO = np.array([
    [2.6342992895083106,1,0.04457406356962057],
    [0.06764048809876368,3.2201108251261896,0.4499640219854816],
    [1.9818998552188003,4.545912258900352,0]
]) #eps=1.3

optical_dispersion_ITO_front = np.array([
    [1.13211,5.045,0],
    [0.165840,3.236105,0.477245],
    [1.2765327,1,0.080605],
    [1.12228,4.240183,0.1243208]
]) #eps=1.5

optical_dispersion_AAO = np.array([
    [0.9,6.2106400,0],
]) #eps=1.44

optical_dispersion_glass = np.array([
    [0.645,6.36885882,0],
]) #eps=1.6



AAO = Material(name='Anodized Aluminium Oxide (Al2O3)',
               refractive_index=np.sqrt(1.44), #only for eps_inf. Not to be confused with wavelength-independent refractive index of this media.
               youngs_modulus=300e9,
               optical_measurements=AAO_experiment,
               optical_dispersions=optical_dispersion_AAO,
               poisson_ratio=0.33,
               density=3900)

Glass = Material(
                name='Glass (SiO2)',
                Description = 'Obtained from measurements at MLU?',
                refractive_index=np.sqrt(1.6), #only for eps_inf. Not to be confused with wavelength-independent refractive index of this media.
                optical_measurements=glass_experiment,
                optical_dispersions=optical_dispersion_glass
                )

Chitin = Material(name='Chitin',
                  refractive_index=1.54)

Au = Material(
                name='Gold (Au)',
                Description = 'Obtained from measurements at MLU',
                refractive_index=np.sqrt(1), #only for eps_inf. Not to be confused with wavelength-independent refractive index of this media.
                optical_dispersions=optical_dispersion_Au,
                optical_measurements=Au_experiment
                )

ITO_rear = Material(
                name='Indium Tin Oxide (ITO)',
                Description = 'Rear measurement of the substrate. Unreliable (as of 23.07.25)',
                refractive_index=np.sqrt(1.3), #only for eps_inf. Not to be confused with wavelength-independent refractive index of this media.
                optical_dispersions=optical_dispersion_ITO,
                optical_measurements=ITO_experiment
                )

ITO_front = Material(
                name='Indium Tin Oxide (ITO)',
                Description = 'Rear measurement of the substrate. Unreliable (as of 23.07.25)',
                refractive_index=np.sqrt(1.5), #only for eps_inf. Not to be confused with wavelength-independent refractive index of this media.
                optical_dispersions=optical_dispersion_ITO_front,
                optical_measurements=ITO_front_experiment
                )

aSi = Material(
                name='Amorphous Silicon (aSi)',
                Description = 'Obtained from measurements at MLU',
                optical_measurements = aSi_experiment,
                refractive_index=np.sqrt(1.15), #only for eps_inf. Not to be confused with wavelength-independent refractive index of this media.
                optical_dispersions=optical_dispersion_aSi,
                )
