import numpy as np
import meep as mp
from tostada.PointDistribution import PointDistribution
from tostada.PhaseDistribution import PhaseDistribution
from tostada.util.materials import Material

class Meep_geometry:
    """
    Create a meep geometrical object from a given PhaseDistribution of PointDistribution and 
    """
    def __init__(self, distribution, Material):
        self.distribution = distribution
        self.Material = Material

    def create_meep_geometry(self,geometry_list=None, center=[0,0,0], object_shape='sphere',pore_medium=None,**shape_kwargs):
        """
        Create or append a list of meep geometrical objects. If PointDistribution, inserts identical objects at positions given by PointDistribution.positions.
        If PhaseDistribution, linearly interpolates the pixels of each phase to permitivitty values.

        Parameters
        ----------
        geometry_list : list, optional
            List of meep geometrical objects that construct the full simulation domain. If None, creates one else appends to it.

        center : list of floats, optional
            Coordinate shifts along each axes required to the point-distribution to correctly place the objects in the meep simulation domain. 
        
        object_shape : string, optional
            Shape of the identical objects to be placed at the points. Available options are: sphere, cylinder, cone, wedge and ellipsoid.       

        pore_medium : meep.Medium , optional
            Meep medium to be used for the pores. Default : refractive index = 1.
        
        shape_kwargs : optional
            Additional arguments required to parameterize the object like height of cylinders or upper radius for truncated cone etc.. 
            Refer to MEEP documentation for further details.
        """
        geometry = [] if geometry_list is None else geometry_list
        if isinstance(self.distribution, PointDistribution):
            return self.meep_geom_from_pointdist(geometry_list=geometry,center=center, object_shape=object_shape, **shape_kwargs)
        elif isinstance(self.distribution, PhaseDistribution):
            pore_medium = mp.Medium(index=1) if pore_medium is None else pore_medium
            return self.meep_geom_from_phasedist(geometry_list=geometry,center=center,pore_medium=pore_medium)
        else:
            raise ValueError("Unsupported class type. Please provide an instance of PointDistribution or PhaseDistribution.")
        
    def meep_geom_from_pointdist(self,center,geometry_list,object_shape='sphere',**shape_kwargs):
        """
        Create a list of identical meep geometrical objects at positions given by tostada.PointDistribution.

        Parameters
        ----------
        geometry_list : list, optional
            List of meep geometrical objects that construct the full simulation domain. If None, creates one else appends to it.
        
        object_shape : string, optional
            Shape of the identical objects to be placed at the points. Available options are sphere, cylinder, cone, wedge and ellipsoid.
        
        center : list of floats, optional
            Coordinate shifts along each axes required to the point-distribution to correctly place the objects in the meep simulation domain. 
        
        **shape_kwargs
            Additional arguments required to parameterize the object like height (for cylinder) or upper radius (for truncated cone) etc.. 
            Refer to MEEP documentation for further details.

        Returns
        -------
        geomtry : list
            List of meep geometrical objects. This can directly be used inside meep.Simulation() class. 
            Refer to `photonic_band_gap_2D.ipynb` for further details.
        """

        x0,y0,z0 = center[0], center[1], center[2]
        def meep_object(keyword, **shape_kwargs):
            shape_dict = {
                'sphere': mp.Sphere,
                'cylinder': mp.Cylinder,
                'ellipsoid': mp.Ellipsoid,
                'cone' : mp.Cone,
                'wedge' : mp.Wedge
            }
            if keyword not in shape_dict:
                raise ValueError(f"Invalid shape type '{keyword}'. Supported: {list(shape_dict.keys())}")
            shape_class = shape_dict[keyword]
            return shape_class

        radius = self.distribution.diameter / 2
        positions = self.distribution.positions
        
        for i in range(positions.shape[0]):
            geometry_list.append(meep_object(keyword=object_shape)(center=mp.Vector3(positions[i,0]+x0, positions[i,1]+y0, positions[i,2] + z0),
                                                               radius=radius,material=self.Material.meep_medium, **shape_kwargs))
            
        return geometry_list
    
    def meep_geom_from_phasedist(self,geometry_list,center,pore_medium):
        """
        Create a meep geometrical object (meep.Block) from tostada.PhaseDistribution. 
        Since a phase distribution is an image, it interpolates the pixel values to two mediums; pore_medium and self.Material with 0 being the former and 1 being the latter, respectively.
        
        Parameters
        ----------
        geometry_list : list, optional
            List of meep geometrical objects that construct the full simulation domain. If None, creates one else appends to it.
        
        center : list of floats, optional
            Coordinate shifts along each axes required to the distribution to correctly place it in the meep simulation domain. 
        
        pore_medium : meep.Medium , optional
            Meep medium to be used for the pores. Default : refractive index = 1.

        Returns
        -------
        geomtry : list
            List of meep geometrical objects. This can directly be used inside meep.Simulation() class. 
            Refer to `photonic_band_gap_2D.ipynb` for further details.
            
        """
        Lx,Ly = self.distribution.Lx,self.distribution.Ly
        Lz = 0 if self.distribution.Lz is None else Lz
        x0,y0,z0 = center[0], center[1], center[2]
        Weights = self.distribution.image
        Weights = Weights/np.max(Weights) #ensures max = 1
        if (self.distribution.ndim==3):
            grid_size = mp.Vector3(Weights.shape[0],Weights.shape[1],Weights.shape[2])
        else:
            grid_size = mp.Vector3(Weights.shape[0],Weights.shape[1])

        design_variables = mp.MaterialGrid(grid_size=grid_size,
                                           medium1=pore_medium,medium2=self.Material.meep_medium,
                                           weights = Weights,grid_type='U_MEAN')
        
        geometry_list.append(mp.Block(center=mp.Vector3(x0,y0,z0), 
                        size=mp.Vector3(Lx,Ly,Lz),
                        material=design_variables)   
                        )
        
        return geometry_list

        