# API Reference

This page is an overview of the relevant classes and functions of TOSTADA. For an easier introduction have a look at {doc}`Getting Started </tutorials/basics/getting_started>`.

Tostada currently defines systems into two categories:

1. [Point distributions](#point-distribution): Systems categorized by spatial distribution of points in space.
2. [Phase distributions](#phase-distribution): Systems categorized by spatial distribution of phases (local regions/volumes) in space.

The core classes for generating distributions are:
1. [Pointprocess](#pointprocess): Generating a point distribution through perturbations of lattices or through Monte-Carlo type probabilistic processes.
2. [Phaseprocess](#phaseprocess): Generating a phase distribution either through gaussian random fields, time-dependent PDEs or by simply mapping of point distributions to phase distribution.

The core classes for studying these distributions are:
1. [Point distributions](#point-distribution): Methods to study statistical properties of point distributions in real-space (point correlations) or reciprocal-space (structure factor).
2. [Phase distributions](#phase-distribution): Methods to study statistical properties of phase distributions in real-space (auto-correlations) or reciprocal-space (spectral density).

```{contents} Table of Contents
:local:
:depth: 4
```

## Generating distributions

### Pointprocess

```{eval-rst}
.. autoclass:: tostada.pointprocess.Pointprocess
```

#### Generate point distributions

```{eval-rst}
    .. automethod:: tostada.pointprocess.Pointprocess.rect_lattice

    .. automethod:: tostada.pointprocess.Pointprocess.hex_lattice

    .. automethod:: tostada.pointprocess.Pointprocess.poisson

    .. automethod:: tostada.pointprocess.Pointprocess.hyperplane_intersection

    .. automethod:: tostada.pointprocess.Pointprocess.fourier_dual_ocp

    .. automethod:: tostada.pointprocess.Pointprocess.ginibre

    .. automethod:: tostada.pointprocess.Pointprocess.anti_hyperuniform

    .. automethod:: tostada.pointprocess.Pointprocess.hermite_gaussian

    .. automethod:: tostada.pointprocess.Pointprocess.reciprocal_space_optimization

    .. automethod:: tostada.pointprocess.Pointprocess.RandomSequentialAdsorption
``` 


### Phaseprocess

```{eval-rst}
.. autoclass:: tostada.phaseprocess.Phaseprocess
```

#### Generate phase distributions

```{eval-rst}
    .. automethod:: tostada.phaseprocess.Phaseprocess.poisson

    .. automethod:: tostada.phaseprocess.Phaseprocess.GRF_microstructure

    .. automethod:: tostada.phaseprocess.Phaseprocess.spinodal_decomposed_microstructure

    .. automethod:: tostada.phaseprocess.Phaseprocess.microstructure_from_pointdistribution

    .. automethod:: tostada.phaseprocess.Phaseprocess.phasefield_microstructure
``` 

### Gaussian random fields

```{eval-rst}
.. autofunction:: tostada.GaussianRandomFunction.generate_grf
```

## Analysing distributions

### Point Distribution

```{eval-rst}
.. autoclass:: tostada.PointDistribution.PointDistribution
```

#### Creating distributions

```{eval-rst}
    .. automethod:: tostada.PointDistribution.PointDistribution.Phaseobject

    .. automethod:: tostada.PointDistribution.PointDistribution.Phaseobject_voronoi

    .. automethod:: tostada.PointDistribution.PointDistribution.Phaseobject_trivalent

    .. automethod:: tostada.PointDistribution.PointDistribution.Phaseobject_CVT
``` 

#### Edit and review point distributions

```{eval-rst}
    .. automethod:: tostada.PointDistribution.PointDistribution.zoom

    .. automethod:: tostada.PointDistribution.PointDistribution.tessellate

    .. automethod:: tostada.PointDistribution.PointDistribution.tesselate_BoxSize

    .. automethod:: tostada.PointDistribution.PointDistribution.lloyd_relaxation

    .. automethod:: tostada.PointDistribution.PointDistribution.relax_points

    .. automethod:: tostada.PointDistribution.PointDistribution.overlap

    .. automethod:: tostada.PointDistribution.PointDistribution.build_tree

    .. automethod:: tostada.PointDistribution.PointDistribution.save

    .. automethod:: tostada.PointDistribution.PointDistribution.load
```


#### Statistics and analysis

```{eval-rst}
    .. automethod:: tostada.PointDistribution.PointDistribution.Pair_correlation

    .. automethod:: tostada.PointDistribution.PointDistribution.RealSpaceCorrelations

    .. automethod:: tostada.PointDistribution.PointDistribution.pair_corr_Sq_jax

    .. automethod:: tostada.PointDistribution.PointDistribution.Hyperuniformity_data

    .. automethod:: tostada.PointDistribution.PointDistribution.hyperuniformity_chi

    .. automethod:: tostada.PointDistribution.PointDistribution.Q_from_chi

    .. automethod:: tostada.PointDistribution.PointDistribution.Dmean_from_q

    .. automethod:: tostada.PointDistribution.PointDistribution.translational_order_metric

    .. automethod:: tostada.PointDistribution.PointDistribution.compute_morphology

    .. automethod:: tostada.PointDistribution.PointDistribution.get_morphological_parameters

    .. automethod:: tostada.PointDistribution.PointDistribution.getMk_theory
```

#### Computation

```{eval-rst}
    .. automethod:: tostada.PointDistribution.PointDistribution.ReciprocalSpace

    .. automethod:: tostada.PointDistribution.PointDistribution.Sk_jax_custom

    .. automethod:: tostada.PointDistribution.PointDistribution.nk_jax_custom

    .. automethod:: tostada.PointDistribution.PointDistribution.SpectralDensity

    .. automethod:: tostada.PointDistribution.PointDistribution.Hermitiansymmetry

    .. automethod:: tostada.PointDistribution.PointDistribution.angular_average
```


### Phase distribution

```{eval-rst}
.. autoclass:: tostada.PhaseDistribution.PhaseDistribution
```

#### Loading and saving

```{eval-rst}
    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.load

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.save

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.get_resolution
```

#### Edit and review phase distributions

```{eval-rst}
    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.zoom

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.tessellate

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.invert_phases

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.remove_plane
```

#### Statistics and analysis

```{eval-rst}
    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.Hyperuniformity_data

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.Dmean_from_q

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.translational_order_metric

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.compute_morphology

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.get_morphological_parameters

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.SphericalContactDistribution
```

#### Computation

```{eval-rst}
    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.ReciprocalSpace

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.Spectraldensity

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.Autocovariance

    .. automethod:: tostada.PhaseDistribution.PhaseDistribution.compute_principal_ACF
```



### Statistics

```{eval-rst}
    .. py:module:: tostada.Statistics
```

Standalone functions for correlation functions, hyperuniformity, analytical models and potentials.

#### Correlation and spectral functions

```{eval-rst}
    .. autofunction:: tostada.Statistics.angular_average

    .. autofunction:: tostada.Statistics.fwhm_and_H

    .. autofunction:: tostada.Statistics.tau

    .. autofunction:: tostada.Statistics.dmean_from_qpeak

    .. autofunction:: tostada.Statistics.SphericalContactDistribution
```

#### Analytical models and potentials

```{eval-rst}
    .. autofunction:: tostada.Statistics.rho_analytic

    .. autofunction:: tostada.Statistics.Sq_analytic

    .. autofunction:: tostada.Statistics.RSA_distribution_function

    .. autofunction:: tostada.Statistics.MaternIII_distribution_function

    .. autofunction:: tostada.Statistics.yukawa_potential_hard

    .. autofunction:: tostada.Statistics.harmonic_potential
```

### Morphology analysis

```{eval-rst}
.. autoclass:: tostada.Statistics.Morphology
```

```{eval-rst}
    .. automethod:: tostada.Statistics.Morphology.morphology

    .. automethod:: tostada.Statistics.Morphology.polygon_normals_and_lengths

    .. automethod:: tostada.Statistics.Morphology.structure_metrics

    .. automethod:: tostada.Statistics.Morphology.misorientation_angles

    .. automethod:: tostada.Statistics.Morphology.get_local_properties

    .. automethod:: tostada.Statistics.Morphology.get_morphology_stats

    .. automethod:: tostada.Statistics.Morphology.draw_polygon
```

## Optimization

```{eval-rst}
.. autoclass:: tostada.Optimization.Optimization
```

```{eval-rst}
    .. automethod:: tostada.Optimization.Optimization.optimize

    .. automethod:: tostada.Optimization.Optimization.Objective

    .. automethod:: tostada.Optimization.Optimization.objective_with_custom_grad

    .. automethod:: tostada.Optimization.Optimization.custom_grad

    .. automethod:: tostada.Optimization.Optimization.pair_correlation_objective

    .. automethod:: tostada.Optimization.Optimization.grad_pair_correlation_objective

    .. automethod:: tostada.Optimization.Optimization.finite_diff_grad

    .. automethod:: tostada.Optimization.Optimization.interpolate_Target1D
```

## Physics solvers

### PhaseField

```{eval-rst}
.. autoclass:: tostada.physics.phase_field_dynamics.PhaseField
```

#### Chemical potentials

```{eval-rst}
    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.mu_Cahn_Hilliard

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.mu_Swift_Hohenberg

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.mu_PFC_square

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.mu_PFC_square_spectral

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.mu_PFC_fcc

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.KS_damped_spectral

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.chemical_potential

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.PFC_eqn
```

#### Solving and visualizing

```{eval-rst}
    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.initialize_lattice

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.initialize_lattice_2

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.forward_euler

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.solve

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.plot_evolution

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.plot_lattice

    .. automethod:: tostada.physics.phase_field_dynamics.PhaseField.animate
```

### CahnHilliard

```{eval-rst}
.. autoclass:: tostada.physics.Cahn_Hilliard.CahnHilliard
```

#### Solving and visualizing Cahn-Hilliard

```{eval-rst}
    .. automethod:: tostada.physics.Cahn_Hilliard.CahnHilliard.CH_eqn

    .. automethod:: tostada.physics.Cahn_Hilliard.CahnHilliard.initialize_lattice

    .. automethod:: tostada.physics.Cahn_Hilliard.CahnHilliard.forward_euler

    .. automethod:: tostada.physics.Cahn_Hilliard.CahnHilliard.solve

    .. automethod:: tostada.physics.Cahn_Hilliard.CahnHilliard.plot_evolution

    .. automethod:: tostada.physics.Cahn_Hilliard.CahnHilliard.plot_lattice

    .. automethod:: tostada.physics.Cahn_Hilliard.CahnHilliard.animate
```

### LatticeParticleMethod

```{eval-rst}
.. autoclass:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod
```

#### Loading, saving and geometry

```{eval-rst}
    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.prepare_geometry

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.get_strain_parameters

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.save

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.load

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.periodize_boundaries

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.boundary_condition
```

#### Running simulations

```{eval-rst}
    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.run_sim

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.run_sim_damage

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.Homogenize
```

#### Displacements, strains and stresses

```{eval-rst}
    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.get_dists_exc

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.compute_local_properties

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.compute_local_properties_damage

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.get_displacements

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.total_displacements

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.get_strains

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.get_strains_old

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.get_stresses

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.reshape_fields

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.surf_integration

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.EnergyDensity

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.strain_energy

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.Global_Energies
```

#### Constitutive models and damage

```{eval-rst}
    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.stress_from_constitutive_relations

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.stress_truncated_rankine

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.stress_drucker_prager

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.stress_ottosen

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.damage_evolution

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.LatticeParticleMethod.animate_fields
```

#### homogenization_results

```{eval-rst}
.. autoclass:: tostada.physics.Lattice_Particle_Method_gpu.homogenization_results
```

#### Loading and saving results

```{eval-rst}
    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.homogenization_results.save

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.homogenization_results.load
```

#### Elastic properties

```{eval-rst}
    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.homogenization_results.build_compliance_matrix2D

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.homogenization_results.voigt_reuss_parameters

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.homogenization_results.get_angular_properties

    .. automethod:: tostada.physics.Lattice_Particle_Method_gpu.homogenization_results.directional_properties
```

## Optics

### MEEP media

```{eval-rst}
.. autoclass:: tostada.util.materials.Material
```

```{eval-rst}
    .. automethod:: tostada.util.materials.Material.Meep_medium

    .. automethod:: tostada.util.materials.Material.meep_dispersion

    .. automethod:: tostada.util.materials.Material.create_dispersion
```

### Meep Geometry

```{eval-rst}
.. autoclass:: tostada.physics.meep_geometry.Meep_geometry
```

```{eval-rst}
    .. automethod:: tostada.physics.meep_geometry.Meep_geometry.create_meep_geometry

    .. automethod:: tostada.physics.meep_geometry.Meep_geometry.meep_geom_from_pointdist

    .. automethod:: tostada.physics.meep_geometry.Meep_geometry.meep_geom_from_phasedist
```

### Postprocessing MEEP fields

```{eval-rst}
.. autoclass:: tostada.util.postprocess_meep_fields.Postprocess_fields
```


```{eval-rst}
    .. automethod:: tostada.util.postprocess_meep_fields.Postprocess_fields.create_kparallel

    .. automethod:: tostada.util.postprocess_meep_fields.Postprocess_fields.get_angles

    .. automethod:: tostada.util.postprocess_meep_fields.Postprocess_fields.helicity_basis_vectors

    .. automethod:: tostada.util.postprocess_meep_fields.Postprocess_fields.theta_mask

    .. automethod:: tostada.util.postprocess_meep_fields.Postprocess_fields.Helicitytransform

    .. automethod:: tostada.util.postprocess_meep_fields.Postprocess_fields.backscattered_efficiencies

    .. automethod:: tostada.util.postprocess_meep_fields.Postprocess_fields.transmitted_efficiencies
```

### Saving MEEP fields

```{eval-rst}
.. autoclass:: tostada.util.postprocess_meep_fields.Save_meepfields
```

```{eval-rst}
    .. automethod:: tostada.util.postprocess_meep_fields.Save_meepfields.create_dataset

    .. automethod:: tostada.util.postprocess_meep_fields.Save_meepfields.save_fluxes
```

## Visualization and utilities

### Visualize

```{eval-rst}
.. autoclass:: tostada.plot_util.Visualize
```

#### Plotting

```{eval-rst}
    .. automethod:: tostada.plot_util.Visualize.plot_distribution

    .. automethod:: tostada.plot_util.Visualize.plot_reciprocal_space

    .. automethod:: tostada.plot_util.Visualize.plot_real_space_correlation

    .. automethod:: tostada.plot_util.Visualize.plot_principal_ACF

    .. automethod:: tostada.plot_util.Visualize.plot_colors

    .. automethod:: tostada.plot_util.Visualize.plot_polar

    .. automethod:: tostada.plot_util.Visualize.plot_detected_polygons

    .. automethod:: tostada.plot_util.Visualize.plot_structure_metric

    .. automethod:: tostada.plot_util.Visualize.plot_fields
```

### ColourSystem

```{eval-rst}
.. autoclass:: tostada.color_system.ColourSystem
```

#### Colour conversion

```{eval-rst}
    .. autofunction:: tostada.color_system.xyz_from_xy

    .. automethod:: tostada.color_system.ColourSystem.gamma_correct

    .. automethod:: tostada.color_system.ColourSystem.colorize_spectra

    .. automethod:: tostada.color_system.ColourSystem.xyz_to_rgb

    .. automethod:: tostada.color_system.ColourSystem.xyz_to_rgb_with_luminance

    .. automethod:: tostada.color_system.ColourSystem.rgb_to_hex

    .. automethod:: tostada.color_system.ColourSystem.spec_to_xyz

    .. automethod:: tostada.color_system.ColourSystem.spectrum_to_XYZ

    .. automethod:: tostada.color_system.ColourSystem.XYZ_to_xyz

    .. automethod:: tostada.color_system.ColourSystem.spec_to_rgb

    .. automethod:: tostada.color_system.ColourSystem.XYZ_to_RGB
```

### Spectrum

```{eval-rst}
    .. autoclass:: tostada.util.Utility.Spectrum
```

#### Angular spectra

```{eval-rst}
    .. automethod:: tostada.util.Utility.Spectrum.interpolate_kspace

    .. automethod:: tostada.util.Utility.Spectrum.Angular_spectrum
```

### File input and output

```{eval-rst}
    .. autofunction:: tostada.util.Utility.read_file

    .. autofunction:: tostada.util.Utility.save_distribution

    .. autofunction:: tostada.util.Utility.load_distribution
```

