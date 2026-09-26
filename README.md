<div align="center">
<img src="src/tostada/images/tostada_pic1.png" width="800">
</div>

# TOolkit for Spatially TAilored Disordered Arrangement (TOSTADA)

_TOolkit for Spatially TAilored Disordered Arrangement (TOSTADA)_ is a GPU-enabled python package for **creating**, **simulating** and **analyzing** spatially disordered distributions with prescribed correlations in 2D/3D.
The key idea behind `tostada` is to translate inverse-design and statistics tools specifically used in the context of disordered media to open-source physics-based solvers in an inter-operable manner.
This repository brings together multiple computational strategies to explore the physics and geometry of complex disordered systems. Whether it is simulating wave propagation through disordered media, characterizing porous microstructure statistics, or using optimization tools to generate materials with tailored correlation functions — this toolkit has you covered.

## 🔧 Features

### 💻 Generation (Inverse Design)
Generate disordered phase or point distributions with **prescribed spatial statistics** using:
  - Reciprocal-space optimization
  - Gaussian random fields
  - Phase field method with customized potential functions.

### ⚛️ Physics simulation 

- **MEEP Plugins**: Export and simulate wave dynamics in disordered media using MIT's Finite-Difference Time-Domain (FDTD) Maxwell solver, _[MEEP](https://meep.readthedocs.io/)_ with direct plugins for particle-type distributions (for example, distribution of nanodisks) or phase-type distributions (for example, porous microstructures).
- **Lattice Particle Method**: GPU-enabled in-house solver to model mechanical response, homogenization and fracture mechanics in two-phase media for linear regime.
- **Phase field method**: GPU-enabled phase field method for creating different morphologies arriving from phase field crystal, Swift-Hohenberg, Cahn-Hilliard equation etc.

### 📊 Analysis
- **Spatial Statistics Tools**: 
  - Pair correlation functions
  - Structure factor and Spectral density 
  - Morphological structure metrics
  - Hyperuniformity index
  
## Installation

### Standard installation

Until the package is published on PyPI, install straight from GitHub
```bash
pip install "git+https://gitlab.informatik.uni-halle.de/mikromd/tostada.git"
pip install "tostada[gpu] @ git+https://gitlab.informatik.uni-halle.de/mikromd/tostada.git" # NVIDIA GPU (CUDA 12)
```
`device_info()` should report a `CudaDevice` on a GPU installation and a `CpuDevice` otherwise. 

That is all you need for inverse design, spatial statistics, phase-field dynamics and the lattice-particle mechanics solver on CPU/GPU.

### Installation with MEEP (optical simulations)

If you wish to use `tostada` for optical simulations of disordered media, tostada offers a plugin to MEEP. However, MEEP is a C++ package whose MPI build (for parallelized simulations) is currently distributed through **conda-forge, not PyPI**. If conda isn't already installed, use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) for a light version. The full installation can then be done through:

```bash
git clone https://gitlab.informatik.uni-halle.de/mikromd/tostada.git

cd tostada

conda env create -f environment.yml        # CPU
# OR
conda env create -f environment_gpu.yml    # NVIDIA GPU

conda activate tostada                     # or tostada-gpu
```

## 📌 Examples

Explore example notebooks and scripts in the `examples/` folder to get started with:

- Generating disordered hyperuniform media with different properties

- Computing and visualizing structure factors and pair correlations

- Export tostada geometries to MEEP for simulating wave scattering

- Extracting color information from the optical response of a disordered media

and many more...

If you do not have [Jupyterlab](https://jupyter.org/) already in your environment and wish to run the examples notebooks, install the development version 
```bash
pip install "tostada[dev] @ git+https://gitlab.informatik.uni-halle.de/mikromd/tostada.git" 
```
If you have already downloaded a CPU/GPU build, you can simply update using 
```bash
cd tostada
pip install -e .[dev]
```

## 📚 Citation and Acknowledgements

If you use this toolkit in your research, please consider citing this repository link and in future the corresponding publication (coming soon!). 

## ☕ Contributing

Pull requests, suggestions, and issue reports are welcome! Feel free to open an issue or contact us directly.

## 📬 Contact

Feel free to reach out to the maintainer:

Prerak Dhawan
prerak.dhawan@physik.uni-halle.de
