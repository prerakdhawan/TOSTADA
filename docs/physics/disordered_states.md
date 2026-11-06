# Disordered states

```{contents} Table of Contents
:local:
:depth: 4
```

## Point Distribution

## Phase Distribution

## Textures

## Formating example

The structure factor $S(\mathbf{q})$ describes how an incoming wave is scattered by a collection of $N$ points. It compares the scattered Intensity by the whole structure to $N$ independant scatterers. Therefore it depends only on how the points are arranged.

For points at positions $\mathbf{r}_1, \dots, \mathbf{r}_N$ it is defined as

$$
S(\mathbf{q}) = \frac{1}{N}\left|\sum_{j=1}^{N} e^{i\mathbf{q}\cdot\mathbf{r}_j}\right|^2 .
$$ 

Here $\mathbf{q}$ is the scattering vector, the difference between the outgoing and the incoming wave vector. Each point contributes a wave with the phase $\mathbf{q}\cdot\mathbf{r}_j$. The sum adds all these waves, and the squared magnitude gives the intensity. In a periodic box of side length $L$ only the values $\mathbf{q} = 2\pi\mathbf{n}/L$ with integer $\mathbf{n}$ are allowed.

The sum is the Fourier component of the point density $\rho(\mathbf{r}) = \sum_j \delta(\mathbf{r}-\mathbf{r}_j)$, so $S(\mathbf{q})$ measures how strongly a density wave with wavelength $2\pi/|\mathbf{q}|$ is present in the arrangement.

Typical limiting cases:

- **Uncorrelated points** (Poisson distribution): the phases are random, hence $S(\mathbf{q}) \approx 1$.
- **Crystal:** sharp Bragg peaks at the reciprocal lattice vectors, and $S \approx 0$ in between.
- **Hyperuniform systems:** large-scale density fluctuations are suppressed, $S(\mathbf{q}) \to 0$ for $|\mathbf{q}| \to 0$.

The value at $\mathbf{q}=0$ is trivially $S = N$, since all phases vanish, and is usually excluded.

The structure factor of a {py:class}`~tostada.PointDistribution.PointDistribution` can be computed by
{py:meth}`~tostada.PointDistribution.PointDistribution.Sk_jax_custom`:

```{literalinclude} ../../src/tostada/PointDistribution.py
:language: python
:pyobject: PointDistribution.Sk_jax_custom
:lines: 1-25
```

### Include a whole function

```{literalinclude} ../../src/tostada/PointDistribution.py
:language: python
:pyobject: PointDistribution.Sk_jax_custom
```