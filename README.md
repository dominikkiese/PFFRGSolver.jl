# PFFRG.jl <img src=https://github.com/dominikkiese/PFFRG.jl/blob/main/logo.png align="right" height="175" width="250">

**P**seudo-**F**ermion **F**unctional **R**enormalization **G**roup Solver in Julia (add version info)

# Introduction

The package PFFRG.jl aims at providing an efficient state of the art multiloop solver for functional renormalization group equations of quantum lattice spin models in the pseudo-fermion representation. It is currently applicable to Heisenberg spin models described by a Hamiltonian of the form

<p align="center">
  <img src=https://github.com/dominikkiese/PFFRG.jl/blob/main/hamiltonian.png height="60" width="200">
</p>
 
which can be defined on a variety of pre-implemented two and three dimensional lattices. Internally, PFFRG.jl first computes a reduced representation of the lattice by employing space group symmetries before initializing the renormalization group flow with a solution of the regularized parquet equations, to which the multiloop truncated FRG converges by construction. The RG equations are integrated using the Bogacki-Shampine method with adaptive step size control. In each stage of the flow, real space spin-spin correlations are computed from the flowing vertices. For a more detailed discussion of the method and its implementation see https://arxiv.org/abs/2011.01269.

# Installation 

(to do when package structure is generated)

# Documentation

All exported functions and structs from PFFRG.jl are equipped with docstrings, which can be accessed by switching to help mode in the Julia REPL.

# Citation

(add bibtex string here)

# Running calculations

To simulate e.g. the nearest-neighbor Heisenberg antiferromagnet on the square lattice for a lattice truncation `L = 3` simply do

```julia
using PFFRG
launch!("/path/to/output", "square", 3, "heisenberg", "su2", [1.0])
```

The FRG solver allows for more fine grained control over the calculation by various keyword arguments detailed in the respective docstring. Currently available lattices and models can be obtained in verbose form with `lattice_avail()` and `model_avail()`.

# Post-processing data

Each calculation generates two output files `"/path/to/output_obs"` and `"/path/to/output_cp"` in the HDF5 format, containing observables measured during the RG flow and checkpoints with full vertex data respectively. 
<br>
The so-obtained real space spin-spin correlations are usually converted to structure factors (or susceptibilities) via a Fourier transform to momentum space, in order to investigate the ground state predicted by pf-FRG. In the following, example code is provided for computing the momentum resolved structure factor for the full FRG flow, as well as a single cutoff, for a Heisenberg model on the square lattice. 

```julia
using PFFRG

# generate 50 x 50 momentum space discretization within first Brillouin zone of the square lattice 
rx = (-1.0 * pi, 1.0 * pi)
ry = (-1.0 * pi, 1.0 * pi)
rz = (0.0, 0.0)
k  = get_momenta(rx, ry, rz, (50, 50, 0))

# open observable file of FRG solver and output file to save structure factors for the full flow 
file_in  = h5open("/path/to/output_obs", "r")
file_out = h5open("/path/to/output_sf",  "cw")

# compute structure factors for the full flow 
compute_structure_factor_flow!(file_in, file_out, k, "diag")

# read so-computed structure factor at cutoff Λ = 1.0 from file_out
sf = read_structure_factor(file_out, 1.0, "diag")

# read so-computed structure factor flow at momentum with largest amplitude with respect to reference scale Λ = 1.0
ref   = read_reference_momentum(file_out, 1.0, "diag")
Λ, sf = read_structure_factor_flow_at_momentum(file_out, "diag", ref)

# read lattice data and real space correlations at cutoff Λ = 1.0 from file_in
l  = read_lattice(file_in)
r  = read_reduced_lattice(file_in)
χ  = read_χ(file_in, 1.0, "diag")

# compute structure factor at cutoff Λ = 1.0
sf = compute_structure_factor(χ, k, l, r)

# close HDF5 files
close(file_in)
close(file_out)
```

Vertex data can be accessed by reading checkpoints from `"/path/to/output_cp"`. For Heisenberg models for example

```julia
using PFFRG 

# open checkpoint file of FRG solver
file = h5open("/path/to/output_cp", "r")

# load checkpoint at cutoff Λ = 1.0
Λ, dΛ, m, a = read_checkpoint(file, "su2", 1.0)

# close HDF5 file 
close(file)
```

# Performance notes 

(explain parallelization)

# SLURM Interface

(explain repository functions)

# Developer notes

(explain how to implement lattices and models)

# Literature

(cite original paper by Johannes, large S and large N paper, Fabian's multiloop paper, multiloop papers by Cologne-Wuerzburg-Madras and LMU group)
