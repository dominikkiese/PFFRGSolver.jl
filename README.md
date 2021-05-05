# PFFRG.jl <img src=https://github.com/dominikkiese/PFFRG.jl/blob/main/logo.png align="right" height="175" width="250">

**P**seudo-**F**ermion **F**unctional **R**enormalization **G**roup Solver in Julia (add version info)

# Introduction

The package PFFRG.jl aims at providing efficient state of the art implementations of multiloop solvers for functional renormalization group equations of quantum lattice spin models in the pseudo-fermion representation. It is currently applicable to Heisenberg spin models described by a Hamiltonian of the form

<p align="center">
  <img src=https://github.com/dominikkiese/PFFRG.jl/blob/main/hamiltonian.png height="60" width="200">
</p>
 
which can be defined on a variety of pre-implemented two and three dimensional lattices. Internally, PFFRG.jl first computes a reduced representation of the lattice by employing space group symmetries before initializing the renormalization group flow with a solution of the regularized parquet equations, to which the multiloop truncated FRG converges by construction. The RG equations are integrated using the Bogacki-Shampine method with adaptive step size control. In each stage of the flow, real space spin-spin correlations are computed from the flowing vertices. For a more detailed discussion of the method and its implementation see https://arxiv.org/abs/2011.01269.

# Installation 

(to do when package structure is generated)

# General Usage

(explain general workflow)

# SLURM Interface

(explain repository functions)

# Citation

(add bibtex string here)
