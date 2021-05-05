# PFFRG.jl <img src=https://github.com/dominikkiese/PFFRG.jl/blob/main/logo.png align="right" height="175" width="250">
**P**seudo-**F**ermion **F**unctional **R**enormalization **G**roup Solver in Julia (add version info)

# Introduction

The package PFFRG.jl aims at providing state of the art implementations of (multiloop) solvers for functional renormalization group (i.e. ordinary integro-differential) equations of quantum lattice spin models in the pseudo-fermion representation. It is currently applicable to microscopic spin models with SU(2) symmetry, described by a Hamiltonian of the form

(add Hamiltonian as png)

which can be defined on a variety of pre-implemented two and three dimensional lattices. Internally, PFFRG.jl first computes a reduced representation of the lattice by employing space group symmetries before initializing the renormalization group flow with a solution of the regularized parquet (i.e. Schwinger-Dyson and Bethe-Salpeter) equations, to which the multiloop truncated FRG converges by construction. The RG equations are integrated using the Bogacki-Shampine (3rd order Runge-Kutta) method with adaptive step size control. In each stage of the flow, real space spin-spin correlations are computed from the flowing vertices. For a more detailed discussion of the method and its implementation see https://arxiv.org/abs/2011.01269.

# Installation 

(to do when package structure is generated)

# General Usage

(explain general workflow)

# SLURM Interface

(explain repository functions)

# Citation

(add bibtex string here)
