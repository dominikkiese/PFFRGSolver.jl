# PFFRGSolver.jl <img src=https://github.com/dominikkiese/PFFRGSolver.jl/blob/main/README/logo.png align="right" height="175" width="250">

**P**seudo-**F**ermion **F**unctional **R**enormalization **G**roup **Solver** <br>
(Julia v1.5 and higher)

# Introduction

The package PFFRGSolver.jl aims at providing an efficient, state-of-the-art multiloop solver for functional renormalization group equations of quantum lattice spin models in the pseudo-fermion representation. It is currently applicable to spin models described by Hamiltonians of the form

<p align="center">
  <img src=https://github.com/dominikkiese/PFFRGSolver.jl/blob/main/README/hamiltonian.png height="70" width="700">
</p>

which can be defined on a variety of pre-implemented two and three dimensional lattices. Internally, PFFRGSolver.jl first computes a reduced representation of the lattice by employing space group symmetries before initializing the renormalization group flow with the bare couplings or, optionally, a solution of the regularized parquet equations, to which the multiloop truncated FRG converges by construction. The RG equations are integrated using the Bogacki-Shampine method with adaptive step size control. In each stage of the flow, real-space spin-spin correlations are computed from the flowing vertices. For a more detailed discussion of the method and its implementation see https://arxiv.org/abs/2011.01269.

# Installation

The package can be installed with the Julia package manager by switching to package mode in the REPL (with `]`) and using

```julia
pkg> add PFFRGSolver
```

# Citation

If you use PFFRGSolver.jl in your work, please acknowledge the package accordingly and cite our preprint

D. Kiese, T.Müller, Y. Iqbal, R. Thomale and S. Trebst, "Multiloop functional renormalization group approach to quantum spin systems", arXiv:2011.01269 (2020)

A suitable bibtex entry is

```
@misc{kiese2020multiloop,
      title={Multiloop functional renormalization group approach to quantum spin systems},
      author={Dominik Kiese and Tobias M\"uller and Yasir Iqbal and Ronny Thomale and Simon Trebst},
      year={2020},
      eprint={2011.01269},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el}
}
```

# Running calculations

In order to simulate e.g. the nearest-neighbor Heisenberg antiferromagnet on the square lattice for a lattice truncation L = 3 simply do

```julia
using PFFRGSolver
launch!("/path/to/output", "square", 3, "heisenberg", "su2", [1.0])
```

The FRG solver allows for more fine grained control over the calculation by various keyword arguments. The full reference can be obtained via `?launch!`. Currently available lattices and models can be obtained in verbose form with `lattice_avail()` and `model_avail()`.

# Post-processing data

Each calculation generates two output files `"/path/to/output_obs"` and `"/path/to/output_cp"` in the HDF5 format, containing observables measured during the RG flow and checkpoints with full vertex data respectively. <br>
The so-obtained real space spin-spin correlations are usually converted to structure factors (or susceptibilities) via a Fourier transform to momentum space, to investigate the ground state predicted by pf-FRG. In the following, example code is provided for computing the momentum resolved structure factor for the full FRG flow, as well as a single cutoff, for Heisenberg models on the square lattice.

```julia
using PFFRGSolver
using HDF5

# generate 50 x 50 momentum space discretization within first Brillouin zone of the square lattice
rx = (-1.0 * pi, 1.0 * pi)
ry = (-1.0 * pi, 1.0 * pi)
rz = (0.0, 0.0)
k  = get_momenta(rx, ry, rz, (50, 50, 0))

# open observable file and output file to save structure factor
file_in  = h5open("/path/to/output_obs", "r")
file_out = h5open("/path/to/output_sf",  "cw")

# compute structure factor for the full flow
compute_structure_factor_flow!(file_in, file_out, k, "diag")

# read so-computed structure factor at cutoff Λ = 1.0 from file_out
sf = read_structure_factor(file_out, 1.0, "diag")

# read so-computed structure factor flow at momentum with largest amplitude with respect to reference scale Λ = 1.0
ref   = read_reference_momentum(file_out, 1.0, "diag")
Λ, sf = read_structure_factor_flow_at_momentum(file_out, ref, "diag")

# read lattice data and real space correlations at cutoff Λ = 1.0 from file_in
l, r = read_lattice(file_in)
χ    = read_χ(file_in, 1.0, "diag")

# compute structure factor at cutoff Λ = 1.0
sf = compute_structure_factor(χ, k, l, r)

# close HDF5 files
close(file_in)
close(file_out)
```

Vertex data can be accessed by reading checkpoints from `"/path/to/output_cp"` like

```julia
using PFFRGSolver
using HDF5

# open checkpoint file of FRG solver
file = h5open("/path/to/output_cp", "r")

# load checkpoint at cutoff Λ = 1.0
Λ, dΛ, m, a = read_checkpoint(file, 1.0)

# close HDF5 file
close(file)
```

The solver generates (if `parquet = true` in the `launch!` command) at least two checkpoints, one with the converged parquet solution used as the initial condition for the FRG and one with the final result at the end of the flow. Additional checkpoints are created according to a timer heuristic, which can be controlled with the `ct` and `wt` keywords in `launch!`.

# Performance notes

The PFFRGSolver.jl package accelerates calculations by making use of Julia's built-in dynamical thread scheduling (`Threads.@spawn`). Even for small systems, the number of flow equations to be integrated is quite tremendous and parallelization is vital to achieve acceptable run times. **We recommend to launch Julia with multiple threads whenever PFFRGSolver.jl is used**, either by setting up the respective enviroment variable `export JULIA_NUM_THREADS=$nthreads` or by adding the `-t` flag when opening the Julia REPL from the terminal i.e. `julia -t $nthreads`. <br>
Note that iterating the parquet equations is quite costly (compared to one loop FRG calculations) and contributes a substantial overhead for computing an initial condition of the flow. It is advisable to turn them on (via the `parquet` keyword in `launch!`) only when accordingly large computing resources are available. <br>
If you are using the package in an HPC environment, make sure that the precompile cache is generated for that CPU architecture on which production runs are performed, as the LoopVectorizations.jl dependency of the solver will unlock compiler optimizations based on the respective hardware.

# SLURM Interface

Calculations with PFFRGSolver.jl on small to medium sized systems can usually be done locally with a low number of threads. However, when the number of loops is increased and high resolution is required, calculations can become quite time consuming and it is advisable to make use of a computing cluster if available. PFFRGSolver.jl exports a few commands, to help people setting up simulations on clusters utilizing the SLURM workload manager (integration for other systems is plannned for future versions). Example code for a rough scan of the phase diagram of the J1-J2 Heisenberg model on the square lattice (with L = 6) is given below.

```julia
using PFFRGSolver

# make new folder and add launcher files for a rough scan of the phase diagram
mkdir("j1j2_square")

for j2 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  save_launcher!("j1j2_square/j2$(j2).jl", "j2$(j2)", "square", 6, "heisenberg", "su2", [1.0, j2], num_σ = 150, num_Ω = 20, num_ν = 30)
end

# set up SLURM parameters as dictionary
sbatch_args = Dict(["account" => "my_account", "nodes" => "1", "ntasks" => "1", "cpus-per-task" => "8", "time" => "02:00:00", "partition" => "my_partition"])

# generate job files
make_repository!("j1j2_square", "/path/to/julia/exe", sbatch_args)
```

The jobs subsequently can be submitted using

```bash
for FILE in j1j2_square/*/*.job; do sbatch $FILE; done
```

After having submitted and run the jobs, results can be gathered in `"j1j2_square/finished"` with the `collect_repository!` command. Note that simulations, which could not be finished in time, have their `overwrite` flag in the respective launcher file set to `false` by `collect_repository!`. As such they can just be resubmitted and will continue calculations from the last available checkpoint.

# Literature

For further reading on technical aspects of (multiloop) pf-FRG see

* [1] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.144410
* [2] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.045144
* [3] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.064416
* [4] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.035162
* [5] https://arxiv.org/abs/2011.01268
