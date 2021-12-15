# function for measurements and checkpointing
function measure(
    symmetry :: String,
    obs_file :: String,
    cp_file  :: String,
    Λ        :: Float64,
    dΛ       :: Float64,
    χ        :: Vector{Matrix{Float64}},
    χ_tol    :: NTuple{2, Float64},
    t        :: DateTime,
    t0       :: DateTime,
    r        :: Reduced_lattice,
    m        :: Mesh,
    a        :: Action,
    wt       :: Float64,
    ct       :: Float64
    )        :: Tuple{DateTime, Bool}

    # open files
    obs = h5open(obs_file, "cw")
    cp  = h5open(cp_file, "cw")

    # compute correlations
    χp = deepcopy(χ)
    compute_χ!(Λ, r, m, a, χ, χ_tol)

    # save correlations and self energy if respective dataset does not yet exist
    if haskey(obs, "χ/$(Λ)") == false
        save_self!(obs, Λ, m, a)
        save_χ!(obs, Λ, symmetry, m, χ)
    end

    # check for monotonicity of static part of dominant on-site correlation
    idx      = argmax(Float64[maximum(abs.(χ[i])) for i in eachindex(χ)])
    static   = (m.num_χ - 1) ÷ 2 + 1
    monotone = χ[idx][1, static] / χp[idx][1, static] >= 0.995

    # compute current run time (in hours)
    h0 = 1e-3 * (Dates.now() - t0).value / 3600.0

    # if more than half an hour is left to the wall time limit, use ct as timer heuristic for checkpointing
    if wt - h0 > 0.5
        # test if time limit for checkpoint (in hours) has been reached
        h = 1e-3 * (Dates.now() - t).value / 3600.0

        if h >= ct
            # generate checkpoint if it does not exist yet
            if haskey(cp, "a/$(Λ)") == false
                println();
                println("Generating checkpoint at cutoff Λ / |J| = $(Λ) ...")
                checkpoint!(cp, Λ, dΛ, m, a)
                println("Successfully generated checkpoint.")
                println();
            end

            # reset timer
            t = Dates.now()
        end
    # if less than half an hour is left to the wall time, generate checkpoints as if ct = 0. Lower bound to prevent cancellation during checkpoint writing
    elseif 0.1 < wt - h0 <= 0.5
        # generate checkpoint if it does not exist yet
        if haskey(cp, "a/$(Λ)") == false
            println(); println()
            println("Generating checkpoint at cutoff Λ / |J| = $(Λ) ...")
            checkpoint!(cp, Λ, dΛ, m, a)
            println("Successfully generated checkpoint.")
            println(); println()
        end
    end

    # close files
    close(obs)
    close(cp)

    return t, monotone
end





"""
    save_launcher!(
        path        :: String,
        f           :: String,
        name        :: String,
        size        :: Int64,
        model       :: String,
        symmetry    :: String,
        J           :: Vector{<:Any}
        ;
        S           :: Float64            = 0.5,
        β           :: Float64            = 1.0,
        euclidean   :: Bool               = false,
        num_σ       :: Int64              = 25,
        num_Ω       :: Int64              = 15,
        num_ν       :: Int64              = 10,
        num_χ       :: Int64              = 10,
        p_σ         :: NTuple{2, Float64} = (0.3, 1.0),
        p_Ωs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
        p_νs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
        p_Ωt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.3, 50.0),
        p_νt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.5, 50.0),
        p_χ         :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
        lins        :: NTuple{5, Float64} = (5.0, 4.0, 8.0, 6.0, 6.0),
        bounds      :: NTuple{5, Float64} = (1.0, 150.0, 500.0, 250.0, 250.0),
        max_iter    :: Int64              = 10,
        min_eval    :: Int64              = 10,
        max_eval    :: Int64              = 100,
        Σ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
        Γ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
        χ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
        parquet_tol :: NTuple{2, Float64} = (1e-8, 1e-6),
        ODE_tol     :: NTuple{2, Float64} = (1e-8, 1e-2),
        loops       :: Int64              = 1,
        parquet     :: Bool               = false,
        Σ_corr      :: Bool               = true,
        initial     :: Float64            = 50.0,
        final       :: Float64            = 0.05,
        bmin        :: Float64            = 1e-4,
        bmax        :: Float64            = 0.2,
        overwrite   :: Bool               = true,
        cps         :: Vector{Float64}    = Float64[],
        wt          :: Float64            = 23.5,
        ct          :: Float64            = 4.0
        )           :: Nothing

Generate executable Julia file `path` which sets up and runs the FRG solver.
For more information on the different solver parameters see documentation of `launch!`.
"""
function save_launcher!(
    path        :: String,
    f           :: String,
    name        :: String,
    size        :: Int64,
    model       :: String,
    symmetry    :: String,
    J           :: Vector{<:Any}
    ;
    S           :: Float64            = 0.5,
    β           :: Float64            = 1.0,
    euclidean   :: Bool               = false,
    num_σ       :: Int64              = 25,
    num_Ω       :: Int64              = 15,
    num_ν       :: Int64              = 10,
    num_χ       :: Int64              = 10,
    p_σ         :: NTuple{2, Float64} = (0.3, 1.0),
    p_Ωs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
    p_νs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
    p_Ωt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.3, 50.0),
    p_νt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.5, 50.0),
    p_χ         :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
    lins        :: NTuple{5, Float64} = (5.0, 4.0, 8.0, 6.0, 6.0),
    bounds      :: NTuple{5, Float64} = (1.0, 150.0, 500.0, 250.0, 250.0),
    max_iter    :: Int64              = 10,
    min_eval    :: Int64              = 10,
    max_eval    :: Int64              = 100,
    Σ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
    Γ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
    χ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
    parquet_tol :: NTuple{2, Float64} = (1e-8, 1e-6),
    ODE_tol     :: NTuple{2, Float64} = (1e-8, 1e-2),
    loops       :: Int64              = 1,
    parquet     :: Bool               = false,
    Σ_corr      :: Bool               = true,
    initial     :: Float64            = 50.0,
    final       :: Float64            = 0.05,
    bmin        :: Float64            = 1e-4,
    bmax        :: Float64            = 0.2,
    overwrite   :: Bool               = true,
    cps         :: Vector{Float64}    = Float64[],
    wt          :: Float64            = 23.5,
    ct          :: Float64            = 4.0
    )           :: Nothing

    # convert J for type safety
    J = Array{Array{Float64,1},1}([[x...] for x in J])

    open(path, "w") do file
        # load source code
        write(file, "using PFFRGSolver \n \n")

        # setup for launcher function
        write(file, """launch!("$(f)",
                    "$(name)",
                    $(size),
                    "$(model)",
                    "$(symmetry)",
                    $(J),
                    S           = $(S),
                    β           = $(β),
                    euclidean   = $(euclidean),
                    num_σ       = $(num_σ),
                    num_Ω       = $(num_Ω),
                    num_ν       = $(num_ν),
                    num_χ       = $(num_χ),
                    p_σ         = $(p_σ),
                    p_Ωs        = $(p_Ωs),
                    p_νs        = $(p_νs),
                    p_Ωt        = $(p_Ωt),
                    p_νt        = $(p_νt),
                    p_χ         = $(p_χ),
                    lins        = $(lins),
                    bounds      = $(bounds),
                    max_iter    = $(max_iter),
                    min_eval    = $(min_eval),
                    max_eval    = $(max_eval),
                    Σ_tol       = $(Σ_tol),
                    Γ_tol       = $(Γ_tol),
                    χ_tol       = $(χ_tol),
                    parquet_tol = $(parquet_tol), 
                    ODE_tol     = $(ODE_tol),
                    loops       = $(loops),
                    parquet     = $(parquet),
                    Σ_corr      = $(Σ_corr),
                    initial     = $(initial),
                    final       = $(final),
                    bmin        = $(bmin),
                    bmax        = $(bmax),
                    overwrite   = $(overwrite),
                    cps         = $(cps),
                    wt          = $(wt),
                    ct          = $(ct))""")
    end

    return nothing
end

"""
    make_job!(
        path        :: String,
        dir         :: String,
        input       :: String,
        exe         :: String,
        sbatch_args :: Dict{String, String}
        )           :: Nothing

Generate a SLURM job file `path` to run the FRG solver on a cluster node. `dir` is the job working directory.
`input` is the launcher file generated by `save_launcher!`. `exe` is the path to the Julia executable.
`sbatch_args` is used to set SLURM parameters, e.g. `sbatch_args = Dict(["account" => "my_account", "mem" => "8gb"])`.
"""
function make_job!(
    path        :: String,
    dir         :: String,
    input       :: String,
    exe         :: String,
    sbatch_args :: Dict{String, String}
    )           :: Nothing

    # assert that input is a valid Julia script 
    @assert endswith(input, ".jl") "Input must be *.jl file."

    # make local copy to prevent global modification of sbatch_args
    args = copy(sbatch_args)

    # set thread affinity, if not done already
    if haskey(args, "export")
        if occursin("JULIA_EXCLUSIVE", args["export"]) == false
            args["export"] *= ",JULIA_EXCLUSIVE=1"
        end
    else
        push!(args, "export" => "ALL,JULIA_EXCLUSIVE=1")
    end

    # set working directory, if not done already 
    if haskey(args, "chdir")
        @warn "Overwriting working directory passed via SBATCH dict ..."
        args["chdir"] = dir
    else 
        push!(args, "chdir" => dir)
    end

    # set output file, if not done already
    if haskey(args, "output") == false
        output = split(input, ".jl")[1] * ".out"
        push!(args, "output" => output)
    end

    open(path, "w") do file
        # set SLURM parameters
        write(file, "#!/bin/bash \n")

        for arg in keys(args)
            write(file, "#SBATCH --$(arg)=$(args[arg]) \n")
        end

        write(file, "\n")

        # start calculation
        write(file, "$(exe) -O3 -t \$SLURM_CPUS_PER_TASK $(input)")
    end

    return nothing
end

"""
    make_repository!(
        dir         :: String,
        exe         :: String,
        sbatch_args :: Dict{String, String}
        )           :: Nothing

Generate file structure for several runs of the FRG solver with presumably different parameters.
Assumes that the target folder `dir` contains only launcher files generated by `save_launcher!`.
`exe` is the path to the Julia executable. For each launcher file in `dir` a separate folder and job file are created.
`sbatch_args` is used to set SLURM parameters, e.g. `sbatch_args = Dict(["account" => "my_account", "mem" => "8gb"])`.
"""
function make_repository!(
    dir         :: String,
    exe         :: String,
    sbatch_args :: Dict{String, String}
    )           :: Nothing

    # init folder for saving finished calculations
    fin_dir = joinpath(dir, "finished")

    if isdir(fin_dir) == false
        mkdir(fin_dir)
    end

    # for each *.jl file, init a new folder, move the *.jl file into it and create a job file
    for file in readdir(dir)
        if endswith(file, ".jl")
            # buffer paths
            subdir = joinpath(dir, split(file, ".jl")[1])
            input  = joinpath(subdir, file)
            path   = joinpath(subdir, split(file, ".jl")[1] * ".job")

            # create subdir and job file
            mkdir(subdir)
            mv(joinpath(dir, file), input)
            make_job!(path, subdir, file, exe, sbatch_args)
        end
    end

    return nothing
end

"""
    collect_repository!(
        dir :: String
        )   :: Nothing

Collect final results in file structure generated by `make_repository!`.
Finished calculations are moved to the finished folder.
"""
function collect_repository!(
    dir :: String
    )   :: Nothing

    println("Collecting results from repository ...")

    # check that finished folder exists
    @assert isdir(joinpath(dir, "finished")) "Folder $(joinpath(dir, "finished")) does not exist."

    # for each folder move *_obs, *_cp and *.out, then remove their parent dir. If calculation has not finished set overwrite = false in *.jl file
    for file in readdir(dir)
        if isdir(joinpath(dir, file)) && file != "finished"
            # get file list of subdir
            subdir   = joinpath(dir, file)
            subfiles = readdir(subdir)

            # check if output files exist
            obs_filter = filter(x -> endswith(x, "_obs"), subfiles)

            if length(obs_filter) != 1
                @warn "Could not find unique *_obs file in $(subdir), skipping ..."
                continue
            end

            cp_filter = filter(x -> endswith(x, "_cp"), subfiles)

            if length(cp_filter) != 1
                @warn "Could not find unique *_cp file in $(subdir), skipping ..."
                continue
            end

            out_filter = filter(x -> endswith(x, ".out"), subfiles)

            if length(out_filter) != 1
                @warn "Could not find unique *.out file in $(subdir), skipping ..."
                continue
            end

            # buffer names of output files
            obs_name = obs_filter[1]
            cp_name  = cp_filter[1]
            out_name = out_filter[1]

            # buffer paths of output files
            obs_file = joinpath(subdir, obs_name)
            cp_file  = joinpath(subdir, cp_name)
            out_file = joinpath(subdir, out_name)

            # check if calculation is finished
            obs_data = h5open(obs_file, "r")
            cp_data  = h5open(cp_file, "r")

            if haskey(obs_data, "finished") && haskey(cp_data, "finished")
                # close files
                close(obs_data)
                close(cp_data)

                # move files to finished folder
                mv(obs_file, joinpath(joinpath(dir, "finished"), obs_name))
                mv(cp_file,  joinpath(joinpath(dir, "finished"), cp_name))
                mv(out_file, joinpath(joinpath(dir, "finished"), out_name))

                # remove parent dir
                rm(subdir, recursive = true)
            else
                # close files
                close(obs_data)
                close(cp_data)

                # load the launcher
                launcher_file = joinpath(subdir, file * ".jl")
                stream        = open(launcher_file, "r")
                launcher      = read(stream, String)

                if occursin("overwrite   = true", launcher)
                    # replace overwrite flag and overwrite stream
                    launcher = replace(launcher, "overwrite   = true" => "overwrite   = false")
                    stream   = open(launcher_file, "w")
                    write(stream, launcher)

                    # close the stream
                    close(stream)
                elseif occursin("overwrite   = false", launcher)
                    # close the stream
                    close(stream)
                else
                    # close the stream
                    close(stream)

                    # print error message
                    println("Parameter file $(launcher_file) seems to be broken.")
                end
            end
        end
    end

    println("Done. Results collected in $(joinpath(dir, "finished")).")

    return nothing
end





# load launchers for parquet equations and FRG
include("parquet.jl")
include("launcher_1l.jl")
include("launcher_2l.jl")
include("launcher_ml.jl")

"""
    launch!(
        f           :: String,
        name        :: String,
        size        :: Int64,
        model       :: String,
        symmetry    :: String,
        J           :: Vector{<:Any}
        ;
        S           :: Float64            = 0.5,
        β           :: Float64            = 1.0,
        euclidean   :: Bool               = false,
        num_σ       :: Int64              = 25,
        num_Ω       :: Int64              = 15,
        num_ν       :: Int64              = 10,
        num_χ       :: Int64              = 10,
        p_σ         :: NTuple{2, Float64} = (0.3, 1.0),
        p_Ωs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
        p_νs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
        p_Ωt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.3, 50.0),
        p_νt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.5, 50.0),
        p_χ         :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
        lins        :: NTuple{5, Float64} = (5.0, 4.0, 8.0, 6.0, 6.0),
        bounds      :: NTuple{5, Float64} = (1.0, 150.0, 500.0, 250.0, 250.0),
        max_iter    :: Int64              = 10,
        min_eval    :: Int64              = 10,
        max_eval    :: Int64              = 100,
        Σ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
        Γ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
        χ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
        parquet_tol :: NTuple{2, Float64} = (1e-8, 1e-6),
        ODE_tol     :: NTuple{2, Float64} = (1e-8, 1e-2),
        loops       :: Int64              = 1,
        parquet     :: Bool               = false,
        Σ_corr      :: Bool               = true,
        initial     :: Float64            = 50.0,
        final       :: Float64            = 0.05,
        bmin        :: Float64            = 1e-4,
        bmax        :: Float64            = 0.2,
        overwrite   :: Bool               = true,
        cps         :: Vector{Float64}    = Float64[],
        wt          :: Float64            = 23.5,
        ct          :: Float64            = 4.0
        )           :: Nothing

Runs the FRG solver. A detailed explanation of the solver parameters is given below:
* `f`           : name of the output files. The solver will generate two files (`f * "_obs"` and `f * "_cp"`) containing observables and checkpoints respectively.
* `name`        : name of the lattice
* `size`        : size of the lattice. Correlations are truncated beyond this range.
* `model`       : name of the spin model. Defines coupling structure.
* `symmetry`    : symmetry of the spin model. Used to reduce computational complexity.
* `J`           : coupling vector of the spin model. J is normalized during initialization of the solver.
* `S`           : total spin quantum number (only relevant for pure Heisenberg models)
* `β`           : damping factor for fixed point iterations of parquet equations
* `euclidean`   : flag to build lattice by Euclidean (aka real space) instead of bond distance
* `num_σ`       : number of non-zero, positive frequencies for the self energy
* `num_Ω`       : number of non-zero, positive frequencies for the bosonic axis of the two-particle irreducible channels
* `num_ν`       : number of non-zero, positive frequencies for the fermionic axis of the two-particle irreducible channels
* `num_χ`       : number of non-zero, positive frequencies for the correlations (total mesh also includes negative frequencies)
* `p_σ`         : parameters for updating self energy mesh between ODE steps \n
                  p_σ[1] gives the percentage of linearly spaced frequencies
                  p_σ[2] sets the linear extent relative to the position of the quasi-particle peak
* `p_Ω / p_ν`   : parameters for updating bosonic / fermionic s (u) and t channel frequency meshes between ODE steps \n
                  p_Γ[1] gives the percentage of linearly spaced frequencies
                  p_Γ[2] (p_Γ[3]) sets the lower (upper) bound for the accepted relative deviation between the values at the origin and the first finite frequency
                  p_Γ[4] sets the lower bound for the linear spacing in units of the cutoff Λ
                  p_Γ[5] sets the upper bound for the linear extent in units of the cutoff Λ 
* `p_χ`         : parameters for updating correlation mesh between ODE steps \n
                  p_χ[1] gives the percentage of linearly spaced frequencies
                  p_χ[2] (p_χ[3]) sets the lower (upper) bound for the accepted relative deviation between the values at the origin and the first finite frequency
                  p_χ[4] sets the lower bound for the linear spacing in units of the cutoff Λ
                  p_χ[5] sets the upper bound for the linear extent in units of the cutoff Λ 
* `lins`        : parameters for controlling the scaling of frequency meshes before adaptive scanning is utilized \n
                  lins[1] gives the scale, in units of |J|, beyond which adaptive meshes are used
                  lins[2] gives the linear extent, in units of the cutoff Λ, for the self energy
                  lins[3] gives the linear extent, in units of the cutoff Λ, for the bosonic axis of the two-particle irreducible channels
                  lins[4] gives the linear extent, in units of the cutoff Λ, for the fermionic axis of the two-particle irreducible channels
                  lins[5] gives the linear extent, in units of the cutoff Λ, for the correlations
* `bounds`      : parameters for controlling the upper mesh bounds \n
                  bounds[1] gives, in units of |J|, the stopping scale beyond which no further contraction of the meshes is performed
                  bounds[2] gives, in units of the cutoff Λ, the upper bound for the self energy
                  bounds[3] gives, in units of the cutoff Λ, the upper bound for the bosonic axis of the two-particle irreducible channels
                  bounds[4] gives, in units of the cutoff Λ, the upper bound for the fermionic axis of the two-particle irreducible channels
                  bounds[5] gives, in units of the cutoff Λ, the upper bound for the correlations
* `max_iter`    : maximum number of parquet iterations
* `min_eval`    : minimum initial number of subdivisions for vertex quadrature. eval is min_eval for parquet iterations.
* `max_eval`    : maximum initial number of subdivisions for vertex quadrature. eval is ramped up from min_eval to max_eval as a function of the cutoff Λ (for Λ < |J|).
* `Σ_tol`       : absolute and relative error tolerance for self energy quadrature
* `Γ_tol`       : absolute and relative error tolerance for vertex quadrature
* `χ_tol`       : absolute and relative error tolerance for correlation quadrature
* `parquet_tol` : absolute and relative error tolerance for convergence of parquet iterations
* `ODE_tol`     : absolute and relative error tolerance for Bogacki-Shampine solver
* `loops`       : number of loops to be calculated
* `parquet`     : flag to enable parquet iterations. If `false`, initial condition is chosen as bare vertex.
* `Σ_corr`      : flag to enable self energy corrections. Has no effect for 'loops <= 2'.
* `initial`     : start value of the cutoff in units of |J|
* `final`       : final value of the cutoff in units of |J|. If `final = initial` and `parquet = true` only a solution of the parquet equations is computed.
* `bmin`        : minimum step size of the ODE solver in units of |J|
* `bmax`        : maximum step size of the ODE solver in units of Λ
* `overwrite`   : flag to indicate whether a new calculation should be started. If false, checks if `f * "_obs"` and `f * "_cp"` exist and continues calculation from available checkpoint with lowest cutoff.
* `cps`         : list of intermediate cutoffs in units of |J|, where a checkpoint with full vertex data shall be generated
* `wt`          : wall time (in hours) for the calculation. Should be set according to cluster configurations. If run remote, set `wt = Inf` to avoid data loss. \n
                  WARNING: For run times longer than wt, no checkpoints are created.
* `ct`          : minimum time (in hours) between subsequent checkpoints
"""
function launch!(
    f           :: String,
    name        :: String,
    size        :: Int64,
    model       :: String,
    symmetry    :: String,
    J           :: Vector{<:Any}
    ;
    S           :: Float64            = 0.5,
    β           :: Float64            = 1.0,
    euclidean   :: Bool               = false,
    num_σ       :: Int64              = 25,
    num_Ω       :: Int64              = 15,
    num_ν       :: Int64              = 10,
    num_χ       :: Int64              = 10,
    p_σ         :: NTuple{2, Float64} = (0.3, 1.0),
    p_Ωs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
    p_νs        :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
    p_Ωt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.3, 50.0),
    p_νt        :: NTuple{5, Float64} = (0.3, 0.15, 0.20, 0.5, 50.0),
    p_χ         :: NTuple{5, Float64} = (0.3, 0.05, 0.10, 0.1, 50.0),
    lins        :: NTuple{5, Float64} = (5.0, 4.0, 8.0, 6.0, 6.0),
    bounds      :: NTuple{5, Float64} = (1.0, 150.0, 500.0, 250.0, 250.0),
    max_iter    :: Int64              = 10,
    min_eval    :: Int64              = 10,
    max_eval    :: Int64              = 100,
    Σ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
    Γ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
    χ_tol       :: NTuple{2, Float64} = (1e-8, 1e-3),
    parquet_tol :: NTuple{2, Float64} = (1e-8, 1e-6),
    ODE_tol     :: NTuple{2, Float64} = (1e-8, 1e-2),
    loops       :: Int64              = 1,
    parquet     :: Bool               = false,
    Σ_corr      :: Bool               = true,
    initial     :: Float64            = 50.0,
    final       :: Float64            = 0.05,
    bmin        :: Float64            = 1e-4,
    bmax        :: Float64            = 0.2,
    overwrite   :: Bool               = true,
    cps         :: Vector{Float64}    = Float64[],
    wt          :: Float64            = 23.5,
    ct          :: Float64            = 4.0
    )           :: Nothing

    # init timers for checkpointing
    t  = Dates.now()
    t0 = Dates.now()

    println()
    println("################################################################################")
    println("Initializing solver ...")
    println(); println()

    # check if symmetry parameter is valid
    symmetries = String["su2", "u1-dm"]
    @assert in(symmetry, symmetries) "Symmetry $(symmetry) unknown. Valid arguments are su2 and u1-dm."

    # init names for observables and checkpoints file
    obs_file = f * "_obs"
    cp_file  = f * "_cp"

    # test if a new calculation should be started
    if overwrite
        println("overwrite = true, starting from scratch ...")

        # delete existing observables
        if isfile(obs_file)
            rm(obs_file)
        end

        # delete existing checkpoints
        if isfile(cp_file)
            rm(cp_file)
        end

        # open new files
        obs = h5open(obs_file, "cw")
        cp  = h5open(cp_file, "cw")

        # convert J for type safety
        J = Vector{Vector{Float64}}([[x...] for x in J])

        # normalize couplings
        normalize!(J)

        # build lattice and save to files
        println();
        l = get_lattice(name, size, euclidean = euclidean)

        println();
        r = get_reduced_lattice(model, J, l)

        save!(obs, r)
        save!(cp, r)

        # close files
        close(obs)
        close(cp)

        # build meshes
        σ  = get_mesh(lins[2] * initial, bounds[2] * max(initial, bounds[1]), num_σ, p_σ[1])
        Ωs = get_mesh(lins[3] * initial, bounds[3] * max(initial, bounds[1]), num_Ω, p_Ωs[1])
        νs = get_mesh(lins[4] * initial, bounds[4] * max(initial, bounds[1]), num_ν, p_νs[1])
        Ωt = get_mesh(lins[3] * initial, bounds[3] * max(initial, bounds[1]), num_Ω, p_Ωt[1])
        νt = get_mesh(lins[4] * initial, bounds[4] * max(initial, bounds[1]), num_ν, p_νt[1])
        χ  = get_mesh(lins[5] * initial, bounds[5] * max(initial, bounds[1]), num_χ, p_χ[1])
        χ  = sort(vcat(-1.0 .* χ[2 : end], χ))
        m  = Mesh(num_σ + 1, num_Ω + 1, num_ν + 1, 2 * num_χ + 1, σ, Ωs, νs, Ωt, νt, χ)

        # build action
        a = get_action_empty(symmetry, r, m, S = S)
        init_action!(l, r, a)

        # initialize by parquet iterations
        if parquet
            println(); println()
            println("Warming up with some parquet iterations ...")
            flush(stdout)
            launch_parquet!(obs_file, cp_file, symmetry, l, r, m, a, initial, bmax * initial, β, max_iter, min_eval, Σ_tol, Γ_tol, χ_tol, parquet_tol, S = S)
            println("Done. Action is initialized with parquet solution.")
        end

        println(); println()
        println("Solver is ready.")
        println("################################################################################")
        println()

        # start calculation
        println("Renormalization group flow with ℓ = $(loops) ...")
        flush(stdout)

        if loops == 1
            launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ωs, p_νs, p_Ωt, p_νt, p_χ, lins, bounds, initial, final, bmax * initial, bmin, bmax, min_eval, max_eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, cps, wt, ct, S = S)
        elseif loops == 2
            launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ωs, p_νs, p_Ωt, p_νt, p_χ, lins, bounds, initial, final, bmax * initial, bmin, bmax, min_eval, max_eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, cps, wt, ct, S = S)
        elseif loops >= 3
            launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ωs, p_νs, p_Ωt, p_νt, p_χ, lins, bounds, loops, Σ_corr, initial, final, bmax * initial, bmin, bmax, min_eval, max_eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, cps, wt, ct, S = S)
        end
    else
        println("overwrite = false, trying to load data ...")

        if isfile(obs_file) && isfile(cp_file)
            println()
            println("Found existing output files, checking status ...")

            # open files
            obs = h5open(obs_file, "cw")
            cp  = h5open(cp_file, "cw")

            if haskey(obs, "finished") && haskey(cp, "finished")
                # close files
                close(obs)
                close(cp)

                println(); println()
                println("Calculation has finished already.")
                println("################################################################################")
                flush(stdout)
            else
                println()
                println("Final Λ has not been reached, resuming calculation ...")

                # load data
                l, r        = read_lattice(cp)
                Λ, dΛ, m, a = read_checkpoint(cp, 0.0)

                # close files
                close(obs)
                close(cp)

                println(); println()
                println("Solver is ready.")
                println("################################################################################")
                println()

                # resume calculation
                println("Renormalization group flow with ℓ = $(loops) ...")
                flush(stdout)

                if loops == 1
                    launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ωs, p_νs, p_Ωt, p_νt, p_χ, lins, bounds, Λ, final, dΛ, bmin, bmax, min_eval, max_eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, cps, wt, ct, S = S)
                elseif loops == 2
                    launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ωs, p_νs, p_Ωt, p_νt, p_χ, lins, bounds, Λ, final, dΛ, bmin, bmax, min_eval, max_eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, cps, wt, ct, S = S)
                elseif loops >= 3
                    launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ωs, p_νs, p_Ωt, p_νt, p_χ, lins, bounds, loops, Σ_corr, Λ, final, dΛ, bmin, bmax, min_eval, max_eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, cps, wt, ct, S = S)
                end
            end
        else
            println(); println()
            println("Found no existing output files, terminating solver ...")
            println("################################################################################")
            flush(stdout)
        end
    end

    println()
    println("################################################################################")
    println("Solver terminated.")
    println("################################################################################")
    flush(stdout)

    return nothing
end