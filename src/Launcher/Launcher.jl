# function for measurements and checkpointing
function measure(
    symmetry :: String,
    obs_file :: String,
    cp_file  :: String,
    Λ        :: Float64,
    dΛ       :: Float64,
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

    # init flag for monotonicity
    monotone = true

    # save observables if dataset does not yet exist (can happen due to checkpointing) and check for monotonicity
    if haskey(obs, "χ/$(Λ)") == false
        # compute observables and save to file
        χ = compute_χ(Λ, r, m, a, χ_tol)
        save_χ!(obs, Λ, symmetry, χ)
        save_self!(obs, Λ, m, a)

        # load correlations from previous step
        cutoffs = sort(parse.(Float64, keys(obs["χ"])))
        index   = min(argmin(abs.(cutoffs .- Λ)) + 1, length(cutoffs))
        χp      = read_χ_all(obs, cutoffs[index], verbose = false)

        # check for monotonicity of dominant correlation
        idx = argmax(Float64[χ[i][1] for i in eachindex(χ)])

        if χ[idx][1] / χp[idx][1] < 0.995
            monotone = false 
        end
    end

    # compute current run time (in hours)
    h0 = 1e-3 * (Dates.now() - t0).value / 3600.0

    # if more than half an hour is left to the wall time limit, use ct as timer heuristic for checkpointing
    if wt - h0 > 0.5
        # test if time limit for checkpoint (in hours) has been reached
        h = 1e-3 * (Dates.now() - t).value / 3600.0

        if h >= ct
            # generate checkpoint if it does not exist yet
            if haskey(cp, "a/$(Λ)") == false
                println()
                println("Generating timed checkpoint at cutoff Λ / |J| = $(Λ) ...")
                checkpoint!(cp, Λ, dΛ, m, a)
                println("Successfully generated checkpoint.")
                println()
            end

            # reset timer
            t = Dates.now()
        end
    # if less than half an hour is left to the wall time, generate checkpoints as if ct = 0. Lower bound to prevent cancellation during checkpoint writing
    elseif 0.1 < wt - h0 <= 0.5
        # generate checkpoint if it does not exist yet
        if haskey(cp, "a/$(Λ)") == false
            println()
            println("Generating forced checkpoint at cutoff Λ / |J| = $(Λ) ...")
            checkpoint!(cp, Λ, dΛ, m, a)
            println("Successfully generated checkpoint.")
            println()
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
        num_σ       :: Int64              = 50,
        num_Ω       :: Int64              = 15,
        num_ν       :: Int64              = 10,
        p_σ         :: NTuple{4, Float64} = (0.5, 1.2, 1e-4, 250.0),
        p_Ω         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4, 150.0),
        p_ν         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4,  75.0),
        max_iter    :: Int64              = 10,
        eval        :: Int64              = 15,
        Σ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
        Γ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
        χ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
        parquet_tol :: NTuple{2, Float64} = (1e-5, 1e-3),
        ODE_tol     :: NTuple{2, Float64} = (1e-5, 1e-3),
        loops       :: Int64              = 1,
        parquet     :: Bool               = false,
        Σ_corr      :: Bool               = true,
        initial     :: Float64            = 50.0,
        final       :: Float64            = 0.05,
        bmin        :: Float64            = 1e-4,
        bmax        :: Float64            = 0.15,
        overwrite   :: Bool               = true,
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
    num_σ       :: Int64              = 50,
    num_Ω       :: Int64              = 15,
    num_ν       :: Int64              = 10,
    p_σ         :: NTuple{4, Float64} = (0.5, 1.2, 1e-4, 250.0),
    p_Ω         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4, 150.0),
    p_ν         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4,  75.0),
    max_iter    :: Int64              = 10,
    eval        :: Int64              = 15,
    Σ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
    Γ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
    χ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
    parquet_tol :: NTuple{2, Float64} = (1e-5, 1e-3),
    ODE_tol     :: NTuple{2, Float64} = (1e-5, 1e-3),
    loops       :: Int64              = 1,
    parquet     :: Bool               = false,
    Σ_corr      :: Bool               = true,
    initial     :: Float64            = 50.0,
    final       :: Float64            = 0.05,
    bmin        :: Float64            = 1e-4,
    bmax        :: Float64            = 0.15,
    overwrite   :: Bool               = true,
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
                    num_σ       = $(num_σ),
                    num_Ω       = $(num_Ω),
                    num_ν       = $(num_ν),
                    p_σ         = $(p_σ),
                    p_Ω         = $(p_Ω),
                    p_ν         = $(p_ν),
                    max_iter    = $(max_iter),
                    eval        = $(eval),
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
        num_σ       :: Int64              = 50,
        num_Ω       :: Int64              = 15,
        num_ν       :: Int64              = 10,
        p_σ         :: NTuple{4, Float64} = (0.5, 1.2, 1e-4, 250.0),
        p_Ω         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4, 150.0),
        p_ν         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4,  75.0),
        max_iter    :: Int64              = 10,
        eval        :: Int64              = 15,
        Σ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
        Γ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
        χ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
        parquet_tol :: NTuple{2, Float64} = (1e-5, 1e-3),
        ODE_tol     :: NTuple{2, Float64} = (1e-5, 1e-3),
        loops       :: Int64              = 1,
        parquet     :: Bool               = false,
        Σ_corr      :: Bool               = true,
        initial     :: Float64            = 50.0,
        final       :: Float64            = 0.05,
        bmin        :: Float64            = 1e-4,
        bmax        :: Float64            = 0.15,
        overwrite   :: Bool               = true,
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
* `num_σ`       : number of non-zero, positive frequencies for the self energy
* `num_Ω`       : number of non-zero, positive frequencies for the bosonic axis of the two-particle irreducible channels
* `num_ν`       : number of non-zero, positive frequencies for the fermionic axis of the two-particle irreducible channels
* `p_σ`         : parameters for updating self energy frequency mesh between ODE steps \n
                  p[1] gives the percentage of linearly spaced frequencies.
                  p[2] sets the linear extent relative to the position of the maximum.
                  p[3] sets the upper bound relative to the value of the maximum.
                  p[4] sets the lower bound for the upper mesh bound in units of the cutoff Λ.
* `p_Ω / p_ν`   : parameters for updating channel frequency meshes between ODE steps \n
                  p[1] gives the percentage of linearly spaced frequencies.
                  p[2] (p[3]) sets the lower (upper) bound for the accepted relative deviation of the value at the first finite frequency and the origin.
                  p[4] sets the linear extent relative to the position of the maximum.
                  p[5] (p[6]) sets the lower (upper) bound for the linear spacing in units of the cutoff Λ.
                  p[7] sets the upper bound relative to the value of the maximum.
                  p[8] sets the lower bound for the upper mesh bound in units of the cutoff Λ.
* `max_iter`    : maximum number of parquet iterations
* `eval`        : initial number of subdivisions for vertex quadrature. Lower number means loss of accuracy, higher will lead to increased runtimes.
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
    num_σ       :: Int64              = 50,
    num_Ω       :: Int64              = 15,
    num_ν       :: Int64              = 10,
    p_σ         :: NTuple{4, Float64} = (0.5, 1.2, 1e-4, 250.0),
    p_Ω         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4, 150.0),
    p_ν         :: NTuple{8, Float64} = (0.5, 0.15, 0.20, 1.2, 0.05, 2.0, 1e-4,  75.0),
    max_iter    :: Int64              = 10,
    eval        :: Int64              = 15,
    Σ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
    Γ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
    χ_tol       :: NTuple{2, Float64} = (1e-5, 1e-3),
    parquet_tol :: NTuple{2, Float64} = (1e-5, 1e-3),
    ODE_tol     :: NTuple{2, Float64} = (1e-5, 1e-3),
    loops       :: Int64              = 1,
    parquet     :: Bool               = false,
    Σ_corr      :: Bool               = true,
    initial     :: Float64            = 50.0,
    final       :: Float64            = 0.05,
    bmin        :: Float64            = 1e-4,
    bmax        :: Float64            = 0.15,
    overwrite   :: Bool               = true,
    wt          :: Float64            = 23.5,
    ct          :: Float64            = 4.0
    )           :: Nothing

    # init timers for checkpointing
    t  = Dates.now()
    t0 = Dates.now()

    println()
    println("################################################################################")
    println("Initializing solver ...")
    println()

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
        println()
        l = get_lattice(name, size)

        println()
        r = get_reduced_lattice(model, J, l)

        save!(obs, r)
        save!(cp, r)

        # close files
        close(obs)
        close(cp)

        # build meshes
        m = get_mesh(symmetry, initial, num_σ, num_Ω, num_ν, p_σ[1], p_Ω[1], p_ν[1])

        # build action
        a = get_action_empty(symmetry, r, m, S = S)
        init_action!(l, r, a)

        # initialize by parquet iterations
        if parquet
            println()
            println("Warming up with some parquet iterations ...")
            flush(stdout)
            launch_parquet!(obs_file, cp_file, symmetry, l, r, m, a, initial, bmax * initial, β, max_iter, eval, Σ_tol, Γ_tol, χ_tol, parquet_tol, S = S)
            println("Done. Action is initialized with parquet solution.")
        end

        println()
        println("Solver is ready.")
        println("################################################################################")
        println()

        # start calculation
        println("Renormalization group flow with ℓ = $(loops) ...")
        flush(stdout)

        if loops == 1
            launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ω, p_ν, initial, final, bmax * initial, bmin, bmax, eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, wt, ct, S = S)
        elseif loops == 2
            launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ω, p_ν, initial, final, bmax * initial, bmin, bmax, eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, wt, ct, S = S)
        elseif loops >= 3
            launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ω, p_ν, loops, Σ_corr, initial, final, bmax * initial, bmin, bmax, eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, wt, ct, S = S)
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

                println()
                println("Calculation has finished already.")
                println("################################################################################")
                flush(stdout)
            else
                println()
                println("Final Λ has not been reached, resuming calculation ...")

                # load data
                println()
                println("Loading data ...")
                println()
                l, r        = read_lattice(cp)
                Λ, dΛ, m, a = read_checkpoint(cp, 0.0)

                # close files
                close(obs)
                close(cp)

                println()
                println("Solver is ready.")
                println("################################################################################")
                println()

                # resume calculation
                println("Renormalization group flow with ℓ = $(loops) ...")
                flush(stdout)

                if loops == 1
                    launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ω, p_ν, Λ, final, dΛ, bmin, bmax, eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, wt, ct, S = S)
                elseif loops == 2
                    launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ω, p_ν, Λ, final, dΛ, bmin, bmax, eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, wt, ct, S = S)
                elseif loops >= 3
                    launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, p_σ, p_Ω, p_ν, loops, Σ_corr, Λ, final, dΛ, bmin, bmax, eval, Σ_tol, Γ_tol, χ_tol, ODE_tol, t, t0, wt, ct, S = S)
                end
            end
        else
            println()
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