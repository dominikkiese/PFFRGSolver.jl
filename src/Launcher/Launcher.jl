# function for measurements and checkpointing
function measure(
    symmetry :: String,
    obs_file :: String,
    cp_file  :: String,
    Λ        :: Float64,
    dΛ       :: Float64,
    t        :: DateTime,
    t0       :: DateTime,
    r        :: reduced_lattice,
    m        :: mesh,
    a        :: action,
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
        χ = compute_χ(Λ, r, m, a)
        save_χ!(obs, Λ, symmetry, χ)
        save_self!(obs, Λ, m, a)

        # load correlations from previous step
        cutoffs = sort(parse.(Float64, keys(obs["χ"])))
        index   = min(argmin(abs.(cutoffs .- Λ)) + 1, length(cutoffs))
        labels  = keys(obs["χ/$(cutoffs[index])"]) 
        χp      = Vector{Float64}[read(obs, "χ/$(cutoffs[index])/" * label) for label in labels]

        # check for monotonicity
        for i in eachindex(χ)
            if χp[i][1] - χ[i][1] > 1e-3
                monotone = false 
                break
            end 
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
                println("Generating timed checkpoint at cutoff Λ / |J| = $(Λ / get_scale(a)) ...")
                checkpoint!(cp, Λ, dΛ, m, a)
                println("Successfully generated checkpoint.")
                println()
            end 

            # reset timer
            t = Dates.now()
        end
    # if less than half an hour is left to the wall time, generate checkpoints as if ct = 0
    elseif 0.2 < wt - h0 <= 0.5
        # generate checkpoint if it does not exist yet 
        if haskey(cp, "a/$(Λ)") == false
            println()
            println("Generating forced checkpoint at cutoff Λ / |J| = $(Λ / get_scale(a)) ...")
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
        path      :: String,
        src       :: String,
        f         :: String,
        name      :: String,
        size      :: Int64,
        model     :: String,
        symmetry  :: String,
        J         :: Vector{<:Any}
        ;
        S         :: Float64            = 0.5,
        N         :: Float64            = 2.0,
        β         :: Float64            = 1.0,
        num_σ     :: Int64              = 50,
        num_Ω     :: Int64              = 15,
        num_ν     :: Int64              = 10,
        p         :: NTuple{5, Float64} = (0.4, 0.05, 0.1, 0.05, 2.0),
        max_iter  :: Int64              = 50,
        eval      :: Int64              = 50,
        loops     :: Int64              = 1,
        parquet   :: Bool               = true, 
        Σ_corr    :: Bool               = true,
        initial   :: Float64            = 5.0,
        final     :: Float64            = 1.0,
        bmin      :: Float64            = 1e-4,
        bmax      :: Float64            = 0.1,
        overwrite :: Bool               = true,
        wt        :: Float64            = 24.0,
        ct        :: Float64            = 1.0
        )         :: Nothing

Generate executable Julia file `path` which sets up and runs the FRG solver.
`src` is the location of the include file of PFFRG.jl.
For more information on the different solver parameters see documentation of `launch!`.
"""
function save_launcher!(
    path      :: String,
    src       :: String,
    f         :: String,
    name      :: String,
    size      :: Int64,
    model     :: String,
    symmetry  :: String,
    J         :: Vector{<:Any}
    ;
    S         :: Float64            = 0.5,
    N         :: Float64            = 2.0,
    β         :: Float64            = 1.0,
    num_σ     :: Int64              = 50,
    num_Ω     :: Int64              = 15,
    num_ν     :: Int64              = 10,
    p         :: NTuple{5, Float64} = (0.4, 0.05, 0.1, 0.05, 2.0),
    max_iter  :: Int64              = 50,
    eval      :: Int64              = 50,
    loops     :: Int64              = 1,
    parquet   :: Bool               = true, 
    Σ_corr    :: Bool               = true,
    initial   :: Float64            = 5.0,
    final     :: Float64            = 1.0,
    bmin      :: Float64            = 1e-4,
    bmax      :: Float64            = 0.1,
    overwrite :: Bool               = true,
    wt        :: Float64            = 24.0,
    ct        :: Float64            = 1.0
    )         :: Nothing

    # convert J for type safety
    J = Array{Array{Float64,1},1}([[x...] for x in J])

    open(path, "w") do file
        # load source code
        write(file, "dir = @__DIR__ \n")
        write(file, """include("$(src)") \n""")
        write(file, "cd(dir) \n \n")

        # setup for launcher function
        write(file, """launch!(joinpath(dir, "$(f)"),
                    "$(name)",
                    $(size),
                    "$(model)",
                    "$(symmetry)",
                    $(J),
                    S         = $(S),
                    N         = $(N),
                    β         = $(β),
                    num_σ     = $(num_σ),
                    num_Ω     = $(num_Ω),
                    num_ν     = $(num_ν),
                    p         = $(p),
                    max_iter  = $(max_iter),
                    eval      = $(eval),
                    loops     = $(loops),
                    parquet   = $(parquet), 
                    Σ_corr    = $(Σ_corr),
                    initial   = $(initial),
                    final     = $(final),
                    bmax      = $(bmax),
                    bmin      = $(bmin),
                    overwrite = $(overwrite),
                    wt        = $(wt),
                    ct        = $(ct)) """)
    end

    return nothing
end

"""
    function make_job!(
        path          :: String,
        dir           :: String,
        input         :: String,
        exe           :: String,
        account       :: String,
        cpus_per_task :: Int64,
        mem           :: String,
        time          :: String,
        partition     :: String,
        output        :: String
        ;
        pinning       :: Bool = true
        )             :: Nothing

Generate a SLURM job file `path` to run the FRG solver on a cluster node. `dir` is the job working directory.
`input` is the launcher file generated by `save_launcher!`. `exe` is the path to the Julia executable.
`pinning` is a flag to enable processor pinning via numactl.
"""
function make_job!(
    path          :: String,
    dir           :: String,
    input         :: String,
    exe           :: String,
    account       :: String,
    cpus_per_task :: Int64,
    mem           :: String,
    time          :: String,
    partition     :: String,
    output        :: String
    ;
    pinning       :: Bool = true
    )             :: Nothing

    open(path, "w") do file
        # set Slurm parameters
        write(file, "#!/bin/bash -l \n")
        write(file, "#SBATCH --account=$(account) \n")
        write(file, "#SBATCH --nodes=1 \n")
        write(file, "#SBATCH --ntasks=1 \n")
        write(file, "#SBATCH --ntasks-per-node=1 \n")
        write(file, "#SBATCH --cpus-per-task=$(cpus_per_task) \n")
        write(file, "#SBATCH --mem=$(mem) \n")
        write(file, "#SBATCH --time=$(time) \n")
        write(file, "#SBATCH --partition=$(partition) \n")
        write(file, "#SBATCH --output=$(output) \n \n")

        # set number of threads
        write(file, "export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK \n")

        # go to working directory
        write(file, "cd $(dir) \n")

        # start calculation (thread pinning via numactl)
        if pinning
            write(file, "numactl --physcpubind=0-$(cpus_per_task - 1) -- $(exe) -O3 $(input) -E 'run(`numactl -s`)'")
        else 
            write(file, "$(exe) -O3 $(input)")
        end
    end

    return nothing
end

"""
    make_repository!(
        dir            :: String,
        exe            :: String,
        account        :: String,
        cpus_per_task  :: Int64,
        mem            :: String,
        time           :: String,
        partition      :: String
        ;
        pinning        :: Bool = true
        )              :: Nothing

Generate file structure for several runs of the FRG solver with presumably different parameters.
Assumes that the target folder `dir` contains only launcher files generated by `save_launcher!`.
`exe` is the path to the Julia executable. For each launcher file in `dir` a separate folder and job file are created.
`pinning` is a flag to regulate processor pinning via numactl.
The jobs can then e.g. be submitted from within `dir` with the bash command `for folder in */; if [[\$folder != finished]]; do cd \$folder; sbatch *.job; cd ..; fi; done`.
"""
function make_repository!(
    dir            :: String,
    exe            :: String,
    account        :: String,
    cpus_per_task  :: Int64,
    mem            :: String,
    time           :: String,
    partition      :: String
    ;
    pinning        :: Bool = true
    )              :: Nothing

    # init folder for saving finished calculations
    fin_dir = joinpath(dir, "finished")
    
    if isdir(fin_dir) == false
        mkdir(fin_dir)
    end

    # for each *.jl file, init a new folder, move the *.jl file into it and create a job file
    for file in readdir(dir)
        if endswith(file, ".jl")
            subdir = joinpath(dir, split(file, ".jl")[1])
            input  = joinpath(subdir, file)
            path   = joinpath(subdir, split(file, ".jl")[1] * ".job")
            output = split(file, ".jl")[1] * ".out"

            mkdir(subdir)
            mv(joinpath(dir, file), input)
            make_job!(path, subdir, file, exe, account, cpus_per_task, mem, time, partition, output, pinning = pinning)
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
Unfinished jobs can then be resubmitted.
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
            # buffer names of output files
            obs_name = file * "_obs"
            cp_name  = file * "_cp"
            out_name = file * ".out"

            # buffer paths of output files and parent dir
            subdir   = joinpath(dir, file)
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

                if occursin("overwrite = true", launcher)
                    # replace overwrite flag and overwrite stream
                    launcher = replace(launcher, "overwrite = true" => "overwrite = false")
                    stream   = open(launcher_file, "w")
                    write(stream, launcher)

                    # close the stream
                    close(stream)
                elseif occursin("overwrite = false", launcher)
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
    function launch!(
        f         :: String,
        name      :: String,
        size      :: Int64,
        model     :: String,
        symmetry  :: String,
        J         :: Vector{<:Any}
        ;
        S         :: Float64            = 0.5,
        N         :: Float64            = 2.0,
        β         :: Float64            = 1.0,
        num_σ     :: Int64              = 50,
        num_Ω     :: Int64              = 15,
        num_ν     :: Int64              = 10,
        p         :: NTuple{5, Float64} = (0.4, 0.05, 0.1, 0.05, 2.0),
        max_iter  :: Int64              = 50,
        eval      :: Int64              = 50,
        loops     :: Int64              = 1,
        parquet   :: Bool               = true, 
        Σ_corr    :: Bool               = true,
        initial   :: Float64            = 5.0,
        final     :: Float64            = 1.0,
        bmin      :: Float64            = 1e-4,
        bmax      :: Float64            = 0.1,
        overwrite :: Bool               = true,
        wt        :: Float64            = 24.0,
        ct        :: Float64            = 1.0
        )         :: Nothing

Runs the FRG solver. A detailed explanation of the solver parameters is given below:
* `f`         : name of the output files. The solver will generate two files (`f * "_obs"` and `f * "_cp"`) containing observables and checkpoints respectively.
* `name`      : name of the lattice
* `size`      : size of the lattice. Correlations are truncated beyond this range.
* `model`     : name of the spin model. Defines overwrite for bond matrix of lattice struct.
* `symmetry`  : symmetry of the spin model. The solver will link the symmetry with the corresponding FRG kernel.
* `J`         : coupling vector of the spin model. The solver parses `J` into `Vector{Vector{Float64}}` if possible and interprets it according to the chosen model.
* `S`         : total spin quantum number
* `β`         : damping factor for fixed point iterations of parquet equations (`0.0 < β <= 1.0`)
* `num_σ`     : number of non-zero, positive frequencies for the self energy
* `num_Ω`     : number of non-zero, positive frequencies for the bosonic axis of the two-particle irreducible channels
* `num_ν`     : number of non-zero, positive frequencies for the fermionic axis of the two-particle irreducible channels
* `p`         : parameters for updating frequency meshes between ODE steps. \n 
                p[1] gives the percentage of linearly spaced frequencies (0.0 < p[1] < 1.0).
                p[2] (p[3]) sets the lower (upper) bound for the accepted relative deviation of the first finite frequency to the origin (0.0 < p[2] < p[3] < 0.25). 
                p[4] (p[5]) sets the lower (upper) bound for the linear spacing in units of the cutoff Λ (0.0 < p[4] < p[5] < 3.0).
* `max_iter`  : maximum number of parquet iterations 
* `eval`      : number of subdomains for adaptive quadrature routine (`20 <= eval <= 100`)
* `loops`     : number of loops to be calculated
* `parquet`   : flag to turn off parquet iterations. If `false`, initial condition is chosen as bare vertex. 
* `Σ_corr`    : flag to turn off self energy corrections when `loops >= 3`
* `initial`   : initial value of the cutoff in units of `norm(J)`
* `final`     : target value fo the cutoff in units of `norm(J)`. If `final = initial` and `parquet = true` a pure solution of the parquet equations is computed.
* `bmin`      : minimum step size of the ODE solver in units of `norm(J)`
* `bmax`      : maximum step size of the ODE solver in units of `Λ`
* `overwrite` : flag to indicate whether a new calculation should be started. If false, checks if `f * "_obs"` and `f * "_cp"` exist and continues calculation from available checkpoint with lowest cutoff.
* `wt`        : wall time (in hours) for the calculation. Should be set according to cluster configurations. If run remote, set `wt = Inf` to avoid data loss. \n 
                WARNING: For run times longer than wt, no checkpoints are created.
* `ct`        : time limit (in hours) for checkpoint creation
"""
function launch!(
    f         :: String,
    name      :: String,
    size      :: Int64,
    model     :: String,
    symmetry  :: String,
    J         :: Vector{<:Any}
    ;
    S         :: Float64            = 0.5,
    N         :: Float64            = 2.0,
    β         :: Float64            = 1.0,
    num_σ     :: Int64              = 50,
    num_Ω     :: Int64              = 15,
    num_ν     :: Int64              = 10,
    p         :: NTuple{5, Float64} = (0.4, 0.05, 0.1, 0.05, 2.0),
    max_iter  :: Int64              = 50,
    eval      :: Int64              = 50,
    loops     :: Int64              = 1,
    parquet   :: Bool               = true, 
    Σ_corr    :: Bool               = true,
    initial   :: Float64            = 5.0,
    final     :: Float64            = 1.0,
    bmin      :: Float64            = 1e-4,
    bmax      :: Float64            = 0.1,
    overwrite :: Bool               = true,
    wt        :: Float64            = 24.0,
    ct        :: Float64            = 1.0
    )         :: Nothing

    # only allow N = 2, since we have different symmetries (which are currently not implemented) for N > 2
    @assert N == 2.0 "N != 2 is currently not supported."

    println()
    println("#------------------------------------------------------------------------------------------------------#")
    println("Initializing solver ...")
    println()

    # init names for observables and checkpoints file
    obs_file = f * "_obs"
    cp_file  = f * "_cp"

    # normalize flow parameter with couplings
    Z        = norm(J)
    initial *= Z
    final   *= Z
    
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
        J = Array{Array{Float64,1},1}([[x...] for x in J])

        # build lattice and save to files
        println()
        l = get_lattice(name, size)
        init_model!(model, J, l)
        println()
        r = get_reduced_lattice(l)
        save!(obs, l)
        save!(obs, r)
        save!(cp, l)
        save!(cp, r)

        # close files
        close(obs)
        close(cp)

        # set reference scale for upper mesh bound
        Λ_ref = max(initial, 0.5 * Z)

        # build meshes
        σ = get_mesh(5.0 * initial, 250.0 * Λ_ref, num_σ, p[1])
        Ω = get_mesh(5.0 * initial, 150.0 * Λ_ref, num_Ω, p[1])
        ν = get_mesh(5.0 * initial,  75.0 * Λ_ref, num_ν, p[1])
        m = mesh(num_σ + 1, num_Ω + 1, num_ν + 1, σ, Ω, ν, Ω, ν, Ω, ν)

        # build action
        a = get_action_empty(symmetry, r, m, S = S, N = N)
        init_action!(l, r, a)

        # initialize by parquet iterations
        if parquet
            println()
            println("Warming up with some parquet iterations ...")
            launch_parquet!(obs_file, cp_file, symmetry, l, r, m, a, initial, bmax * initial, β, max_iter, eval, S = S, N = N)
            println("Done. Action is initialized with parquet solution.")
        end

        println()
        println("Solver is ready.")
        println("#------------------------------------------------------------------------------------------------------#")
        println()

        # start calculation
        println("Renormalization group flow with ℓ = $(loops) ...")

        if loops == 1
            launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, p, initial, final, bmax * initial, bmin, bmax, eval, wt, ct, S = S, N = N)
        elseif loops == 2
            launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, p, initial, final, bmax * initial, bmin, bmax, eval, wt, ct, S = S, N = N)
        elseif loops >= 3
            launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, p, loops, Σ_corr, initial, final, bmax * initial, bmin, bmax, eval, wt, ct, S = S, N = N)
        end
    else
        println("overwrite = false, trying to load data ...")

        if isfile(obs_file) && isfile(cp_file)
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
                println("#------------------------------------------------------------------------------------------------------#")
            else
                println("Final Λ has not been reached, resuming calculation ...")

                # load data
                println("Loading data ...")
                l           = read_lattice(cp)
                r           = read_reduced_lattice(cp)
                Λ, dΛ, m, a = read_checkpoint(cp, symmetry, 0.0)

                # close files
                close(obs)
                close(cp)

                println()
                println("Solver is ready.")
                println("#------------------------------------------------------------------------------------------------------#")
                println()

                # resume calculation
                println("Renormalization group flow with ℓ = $(loops) ...")

                if loops == 1
                    launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, p, Λ, final, dΛ, bmin, bmax, eval, wt, ct, S = S, N = N)
                elseif loops == 2
                    launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, p, Λ, final, dΛ, bmin, bmax, eval, wt, ct, S = S, N = N)
                elseif loops >= 3
                    launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, p, loops, Σ_corr, Λ, final, dΛ, bmin, bmax, eval, wt, ct, S = S, N = N)
                end
            end 
        else 
            println()
            println("Found no existing output files, terminating solver ...")
            println("#------------------------------------------------------------------------------------------------------#")
        end
    end

    println()
    println("#------------------------------------------------------------------------------------------------------#")
    println("Solver terminated.")
    println("#------------------------------------------------------------------------------------------------------#")
    println()

    return nothing
end

# include tests and timers 
include("test.jl")
include("timers.jl")
