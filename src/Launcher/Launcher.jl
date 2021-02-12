# do measurements and checkpointing 
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
    )        :: DateTime 

    # open files 
    obs = h5open(obs_file, "cw")
    cp  = h5open(cp_file, "cw")

    # save observables if dataset does not yet exist (can happen due to checkpointing)
    if haskey(obs, "χ/$(Λ)") == false 
        χ = compute_χ(Λ, r, m, a)
        save_χ!(obs, Λ, symmetry, χ)
        save_self!(obs, Λ, m, a)
    end

    # test if enough time remains to wall time (in hours)
    h0 = 1e-3 * (Dates.now() - t0).value / 3600.0

    if wt - h0 > 0.5
        # test if time limit for checkpoint (in hours) has been reached 
        h = 1e-3 * (Dates.now() - t).value / 3600.0

        if h >= ct 
            println("Generating checkpoint at cutoff Λ = $(Λ) ...")

            if haskey(cp, "a/$(Λ)") == false 
                checkpoint!(cp, Λ, dΛ, m, a)
            end

            println("Successfully generated checkpoint.")
            t = Dates.now() 
        end 
    end 

    # close files 
    close(obs)
    close(cp)

    return t 
end





# generate launcher file
function save_launcher!(
    path      :: String,
    src       :: String,
    f         :: String,
    name      :: String,
    size      :: Int64,
    model     :: String,
    symmetry  :: String,
    J         :: Vector{Float64}
    ;
    S         :: Float64 = 0.5,
    N         :: Float64 = 2.0,
    β         :: Float64 = 1.0,
    num_σ     :: Int64   = 50,
    num_Ω     :: Int64   = 15,
    num_ν     :: Int64   = 10,
    max_iter  :: Int64   = 30,
    eval      :: Int64   = 25,
    loops     :: Int64   = 1,
    initial   :: Float64 = 5.0,
    final     :: Float64 = 0.5,
    bmin      :: Float64 = 0.02,
    bmax      :: Float64 = 0.2,
    overwrite :: Bool    = true,
    wt        :: Float64 = 24.0,
    ct        :: Float64 = 1.0
    )         :: Nothing

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
                    max_iter  = $(max_iter),
                    eval      = $(eval),
                    loops     = $(loops),
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

# generate job file for Slurm cluster 
function make_job!(
    path          :: String, 
    dir           :: String,
    input         :: String,
    exe           :: String,
    account       :: String,
    cpus_per_task :: Int64, 
    time          :: String, 
    partition     :: String,
    output        :: String
    )             :: Nothing 

    open(path, "w") do file
        # set Slurm parameters 
        write(file, "#!/bin/bash -l \n")
        write(file, "#SBATCH --account=$(account) \n")
        write(file, "#SBATCH --nodes=1 \n")
        write(file, "#SBATCH --ntasks=1 \n")
        write(file, "#SBATCH --ntasks-per-node=1 \n")
        write(file, "#SBATCH --cpus-per-task=$(cpus_per_task) \n")
        write(file, "#SBATCH --time=$(time) \n")
        write(file, "#SBATCH --partition=$(partition) \n")
        write(file, "#SBATCH --output=$(output) \n \n")

        # set number of threads 
        write(file, "export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK \n")

        # go to working directory 
        write(file, "cd $(dir) \n")

        # start calculation 
        write(file, "numactl --physcpubind=0-$(cpus_per_task - 1) -- $(exe) $(input) -E 'run(`numactl -s`)'")
    end 

    return nothing 
end

# generate file structure for calculations with different parameters given path to a directory with launcher files 
function make_repository!(
    dir            :: String,
    exe            :: String,
    account        :: String,
    cpus_per_task  :: Int64, 
    time           :: String, 
    partition      :: String
    )              :: Nothing 

    # init folder for saving finished calculations 
    mkdir(joinpath(dir, "finished"))

    # for each *.jl file, init a new folder, move the *.jl file into it and create a job file
    for file in readdir(dir)
        if endswith(file, ".jl")
            subdir = joinpath(dir, split(file, ".jl")[1])
            input  = joinpath(subdir, file)
            path   = joinpath(subdir, split(file, ".jl")[1] * ".job")
            output = split(file, ".jl")[1] * ".out"

            mkdir(subdir)
            mv(joinpath(dir, file), input)
            make_job!(path, subdir, file, exe, account, cpus_per_task, time, partition, output)
        end
    end 

    return nothing 
end

# collect final results from repository, by copying finished calculations to the respective folder 
function collect_repository!(
    dir :: String 
    )   :: Nothing 

    println("Collecting results from repository, this may take a while ...")

    # check that finished folder exists
    @assert isdir(joinpath(dir, "finished")) "Folder $(joinpath(dir, "finished")) does not exist."

    # for each folder move *_obs and *_cp, then remove their parent dir. If calculation has not finished set overwrite = false in *.jl file
    for file in readdir(dir) 
        if isdir(joinpath(dir, file)) && file != "finished"
            # buffer names of output files
            obs_name = file * "_obs"
            cp_name  = file * "_cp"

            # buffer paths of output files and parent dir
            subdir   = joinpath(dir, file)
            obs_file = joinpath(subdir, obs_name)
            cp_file  = joinpath(subdir, cp_name)

            # check if calculation is finished 
            obs_data = h5open(obs_file, "r") 
            cp_data  = h5open(cp_file, "r") 

            if haskey(obs_data, "finished") && haskey(cp_data, "finished")
                # close files 
                close(obs_data)
                close(cp_data)

                # move files to finished folder 
                mv(obs_file, joinpath(joinpath(dir, "finished"), obs_name))
                mv(cp_file, joinpath(joinpath(dir, "finished"), cp_name))

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





# load launcher for parquet and FRG with different numbers of loops 
include("parquet.jl")
include("launcher_1l.jl")
include("launcher_2l.jl")
include("launcher_ml.jl")

# launcher for the solver 
function launch!(
    f         :: String,
    name      :: String,
    size      :: Int64,
    model     :: String,
    symmetry  :: String,
    J         :: Vector{Float64}
    ;
    S         :: Float64 = 0.5,
    N         :: Float64 = 2.0,
    β         :: Float64 = 1.0,
    num_σ     :: Int64   = 50,
    num_Ω     :: Int64   = 15,
    num_ν     :: Int64   = 10,
    max_iter  :: Int64   = 30,
    eval      :: Int64   = 25,
    loops     :: Int64   = 1,
    initial   :: Float64 = 5.0,
    final     :: Float64 = 0.5,
    bmin      :: Float64 = 0.02,
    bmax      :: Float64 = 0.2,
    overwrite :: Bool    = true,
    wt        :: Float64 = 24.0,
    ct        :: Float64 = 1.0
    )         :: Nothing

    println()
    println("#--------------------------------------------------------------------------------------#")
    println("Initializing solver ...")
    println()

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

        # build lattice and save to files
        println()
        l = get_lattice(name, size)
        init_model!(model, J, l)
        r = get_reduced_lattice(l)
        save!(obs, l)
        save!(obs, r)
        save!(cp, l)
        save!(cp, r)

        # close files 
        close(obs)
        close(cp)

        # build frequency meshes
        σ = get_mesh(4.0 * initial, 800.0 * initial, num_σ)
        Ω = get_mesh(2.0 * initial, 300.0 * initial, num_Ω)
        ν = get_mesh(3.0 * initial, 500.0 * initial, num_ν)
        m = mesh(σ, Ω, ν)

        # build action 
        a = get_action_empty(symmetry, r, m, S = S, N = N)
        init_action!(l, r, a)

        # initialize by parquet iterations
        println()
        println("Warming up with some parquet iterations, this may take a while ...")
        launch_parquet!(obs_file, cp_file, symmetry, l, r, m, a, initial, bmax * initial, β, max_iter, eval, S = S, N = N)
        println("Done. Action is initialized with parquet solution.")

        println()
        println("Solver is ready.")
        println("#--------------------------------------------------------------------------------------#")
        println()

        # start calculation 
        println("Renormalization group flow with ℓ = $(loops) ...")

        if loops == 1 
            launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, initial, final, bmax * initial, bmin, bmax, eval, wt, ct, S = S, N = N)
        elseif loops == 2
            launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, initial, final, bmax * initial, bmin, bmax, eval, wt, ct, S = S, N = N)
        elseif loops >= 3
            launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, loops, initial, final, bmax * initial, bmin, bmax, eval, wt, ct, S = S, N = N)
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
                println("#--------------------------------------------------------------------------------------#")
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
                println("#--------------------------------------------------------------------------------------#")
                println()

                # resume calculation
                println("Renormalization group flow with loops = $(loops) ...")
                
                if loops == 1 
                    launch_1l!(obs_file, cp_file, symmetry, l, r, m, a, Λ, final, dΛ, bmin, bmax, eval, wt, ct, S = S, N = N)
                elseif loops == 2
                    launch_2l!(obs_file, cp_file, symmetry, l, r, m, a, Λ, final, dΛ, bmin, bmax, eval, wt, ct, S = S, N = N)
                elseif loops >= 3
                    launch_ml!(obs_file, cp_file, symmetry, l, r, m, a, loops, Λ, final, dΛ, bmin, bmax, eval, wt, ct, S = S, N = N)
                end 
            end 
        else 
            println()
            println("Found no existing output files, terminating solver ...")
            println("#--------------------------------------------------------------------------------------#")
        end 
    end

    println()
    println("#--------------------------------------------------------------------------------------#")
    println("Solver terminated.")
    println("#--------------------------------------------------------------------------------------#")
    println()

    return nothing 
end

            



        













