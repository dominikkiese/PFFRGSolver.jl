# save channel to file 
function save!(
    file  :: HDF5.File,
    label :: String,
    ch    :: channel
    )     :: Nothing 

    file[label * "/q1"]   = ch.q1
    file[label * "/q2_1"] = ch.q2_1
    file[label * "/q2_2"] = ch.q2_2
    file[label * "/q3"]   = ch.q3

    return nothing 
end

# read channel from file 
function read_channel(
    file  :: HDF5.File,
    label :: String,
    )     :: channel 

    # read kernels
    q1   = read(file, label * "/q1")
    q2_1 = read(file, label * "/q2_1")
    q2_2 = read(file, label * "/q2_2")
    q3   = read(file, label * "/q3")

    # build channel 
    ch = channel(q1, q2_1, q2_2, q3)

    return ch 
end

# save self energy to file 
function save_self!(
    file :: HDF5.File, 
    Λ    :: Float64,
    m    :: mesh,
    a    :: Action
    )    :: Nothing 

    # save self energy mesh to file 
    file["σ/$(Λ)"] = m.σ

    # save self energy to file
    file["Σ/$(Λ)"] = a.Σ 

    return nothing 
end

"""
    read_self(
        file :: HDF5.File, 
        Λ    :: Float64,
        )    :: NTuple{2, Vector{Float64}}

Read self energy mesh and values from HDF5 file at cutoff Λ.
""" 
function read_self(
    file :: HDF5.File, 
    Λ    :: Float64,
    )    :: NTuple{2, Vector{Float64}}

    # filter out nearest available cutoff 
    list    = keys(file["σ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read self energy mesh
    σ = read(file, "σ/$(cutoffs[index])")

    # read self energy 
    Σ = read(file, "Σ/$(cutoffs[index])")

    return σ, Σ 
end 

# save vertex to file
function save!(
    file  :: HDF5.File, 
    label :: String,
    Γ     :: vertex
    )     :: Nothing 

    # save bare vertex
    file[label * "/bare"] = Γ.bare

    # save channels 
    save!(file, label * "/ch_s", Γ.ch_s)
    save!(file, label * "/ch_t", Γ.ch_t)
    save!(file, label * "/ch_u", Γ.ch_u)

    return nothing 
end

# read vertex from file 
function read_vertex(
    file  :: HDF5.File, 
    label :: String
    )     :: vertex

    # read bare vertex 
    bare = read(file, label * "/bare")

    # read channels
    ch_s = read_channel(file, label * "/ch_s")
    ch_t = read_channel(file, label * "/ch_t")
    ch_u = read_channel(file, label * "/ch_u")

    # build vertex 
    Γ = vertex(bare, ch_s, ch_t, ch_u)

    return Γ
end