# save checkpoint to file
function checkpoint!(
    file :: HDF5.File, 
    Λ    :: Float64,
    dΛ   :: Float64,
    m    :: mesh,
    a    :: action_sun
    )    :: Nothing

    # save step size
    file["dΛ/$(Λ)"] = dΛ

    # save frequency meshes
    file["σ/$(Λ)"] = m.σ
    file["Ω/$(Λ)"] = m.Ω
    file["ν/$(Λ)"] = m.ν

    # save spin length
    if haskey(file, "S") == false 
        file["S"] = a.S 
    end 

    # save symmetry group
    if haskey(file, "N") == false 
        file["N"] = a.N 
    end

    # save self energy
    file["a/$(Λ)/Σ"] = a.Σ

    # save vertex 
    save!(file, "a/$(Λ)/Γ/spin", a.Γ[1]) 
    save!(file, "a/$(Λ)/Γ/dens", a.Γ[2]) 

    return nothing 
end

# read checkpoint from file 
function read_checkpoint_sun(
    file :: HDF5.File,
    Λ    :: Float64
    )    :: Tuple{Float64, Float64, mesh, action_sun}

    # filter out nearest available cutoff 
    list    = keys(file["σ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read step size
    dΛ = read(file, "dΛ/$(cutoffs[index])")

    # read frequency meshes 
    σ = read(file, "σ/$(cutoffs[index])")
    Ω = read(file, "Ω/$(cutoffs[index])")
    ν = read(file, "ν/$(cutoffs[index])")
    m = mesh(σ, Ω, ν)

    # read spin length and symmetry group 
    S = read(file, "S")
    N = read(file, "N")

    # read self energy
    Σ = read(file, "a/$(cutoffs[index])/Σ")

    # read vertex 
    Γ = vertex[read_vertex(file, "a/$(cutoffs[index])/Γ/spin"), 
               read_vertex(file, "a/$(cutoffs[index])/Γ/dens")]

    # build action 
    a = action_sun(S, N, Σ, Γ)

    return cutoffs[index], dΛ, m, a 
end