# save current status to file
function checkpoint!(
    file :: HDF5.File,
    Λ    :: Float64,
    dΛ   :: Float64,
    m    :: Mesh_u1_dm,
    a    :: Action_u1_dm
    )    :: Nothing

    # save step size
    file["dΛ/$(Λ)"] = dΛ

    # save frequency meshes
    file["σ/$(Λ)"]  = m.σ

    for comp in 1 : 6
        file["Ωs/$(Λ)/$(comp)"] = m.Ωs[comp]
        file["νs/$(Λ)/$(comp)"] = m.νs[comp]
        file["Ωt/$(Λ)/$(comp)"] = m.Ωt[comp]
        file["νt/$(Λ)/$(comp)"] = m.νt[comp]
        file["Ωu/$(Λ)/$(comp)"] = m.Ωu[comp]
        file["νu/$(Λ)/$(comp)"] = m.νu[comp]
    end

    # save symmetry group
    if haskey(file, "symmetry") == false
        file["symmetry"] = "u1-sym"
    end

    # save self energy
    file["a/$(Λ)/Σ"] = a.Σ

    # save vertex
    save!(file, "a/$(Λ)/Γ/Γxx", a.Γ[1])
    save!(file, "a/$(Λ)/Γ/Γzz", a.Γ[2])
    save!(file, "a/$(Λ)/Γ/ΓDM", a.Γ[3])
    save!(file, "a/$(Λ)/Γ/Γdd", a.Γ[4])
    save!(file, "a/$(Λ)/Γ/Γzd", a.Γ[5])
    save!(file, "a/$(Λ)/Γ/Γdz", a.Γ[6])

    return nothing
end

# read checkpoint from file
function read_checkpoint_u1_dm(
    file :: HDF5.File,
    Λ    :: Float64
    )    :: Tuple{Float64, Float64, Mesh_u1_m, Action_u1_dm}

    # filter out nearest available cutoff
    list    = keys(file["σ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read step size
    dΛ = read(file, "dΛ/$(cutoffs[index])")

    # read frequency meshes
    σ  = read(file, "σ/$(cutoffs[index])")
    Ωs = SVector(ntuple(comp -> read(file, "Ωs/$(cutoffs[index])/$(comp)"), 6))
    νs = SVector(ntuple(comp -> read(file, "νs/$(cutoffs[index])/$(comp)"), 6))
    Ωt = SVector(ntuple(comp -> read(file, "Ωt/$(cutoffs[index])/$(comp)"), 6))
    νt = SVector(ntuple(comp -> read(file, "νt/$(cutoffs[index])/$(comp)"), 6))
    Ωu = SVector(ntuple(comp -> read(file, "Ωu/$(cutoffs[index])/$(comp)"), 6))
    νu = SVector(ntuple(comp -> read(file, "νu/$(cutoffs[index])/$(comp)"), 6))
    m  = Mesh_u1_dm(length(σ), length(Ωs[1]), length(νs[1]), σ, Ωs, νs, Ωt, νt, Ωu, νu)

    # read self energy
    Σ = read(file, "a/$(cutoffs[index])/Σ")

    # read vertex
    Γ = Vertex[read_vertex(file, "a/$(cutoffs[index])/Γ/Γxx"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γzz"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/ΓDM"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γdd"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γzd"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γdz")]

    # build action
    a = Action_u1_dm(Σ, Γ)

    return cutoffs[index], dΛ, m, a
end