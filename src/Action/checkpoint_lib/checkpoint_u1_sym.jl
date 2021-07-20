# save current status to file
function checkpoint!(
    file :: HDF5.File,
    Λ    :: Float64,
    dΛ   :: Float64,
    m    :: Mesh,
    a    :: Action_u1_sym
    )    :: Nothing

    # save step size
    file["dΛ/$(Λ)"] = dΛ

    # save frequency meshes
    file["σ/$(Λ)"] = m.σ

    for i in 1 : 6
        file["Ωs/$(Λ)/$(i)"] = m.Ωs[i]
        file["νs/$(Λ)/$(i)"] = m.νs[i]
        file["Ωt/$(Λ)/$(i)"] = m.Ωt[i]
        file["νt/$(Λ)/$(i)"] = m.νt[i]
        file["Ωu/$(Λ)/$(i)"] = m.Ωu[i]
        file["νu/$(Λ)/$(i)"] = m.νu[i]
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
function read_checkpoint_u1_sym(
    file :: HDF5.File,
    Λ    :: Float64
    )    :: Tuple{Float64, Float64, Mesh, Action_u1_sym}

    # filter out nearest available cutoff
    list    = keys(file["σ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read step size
    dΛ = read(file, "dΛ/$(cutoffs[index])")

    # read frequency meshes
    σ  = read(file, "σ/$(cutoffs[index])")
    Ωs = Vector{Vector{Float64}}(undef, 2)
    νs = Vector{Vector{Float64}}(undef, 2)
    Ωt = Vector{Vector{Float64}}(undef, 2)
    νt = Vector{Vector{Float64}}(undef, 2)
    Ωu = Vector{Vector{Float64}}(undef, 2)
    νu = Vector{Vector{Float64}}(undef, 2)

    for i in 1 : 6
        Ωs[i] = read(file, "Ωs/$(cutoffs[index])/$(i)")
        νs[i] = read(file, "νs/$(cutoffs[index])/$(i)")
        Ωt[i] = read(file, "Ωt/$(cutoffs[index])/$(i)")
        νt[i] = read(file, "νt/$(cutoffs[index])/$(i)")
        Ωu[i] = read(file, "Ωu/$(cutoffs[index])/$(i)")
        νu[i] = read(file, "νu/$(cutoffs[index])/$(i)")
    end 

    m = Mesh(length(σ), length(Ωs[1]), length(νs[1]), σ, Ωs, νs, Ωt, νt, Ωu, νu)

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
    a = Action_u1_sym(Σ, Γ)

    return cutoffs[index], dΛ, m, a
end