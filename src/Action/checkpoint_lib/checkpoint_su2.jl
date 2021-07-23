# save current status to file
function checkpoint!(
    file :: HDF5.File,
    Λ    :: Float64,
    dΛ   :: Float64,
    m    :: Mesh,
    a    :: Action_su2
    )    :: Nothing

    # save step size
    file["dΛ/$(Λ)"] = dΛ

    # save frequency meshes
    file["σ/$(Λ)"]  = m.σ

    for comp in 1 : 2
        file["Ωs/$(Λ)/$(comp)"] = m.Ωs[comp]
        file["νs/$(Λ)/$(comp)"] = m.νs[comp]
        file["Ωt/$(Λ)/$(comp)"] = m.Ωt[comp]
        file["νt/$(Λ)/$(comp)"] = m.νt[comp]
        file["Ωu/$(Λ)/$(comp)"] = m.Ωu[comp]
        file["νu/$(Λ)/$(comp)"] = m.νu[comp]
    end

    # save spin length
    if haskey(file, "S") == false
        file["S"] = a.S
    end

    # save symmetry group
    if haskey(file, "symmetry") == false
        file["symmetry"] = "su2"
    end

    # save self energy
    file["a/$(Λ)/Σ"] = a.Σ

    # save vertex
    save!(file, "a/$(Λ)/Γ/spin", a.Γ[1])
    save!(file, "a/$(Λ)/Γ/dens", a.Γ[2])

    return nothing
end

# read checkpoint from file
function read_checkpoint_su2(
    file :: HDF5.File,
    Λ    :: Float64
    )    :: Tuple{Float64, Float64, Mesh, Action_su2}

    # filter out nearest available cutoff
    list    = keys(file["σ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read step size
    dΛ = read(file, "dΛ/$(cutoffs[index])")

    # read frequency meshes
    σ  = read(file, "σ/$(cutoffs[index])")
    Ωs = Vector{Float64}[read(file, "Ωs/$(cutoffs[index])/$(comp)") for comp in 1 : 2]
    νs = Vector{Float64}[read(file, "νs/$(cutoffs[index])/$(comp)") for comp in 1 : 2]
    Ωt = Vector{Float64}[read(file, "Ωt/$(cutoffs[index])/$(comp)") for comp in 1 : 2]
    νt = Vector{Float64}[read(file, "νt/$(cutoffs[index])/$(comp)") for comp in 1 : 2]
    Ωu = Vector{Float64}[read(file, "Ωu/$(cutoffs[index])/$(comp)") for comp in 1 : 2]
    νu = Vector{Float64}[read(file, "νu/$(cutoffs[index])/$(comp)") for comp in 1 : 2]
    m  = Mesh(length(σ), length(Ωs[1]), length(νs[1]), σ, Ωs, νs, Ωt, νt, Ωu, νu)

    # read spin length 
    S = read(file, "S")

    # read self energy
    Σ = read(file, "a/$(cutoffs[index])/Σ")

    # read vertex
    Γ = Vertex[read_vertex(file, "a/$(cutoffs[index])/Γ/spin"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/dens")]

    # build action
    a = Action_su2(S, Σ, Γ)

    return cutoffs[index], dΛ, m, a
end