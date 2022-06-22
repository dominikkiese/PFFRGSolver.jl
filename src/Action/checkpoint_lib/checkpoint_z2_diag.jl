# save current status to file
function checkpoint!(
    file :: HDF5.File,
    Λ    :: Float64,
    dΛ   :: Float64,
    m    :: Mesh,
    a    :: Action_z2_diag
    )    :: Nothing

    # save step size
    file["dΛ/$(Λ)"] = dΛ

    # save frequency meshes
    file["σ/$(Λ)"]  = m.σ
    file["Ωs/$(Λ)"] = m.Ωs
    file["νs/$(Λ)"] = m.νs
    file["Ωt/$(Λ)"] = m.Ωt
    file["νt/$(Λ)"] = m.νt
    file["χ/$(Λ)"]  = m.χ

    # save symmetry group
    if haskey(file, "symmetry") == false
        file["symmetry"] = "z2-diag"
    end

    # save self energy
    file["a/$(Λ)/Σ"] = a.Σ

    # save vertex
    save!(file, "a/$(Λ)/Γ/Γxx", a.Γ[1])
    save!(file, "a/$(Λ)/Γ/Γyy", a.Γ[2])
    save!(file, "a/$(Λ)/Γ/Γzz", a.Γ[3])
    save!(file, "a/$(Λ)/Γ/Γdd", a.Γ[4])

    return nothing
end

# read checkpoint from file
function read_checkpoint_z2_diag(
    file :: HDF5.File,
    Λ    :: Float64
    )    :: Tuple{Float64, Float64, Mesh, Action_z2_diag}

    # filter out nearest available cutoff
    list    = keys(file["σ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read step size
    dΛ = read(file, "dΛ/$(cutoffs[index])")

    # read frequency meshes
    σ  = read(file, "σ/$(cutoffs[index])")
    Ωs = read(file, "Ωs/$(cutoffs[index])")
    νs = read(file, "νs/$(cutoffs[index])")
    Ωt = read(file, "Ωt/$(cutoffs[index])")
    νt = read(file, "νt/$(cutoffs[index])")
    χ  = read(file, "χ/$(cutoffs[index])")
    m  = Mesh(length(σ), length(Ωs), length(νs), length(χ), σ, Ωs, νs, Ωt, νt, χ)

    # read self energy
    Σ = read(file, "a/$(cutoffs[index])/Σ")

    # read vertex
    Γ = Vertex[read_vertex(file, "a/$(cutoffs[index])/Γ/Γxx"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γyy"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γzz"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γdd")]

    # build action
    a = Action_z2_diag(Σ, Γ)

    return cutoffs[index], dΛ, m, a
end