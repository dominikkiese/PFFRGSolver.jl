"""
    checkpoint!(
        file :: HDF5.File,
        Λ    :: Float64,
        dΛ   :: Float64,
        m    :: mesh,
        a    :: action_su2
        )    :: Nothing

Save current status of FRG calculation with SU(2) symmetry to HDF5 file.
Requires cutoff Λ, ODE stepwidth dΛ, frequency meshes (wrapped in mesh struct) and vertices (wrapped in action_su2 struct).
"""
function checkpoint!(
    file :: HDF5.File,
    Λ    :: Float64,
    dΛ   :: Float64,
    m    :: mesh,
    a    :: action_su2
    )    :: Nothing

    # save step size
    file["dΛ/$(Λ)"] = dΛ

    # save frequency meshes
    file["σ/$(Λ)"]  = m.σ
    file["Ωs/$(Λ)"] = m.Ωs
    file["νs/$(Λ)"] = m.νs
    file["Ωt/$(Λ)"] = m.Ωt
    file["νt/$(Λ)"] = m.νt

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

"""
    read_checkpoint_su2(
        file :: HDF5.File,
        Λ    :: Float64
        )    :: Tuple{Float64, Float64, mesh, action_su2}

Read checkpoint of FRG calculation with SU(2) symmetry from HDF5 file.
Returns cutoff Λ, ODE stepwidth dΛ, frequency meshes (wrapped in mesh struct) and vertices (wrapped in action_su2 struct).
"""
function read_checkpoint_su2(
    file :: HDF5.File,
    Λ    :: Float64
    )    :: Tuple{Float64, Float64, mesh, action_su2}

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
    m  = mesh(length(σ), length(Ωs), length(νs), σ, Ωs, νs, Ωt, νt)

    # read spin length and symmetry group
    S = read(file, "S")
    N = read(file, "N")

    # read self energy
    Σ = read(file, "a/$(cutoffs[index])/Σ")

    # read vertex
    Γ = vertex[read_vertex(file, "a/$(cutoffs[index])/Γ/spin"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/dens")]

    # build action
    a = action_su2(S, N, Σ, Γ)

    return cutoffs[index], dΛ, m, a
end
