function checkpoint!(
    file :: HDF5.File,
    Λ    :: Float64,
    dΛ   :: Float64,
    m    :: Mesh,
    a    :: Action_su2_hkg
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

    # save spin length
    if haskey(file, "S") == false
        file["S"] = a.S
    end

    # save symmetry group
    if haskey(file, "symmetry") == false
        file["symmetry"] = "su2-hkg"
    end 

    # save self energy
    file["a/$(Λ)/Σ"] = a.Σ

    #save vertex 
    save!(file, "a/$(Λ)/Γ/Γxx", a.Γ[1])
    save!(file, "a/$(Λ)/Γ/Γyy", a.Γ[2])
    save!(file, "a/$(Λ)/Γ/Γzz", a.Γ[3])
    save!(file, "a/$(Λ)/Γ/Γxy", a.Γ[4])
    save!(file, "a/$(Λ)/Γ/Γxz", a.Γ[5])
    save!(file, "a/$(Λ)/Γ/Γyz", a.Γ[6])
    save!(file, "a/$(Λ)/Γ/Γyx", a.Γ[7])
    save!(file, "a/$(Λ)/Γ/Γzx", a.Γ[8])
    save!(file, "a/$(Λ)/Γ/Γzy", a.Γ[9])
    save!(file, "a/$(Λ)/Γ/Γdd", a.Γ[10])
    save!(file, "a/$(Λ)/Γ/Γxd", a.Γ[11])
    save!(file, "a/$(Λ)/Γ/Γyd", a.Γ[12])
    save!(file, "a/$(Λ)/Γ/Γzd", a.Γ[13])
    save!(file, "a/$(Λ)/Γ/Γdx", a.Γ[14])
    save!(file, "a/$(Λ)/Γ/Γdy", a.Γ[15])
    save!(file, "a/$(Λ)/Γ/Γdz", a.Γ[16])
    

    return nothing
end

# read checkpoint from file
function read_checkpoint_su2_hkg(
    file :: HDF5.File,
    Λ    :: Float64
    )    :: Tuple{Float64, Float64, Mesh, Action_su2_hkg}

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

    # read spin length
    S = read(file, "S")

    # read self energy
    Σ = read(file, "a/$(cutoffs[index])/Σ")

    # read vertex 
    Γ = Vertex[read_vertex(file, "a/$(cutoffs[index])/Γ/Γxx"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γyy"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γzz"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γxy"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γxz"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γyz"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γyx"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γzx"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γzy"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γdd"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γdx"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γdy"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γdz"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γxd"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γyd"),
               read_vertex(file, "a/$(cutoffs[index])/Γ/Γzd"),] 

    # build action 
    a = Action_su2_hkg(S, Σ, Γ)

    return cutoffs[index], dΛ, m, a
end