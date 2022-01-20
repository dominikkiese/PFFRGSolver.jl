# save real space correlations to HDF5 file
function save_χ!(
    file     :: HDF5.File,
    Λ        :: Float64,
    symmetry :: String,
    m        :: Mesh,
    χ        :: Vector{Matrix{Float64}}
    )        :: Nothing

    # save symmetry group
    if haskey(file, "symmetry") == false
        file["symmetry"] = symmetry
    end

    # save frequency mesh 
    file["χ/$(Λ)/mesh"] = m.χ

    if symmetry == "su2"
        file["χ/$(Λ)/diag"] = χ[1]
    elseif symmetry == "u1-dm"
        file["χ/$(Λ)/xx"] = χ[1]
        file["χ/$(Λ)/zz"] = χ[2]
        file["χ/$(Λ)/xy"] = χ[3]
    end

    return nothing
end

"""
    read_χ_labels(
        file :: HDF5.File
        )    :: Vector{String}

Read labels of available real space correlations from HDF5 file (*_obs).
"""
function read_χ_labels(
    file :: HDF5.File
    )    :: Vector{String}

    ref    = keys(file["χ"])[1]
    labels = String[]

    for key in keys(file["χ/$(ref)"])
        if key != "mesh"
            push!(labels, key)
        end 
    end

    return labels 
end

"""
    read_χ(
        file    :: HDF5.File,
        Λ       :: Float64,
        label   :: String
        ;
        verbose :: Bool = true
        )       :: Tuple{Vector{Float64}, Matrix{Float64}}

Read real space correlations with name `label` and the associated frequency mesh from HDF5 file (*_obs) at cutoff Λ.
"""
function read_χ(
    file    :: HDF5.File,
    Λ       :: Float64,
    label   :: String
    ;
    verbose :: Bool = true
    )       :: Tuple{Vector{Float64}, Matrix{Float64}}

    # filter out nearest available cutoff 
    list    = keys(file["χ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))

    if verbose
        println("Λ was adjusted to $(cutoffs[index]).")
    end

    # read frequency mesh
    m = read(file, "χ/$(cutoffs[index])/mesh")

    # read correlations with requested label
    χ = read(file, "χ/$(cutoffs[index])/" * label)

    return m, χ
end

"""
    read_χ_all(
        file    :: HDF5.File,
        Λ       :: Float64
        ;
        verbose :: Bool = true
        )       :: Tuple{Vector{Float64}, Vector{Matrix{Float64}}}

Read all available real space correlations and the associated frequency mesh from HDF5 file (*_obs) at cutoff Λ.
"""
function read_χ_all(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Tuple{Vector{Float64}, Vector{Matrix{Float64}}}

    # filter out nearest available cutoff 
    list    = keys(file["χ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))

    if verbose
        println("Λ was adjusted to $(cutoffs[index]).")
    end

    # read symmetry group 
    symmetry = read(file, "symmetry")

    # read frequency mesh
    m = read(file, "χ/$(cutoffs[index])/mesh")

    # read correlations 
    χ = Matrix{Float64}[]

    for label in read_χ_labels(file)
        push!(χ, read(file, "χ/$(cutoffs[index])/" * label))
    end

    return m, χ 
end

"""
    read_χ_flow_at_site(
        file  :: HDF5.File,
        site  :: Int64,
        label :: String
        )     :: NTuple{2, Vector{Float64}}

Read flow of static real space correlations with name `label` from HDF5 file (*_obs) at irreducible site.
"""
function read_χ_flow_at_site(
    file  :: HDF5.File,
    site  :: Int64,
    label :: String
    )     :: NTuple{2, Vector{Float64}}

    # filter out a sorted list of cutoffs
    list    = keys(file["χ"])
    cutoffs = sort(parse.(Float64, list), rev = true)

    # allocate array to store values
    χ = zeros(Float64, length(cutoffs))

    # fill array with values at given site 
    for i in eachindex(cutoffs)
        χ[i] = read(file, "χ/$(cutoffs[i])/" * label)[site, 1]
    end

    return cutoffs, χ 
end

"""
    compute_structure_factor_flow!(
        file_in  :: HDF5.File,
        file_out :: HDF5.File,
        k        :: Matrix{Float64},
        label    :: String 
        ;
        static   :: Bool = false
        )        :: Nothing

Compute the flow of the structure factor from real space correlations with name `label` in file_in (*_obs) and save the result to file_out.
The momentum space discretization k should be formatted such that k[:, n] is the n-th momentum. If static = true, only the static 
structure factor (i.e. the w = 0 component) is computed.
"""
function compute_structure_factor_flow!(
    file_in  :: HDF5.File,
    file_out :: HDF5.File,
    k        :: Matrix{Float64},
    label    :: String
    ;
    static   :: Bool = false
    )        :: Nothing

    # filter out a sorted list of cutoffs
    list    = keys(file_in["χ"])
    cutoffs = sort(parse.(Float64, list), rev = true)

    # read lattice and reduced lattice
    l, r = read_lattice(file_in)

    # save momenta 
    if haskey(file_out, "k") == false
        file_out["k"] = k 
    end

    println("Computing structure factor flow ...")

    # compute and save structure factors 
    for Λ in cutoffs 
        # read correlations
        m, χ = read_χ(file_in, Λ, label, verbose = false)

        if static == false 
            # save frequency mesh 
            file_out["s/$(Λ)/mesh"] = m

            # allocate output matrix 
            smat = zeros(Float64, size(k, 2), length(m))

            # compute structure factors for all Matsubara frequencies
            for w in eachindex(m)
                smat[:, w] .= compute_structure_factor(χ[:, w], k, l, r)
            end

            # save structure factors 
            file_out["s/$(Λ)/" * label] = smat
        else 
            # save frequency mesh 
            file_out["s/$(Λ)/mesh"] = Float64[0.0]

            # allocate output matrix 
            smat = zeros(Float64, size(k, 2), 1)

            # compute structure factors for all Matsubara frequencies
            smat[:, 1] .= compute_structure_factor(χ[:, 1], k, l, r)

            # save structure factors 
            file_out["s/$(Λ)/" * label] = smat
        end
    end 

    println("Done.")

    return nothing 
end

"""
    compute_structure_factor_flow_all!(
        file_in  :: HDF5.File,
        file_out :: HDF5.File,
        k        :: Matrix{Float64} 
        ;
        static   :: Bool = false
        )        :: Nothing

Compute the flows of the structure factors for all available real space correlations in file_in (*_obs) and save the result to file_out.
The momentum space discretization k should be formatted such that k[:, n] is the n-th momentum. If static = true, only the static 
structure factor (i.e. the w = 0 component) is computed.
"""
function compute_structure_factor_flow_all!(
    file_in  :: HDF5.File,
    file_out :: HDF5.File,
    k        :: Matrix{Float64}
    ;
    static   :: Bool = false
    )        :: Nothing

    # filter out a sorted list of cutoffs
    list    = keys(file_in["χ"])
    cutoffs = sort(parse.(Float64, list), rev = true)

    # read lattice and reduced lattice
    l, r = read_lattice(file_in)

    # save momenta 
    if haskey(file_out, "k") == false
        file_out["k"] = k 
    end

    println("Computing structure factor flows ...")

    # read available labels
    for label in read_χ_labels(file_in)
        # compute and save structure factors 
        for Λ in cutoffs 
            # read correlations
            m, χ = read_χ(file_in, Λ, label, verbose = false)

            if static == false 
                # save frequency mesh 
                file_out["s/$(Λ)/mesh"] = m

                # allocate output matrix 
                smat = zeros(Float64, size(k, 2), length(m))

                # compute structure factors for all Matsubara frequencies
                for w in eachindex(m)
                    smat[:, w] .= compute_structure_factor(χ[:, w], k, l, r)
                end

                # save structure factors 
                file_out["s/$(Λ)/" * label] = smat
            else 
                # save frequency mesh 
                file_out["s/$(Λ)/mesh"] = Float64[0.0]

                # allocate output matrix 
                smat = zeros(Float64, size(k, 2), 1)

                # compute structure factors for all Matsubara frequencies
                smat[:, 1] .= compute_structure_factor(χ[:, 1], k, l, r)

                # save structure factors 
                file_out["s/$(Λ)/" * label] = smat
            end
        end 
    end 

    println("Done.")

    return nothing 
end

"""
    read_structure_factor_labels(
        file :: HDF5.File
        )    :: Vector{String}

Read labels of available structure factors from HDF5 file (*_obs).
"""
function read_structure_factor_labels(
    file :: HDF5.File
    )    :: Vector{String}

    ref    = keys(file["s"])[1]
    labels = String[]

    for key in keys(file["s/$(ref)"])
        if key != "mesh"
            push!(labels, key)
        end 
    end

    return labels 
end

"""
    read_structure_factor(
        file    :: HDF5.File,
        Λ       :: Float64,
        label   :: String
        ;
        verbose :: Bool = true
        )       :: Tuple{Vector{Float64}, Matrix{Float64}}

Read structure factor with name `label` and the associated frequency mesh from HDF5 file at cutoff Λ.
"""
function read_structure_factor(
    file    :: HDF5.File,
    Λ       :: Float64,
    label   :: String
    ;
    verbose :: Bool = true
    )       :: Tuple{Vector{Float64}, Matrix{Float64}}

    # filter out nearest available cutoff 
    list    = keys(file["s"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))

    if verbose
        println("Λ was adjusted to $(cutoffs[index]).")
    end

    # read frequency mesh
    m = read(file, "s/$(cutoffs[index])/mesh")

    # read structure factor with requested label 
    s = read(file, "s/$(cutoffs[index])/" * label)

    return m, s 
end 

"""
    read_structure_factor_all(
        file    :: HDF5.File,
        Λ       :: Float64
        ;
        verbose :: Bool = true
        )       :: Tuple{Vector{Float64}, Vector{Matrix{Float64}}}

Read all available structure factors and the associated frequency mesh from HDF5 file at cutoff Λ.
"""
function read_structure_factor_all(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Tuple{Vector{Float64}, Vector{Matrix{Float64}}}

    # filter out nearest available cutoff 
    list    = keys(file["s"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))

    if verbose
        println("Λ was adjusted to $(cutoffs[index]).")
    end

    # read frequency mesh
    m = read(file, "s/$(cutoffs[index])/mesh")

    # read structure factors 
    s = Matrix{Float64}[]

    for label in read_structure_factor_labels(file)
        push!(s, read(file, "s/$(cutoffs[index])/" * label))
    end

    return m, s 
end 

"""
    read_structure_factor_flow_at_momentum(
        file    :: HDF5.File,
        p       :: Vector{Float64},
        label   :: String
        ;
        verbose :: Bool = true
        )       :: NTuple{2, Vector{Float64}}

Read flow of static structure factor with name `label` from HDF5 file at momentum p.
"""
function read_structure_factor_flow_at_momentum(
    file    :: HDF5.File,
    p       :: Vector{Float64},
    label   :: String
    ;
    verbose :: Bool = true
    )       :: NTuple{2, Vector{Float64}}

    # filter out a sorted list of cutoffs
    list    = keys(file["s"])
    cutoffs = sort(parse.(Float64, list), rev = true)

    # locate closest momentum 
    k     = read(file, "k")
    dists = Float64[norm(k[:, i] .- p) for i in 1 : size(k, 2)]
    index = argmin(dists)

    if verbose
        println("Momentum was adjusted to k = $(k[:, index]).")
    end

    # allocate array to store values
    s = zeros(Float64, length(cutoffs))

    # fill array with values at given momentum
    for i in eachindex(cutoffs) 
        s[i] = read(file, "s/$(cutoffs[i])/" * label)[index, 1]
    end

    return cutoffs, s 
end

"""
    read_reference_momentum(
        file  :: HDF5.File,
        Λ     :: Float64,
        label :: String
        )     :: Vector{Float64}

Read the momentum with maximum static structure factor value with name `label` at cutoff Λ from HDF5 file.
"""
function read_reference_momentum(
    file  :: HDF5.File,
    Λ     :: Float64,
    label :: String
    )     :: Vector{Float64}

    # read struture factor 
    m, s = read_structure_factor(file, Λ, label)

    # determine momentum with maximum amplitude 
    k = read(file, "k")
    p = k[:, argmax(s[:, 1])]

    return p 
end