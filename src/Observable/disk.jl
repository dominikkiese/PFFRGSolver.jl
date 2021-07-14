# save real space correlations to HDF5 file
function save_χ!(
    file     :: HDF5.File,
    Λ        :: Float64,
    symmetry :: String,
    χ        :: Vector{Vector{Float64}}
    )        :: Nothing

    # save symmetry group
    if haskey(file, "symmetry") == false
        file["symmetry"] = symmetry
    end

    if symmetry == "su2"
        file["χ/$(Λ)/diag"] = χ[1]
    elseif symmetry == "u1-sym"
        file["χ/$(Λ)/xx"] = χ[1]
        file["χ/$(Λ)/zz"] = χ[2]
        file["χ/$(Λ)/xy"] = χ[3]
    end

    return nothing
end

"""
    read_χ_all(
        file    :: HDF5.File,
        Λ       :: Float64
        ;
        verbose :: Bool = true
        )       :: Vector{Vector{Float64}}

Read all available real space correlations from HDF5 file (*_obs) at cutoff Λ.
"""
function read_χ_all(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Vector{Vector{Float64}}

    # filter out nearest available cutoff 
    list    = keys(file["χ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))

    if verbose
        println("Λ was adjusted to $(cutoffs[index]).")
    end

    # read symmetry group 
    symmetry = read(file, "symmetry")

    # read correlations 
    χ = Vector{Float64}[]

    if symmetry == "su2"
        push!(χ, read(file, "χ/$(cutoffs[index])/diag"))
    elseif symmetry == "u1-sym"
        push!(χ, read(file, "χ/$(cutoffs[index])/xx"))
        push!(χ, read(file, "χ/$(cutoffs[index])/zz"))
        push!(χ, read(file, "χ/$(cutoffs[index])/xy"))
    end

    return χ 
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
    labels = keys(file["χ/$(ref)"])

    return labels 
end

"""
    read_χ(
        file    :: HDF5.File,
        Λ       :: Float64,
        label   :: String
        ;
        verbose :: Bool = true
        )       :: Vector{Float64}

Read real space correlations with name `label` from HDF5 file (*_obs) at cutoff Λ.
"""
function read_χ(
    file    :: HDF5.File,
    Λ       :: Float64,
    label   :: String
    ;
    verbose :: Bool = true
    )       :: Vector{Float64}

    # filter out nearest available cutoff 
    list    = keys(file["χ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))

    if verbose
        println("Λ was adjusted to $(cutoffs[index]).")
    end

    # read correlations with requested label
    χ = read(file, "χ/$(cutoffs[index])/" * label)

    return χ
end

"""
    read_χ_flow_at_site(
        file  :: HDF5.File,
        site  :: Int64,
        label :: String
        )     :: NTuple{2, Vector{Float64}}

Read flow of real space correlations with name `label` from HDF5 file (*_obs) at irreducible site.
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
        χ[i] = read(file, "χ/$(cutoffs[i])/" * label)[site]
    end

    return cutoffs, χ 
end

"""
    compute_structure_factor_flow!(
        file_in  :: HDF5.File,
        file_out :: HDF5.File,
        k        :: Matrix{Float64},
        label    :: String  
        )        :: Nothing

Compute the flow of the static structure factor from real space correlations with name `label` in file_in (*_obs) and save the result to file_out.
The momentum space discretization k should be formatted such that k[:, n] is the n-th momentum.
"""
function compute_structure_factor_flow!(
    file_in  :: HDF5.File,
    file_out :: HDF5.File,
    k        :: Matrix{Float64},
    label    :: String  
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
        χ = read_χ(file_in, Λ, label, verbose = false)

        # compute structure factor
        s = compute_structure_factor(χ, k, l, r)

        # save structure factor
        file_out["s/$(Λ)/" * label] = s 
    end 

    println("Done.")

    return nothing 
end

"""
    compute_structure_factor_flow_all!(
        file_in  :: HDF5.File,
        file_out :: HDF5.File,
        k        :: Matrix{Float64} 
        )        :: Nothing

Compute the flows of the static structure factors for all available real space correlations in file_in (*_obs) and save the result to file_out.
The momentum space discretization k should be formatted such that k[:, n] is the n-th momentum.
"""
function compute_structure_factor_flow_all!(
    file_in  :: HDF5.File,
    file_out :: HDF5.File,
    k        :: Matrix{Float64}
    )        :: Nothing

    # filter out a sorted list of cutoffs
    list    = keys(file_in["χ"])
    cutoffs = sort(parse.(Float64, list), rev = true)

    # read lattice and reduced lattice
    l = read_lattice(file_in)
    r = read_reduced_lattice(file_in)

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
            χ = read_χ(file_in, Λ, label, verbose = false)

            # compute structure factor
            s = compute_structure_factor(χ, k, l, r)

            # save structure factor
            file_out["s/$(Λ)/" * label] = s 
        end
    end 

    println("Done.")

    return nothing 
end

"""
    read_structure_factor(
        file    :: HDF5.File,
        Λ       :: Float64,
        label   :: String
        ;
        verbose :: Bool = true
        )       :: Vector{Float64}

Read structure factor with name `label` from HDF5 file at cutoff Λ.
"""
function read_structure_factor(
    file    :: HDF5.File,
    Λ       :: Float64,
    label   :: String
    ;
    verbose :: Bool = true
    )       :: Vector{Float64}

    # filter out nearest available cutoff 
    list    = keys(file["s"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))

    if verbose
        println("Λ was adjusted to $(cutoffs[index]).")
    end

    # read structure factor with requested label 
    s = read(file, "s/$(cutoffs[index])/" * label)

    return s 
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
        s[i] = read(file, "s/$(cutoffs[i])/" * label)[index]
    end

    return cutoffs, s 
end

"""
    read_reference_momentum(
        file  :: HDF5.File,
        Λ     :: Float64,
        label :: String
        )     :: Vector{Float64}

Read the momentum with maximum structure factor value with name `label` at cutoff Λ from HDF5 file.
"""
function read_reference_momentum(
    file  :: HDF5.File,
    Λ     :: Float64,
    label :: String
    )     :: Vector{Float64}

    # read struture factor 
    s = read_structure_factor(file, Λ, label)

    # determine momentum with maximum amplitude 
    k = read(file, "k")
    p = k[:, argmax(s)]

    return p 
end