# save set of spin-spin correlations to file 
function save_χ!(
    file     :: HDF5.File,
    Λ        :: Float64,
    symmetry :: String,
    χ        :: Vector{Vector{Float64}}
    )        :: Nothing

    if symmetry == "sun"
        file["χ/$(Λ)/diag"] = χ[1]
    end

    return nothing
end

# read spin-spin correlations from file 
function read_χ(
    file  :: HDF5.File,
    Λ     :: Float64,
    label :: String
    )     :: Vector{Float64}

    # filter out nearest available cutoff 
    list    = keys(file["χ"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read correlations with requested label
    χ = read(file, "χ/$(cutoffs[index])/" * label)

    return χ
end

# read flow of spin-spin correlations at a given site from file 
function read_χ_flow_at_site(
    file  :: HDF5.File,
    label :: String,
    site  :: Int64
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

# compute flow of structure factor and save to new file. Input file must contain real space correlations and lattice / reduced_lattice
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
    l = read_lattice(file_in)
    r = read_reduced_lattice(file_in)

    # save momenta 
    file_out["k"] = k 

    # compute and save structure factors 
    for Λ in cutoffs 
        # print status
        println("Current cutoff Λ = $(Λ).")

        # read correlations
        χ = read_χ(file_in, Λ, label)

        # compute structure factor
        s = compute_structure_factor(χ, k, l, r)

        # save structure factor
        file_out["s/$(Λ)/" * label] = s 
    end 

    return nothing 
end

# read structure factor from file 
function read_structure_factor(
    file  :: HDF5.File,
    Λ     :: Float64,
    label :: String
    )     :: Vector{Float64}

    # filter out nearest available cutoff 
    list    = keys(file["s"])
    cutoffs = parse.(Float64, list)
    index   = argmin(abs.(cutoffs .- Λ))
    println("Λ was adjusted to $(cutoffs[index]).")

    # read structure factor with requested label 
    s = read(file, "s/$(cutoffs[index])/" * label)

    return s 
end 

# read flow of structure factor at a given momentum
function read_structure_factor_flow_at_momentum(
    file  :: HDF5.File,
    label :: String, 
    p     :: Vector{Float64}
    )     :: NTuple{2, Vector{Float64}}

    # filter out a sorted list of cutoffs
    list    = keys(file["s"])
    cutoffs = sort(parse.(Float64, list), rev = true)

    # locate closest momentum 
    k     = read(file, "k")
    dists = Float64[norm(k[:, i] .- p) for i in 1 : size(k, 2)]
    index = argmin(dists)
    println("Momentum was adjusted to k = $(k[:, index]).")

    # allocate array to store values
    s = zeros(Float64, length(cutoffs))

    # fill array with values at given momentum
    for i in eachindex(cutoffs)
        s[i] = read(file, "s/$(cutoffs[i])/" * label)[index]
    end

    return cutoffs, s 
end

# read momentum with maximum structure factor amplitude
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










