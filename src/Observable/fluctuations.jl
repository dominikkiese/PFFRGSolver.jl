# load fluctuation implementations
include("fluctuation_lib/fluctuation_su2.jl")
include("fluctuation_lib/fluctuation_u1_dm.jl")

# auxiliary function to compute occupation number fluctuations from correlations
function compute_fluctuations(
    symmetry :: String,
    m        :: Mesh,
    χ        :: Vector{Matrix{Float64}}
    )        :: Float64

    # init number fluctuations
    eq_corr = 0.0

    if symmetry == "su2"
        eq_corr = compute_eq_corr_su2(m, χ)
    elseif symmetry == "u1-dm"
        eq_corr = compute_eq_corr_u1_dm(m, χ)
    end 

    # compute variance of occupation number operator
    var = 1.0 - 4.0 * eq_corr / 3.0

    return var
end

# auxiliary function to compute occupation number fluctuations from file
function compute_fluctuations(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Float64

    # read symmetry 
    symmetry = read(file["symmetry"])

    # init number fluctuations
    eq_corr = 0.0

    if symmetry == "su2"
        eq_corr = compute_eq_corr_su2(file, Λ, verbose = verbose)
    elseif symmetry == "u1-dm"
        eq_corr = compute_eq_corr_u1_dm(file, Λ, verbose = verbose)
    end 

    # compute variance of occupation number operator
    var = 1.0 - 4.0 * eq_corr / 3.0

    return var
end

# auxiliary function to compute flow of occupation number fluctuations from file
function compute_fluctuations_flow(
    file :: HDF5.File,
    )    :: NTuple{2, Vector{Float64}}

    # filter out a sorted list of cutoffs
    list    = keys(file["χ"])
    cutoffs = sort(parse.(Float64, list), rev = true)

    # allocate array to store values
    vars = zeros(Float64, length(cutoffs))

    # fill array with values
    for i in eachindex(cutoffs)
        vars[i] = compute_fluctuations(file, cutoffs[i], verbose = false)
    end

    return cutoffs, vars 
end