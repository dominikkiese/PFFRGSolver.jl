# auxiliary function to compute occupation number fluctuations
function compute_fluctuations(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Float64

    # read symmetry 
    symmetry = read(file["symmetry"])

    # init number fluctuations
    var = 0.0

    if symmetry == "su2"
        # load diagonal correlations 
        m, χ = read_χ(file, Λ, "diag", verbose = verbose)

        # compute equal-time on-site correlation 
        eq_corr = 0.0

        for i in 1 : length(m) - 1 
            eq_corr += (m[i + 1] - m[i]) * (χ[1, i + 1] + χ[1, i])
        end 

        eq_corr /= 2.0 * pi 

        # compute variance of occupation number operator
        var = 1.0 - 4.0 * eq_corr
        
    elseif symmetry == "u1-dm"
        # load diagonal correlations 
        m, χxx = read_χ(file, Λ, "xx", verbose = verbose)
        m, χzz = read_χ(file, Λ, "zz", verbose = verbose)

        # compute equal-time on-site correlation 
        eq_corr = 0.0

        for i in 1 : length(m) - 1 
            eq_corr += 2.0 * (m[i + 1] - m[i]) * (χxx[1, i + 1] + χxx[1, i])
            eq_corr += 1.0 * (m[i + 1] - m[i]) * (χzz[1, i + 1] + χzz[1, i])
        end 

        eq_corr /= 2.0 * pi 

        # compute variance of occupation number operator
        var = 1.0 - 4.0 * eq_corr / 3.0
    end 

    return var
end

# auxiliary function to compute flow of occupation number fluctuations
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