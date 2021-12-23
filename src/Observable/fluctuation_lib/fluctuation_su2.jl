# auxiliary function to compute equal time correlator for su2 symmetry
function compute_eq_corr_su2(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Float64

    # load diagonal correlations 
    m, χ = read_χ(file, Λ, "diag", verbose = verbose)

    # compute equal-time on-site correlation 
    eq_corr = 0.0

    for i in 1 : length(m) - 1 
        eq_corr += (m[i + 1] - m[i]) * (χ[1, i + 1] + χ[1, i])
    end 

    # calculate full correlator
    eq_corr *= 3.0

    # normalization from Fourier transformation
    eq_corr /= 2.0 * pi

    return eq_corr
end