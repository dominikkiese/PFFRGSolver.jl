# auxiliary function to compute equal time correlator for diagonal z2 symmetry
function compute_eq_corr_z2_diag(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Float64

    # load diagonal correlations 
    m, χxx = read_χ(file, Λ, "xx", verbose = verbose)
    m, χyy = read_χ(file, Λ, "yy", verbose = verbose)
    m, χzz = read_χ(file, Λ, "zz", verbose = verbose)

    # compute equal-time on-site correlation 
    eq_corr = 0.0

    for i in 1 : length(m) - 1 
        eq_corr += (m[i + 1] - m[i]) * (χxx[1, i + 1] + χxx[1, i])
        eq_corr += (m[i + 1] - m[i]) * (χyy[1, i + 1] + χyy[1, i])
        eq_corr += (m[i + 1] - m[i]) * (χzz[1, i + 1] + χzz[1, i])
    end 

    # normalization from Fourier transformation
    eq_corr /= 2.0 * pi 

    return eq_corr
end