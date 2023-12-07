# auxiliary function to compute equal time correlator for u1-dm symmetry from correlations
function compute_eq_corr_u1_dm(
    m :: Mesh,
    χ :: Vector{Matrix{Float64}}
    ) :: Float64

    # compute equal-time on-site correlation 
    eq_corr = 0.0
    grid    = m.χ

    @turbo for i in 1 : length(grid) - 1 
        eq_corr += 2.0 * (grid[i + 1] - grid[i]) * (χ[1][1, i + 1] + χ[1][1, i])
        eq_corr += 1.0 * (grid[i + 1] - grid[i]) * (χ[2][1, i + 1] + χ[2][1, i])
    end 

    # normalization from Fourier transformation
    eq_corr /= 2.0 * pi 

    return eq_corr
end

# auxiliary function to compute equal time correlator for u1-dm symmetry from file
function compute_eq_corr_u1_dm(
    file    :: HDF5.File,
    Λ       :: Float64
    ;
    verbose :: Bool = true
    )       :: Float64

    # load diagonal correlations 
    m, χxx = read_χ(file, Λ, "xx", verbose = verbose)
    m, χzz = read_χ(file, Λ, "zz", verbose = verbose)

    # compute equal-time on-site correlation 
    eq_corr = 0.0

    @turbo for i in 1 : length(m) - 1 
        eq_corr += 2.0 * (m[i + 1] - m[i]) * (χxx[1, i + 1] + χxx[1, i])
        eq_corr += 1.0 * (m[i + 1] - m[i]) * (χzz[1, i + 1] + χzz[1, i])
    end 

    # normalization from Fourier transformation
    eq_corr /= 2.0 * pi 

    return eq_corr
end