# auxiliary function to compute correlation integrals 
function integrate_χ_boxes(
    integrand :: Function,
    width     :: Float64,
    χ_tol     :: NTuple{2, Float64}
    )         :: Float64

    # map width to interval [0.0, 1.0]
    widthp = (width - 2.0 + sqrt(width * width + 4.0)) / (2.0 * width)

    # determine box size 
    size = abs(widthp - 0.5)

    # perform integation for top line
    I  = hcubature_v((vv, buff) -> integrand(vv, buff), Float64[       0.0, 0.5 + size], Float64[0.5 - size,        1.0], abstol = χ_tol[1], reltol = χ_tol[2])[1]
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[0.5 - size, 0.5 + size], Float64[0.5 + size,        1.0], abstol = χ_tol[1], reltol = χ_tol[2])[1]
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[0.5 + size, 0.5 + size], Float64[       1.0         1.0], abstol = χ_tol[1], reltol = χ_tol[2])[1]

    # perform integation for middle line
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[       0.0, 0.5 - size], Float64[0.5 - size, 0.5 + size], abstol = χ_tol[1], reltol = χ_tol[2])[1]
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[0.5 - size, 0.5 - size], Float64[0.5 + size, 0.5 + size], abstol = χ_tol[1], reltol = χ_tol[2])[1]
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[0.5 + size, 0.5 - size], Float64[       1.0, 0.5 + size], abstol = χ_tol[1], reltol = χ_tol[2])[1]

    # perform integation for bottom line
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[       0.0,        0.0], Float64[0.5 - size, 0.5 - size], abstol = χ_tol[1], reltol = χ_tol[2])[1]
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[0.5 - size,        0.0], Float64[0.5 + size, 0.5 - size], abstol = χ_tol[1], reltol = χ_tol[2])[1]
    I += hcubature_v((vv, buff) -> integrand(vv, buff), Float64[0.5 + size,        0.0], Float64[       1.0, 0.5 - size], abstol = χ_tol[1], reltol = χ_tol[2])[1]

    return I 
end

# load correlations for different symmetries 
include("correlation_su2.jl")
include("correlation_u1_dm.jl")

# generate correlation dummy 
function get_χ_empty(
    symmetry :: String,
    r        :: Reduced_lattice,
    m        :: Mesh
    )        :: Vector{Matrix{Float64}}

    if symmetry == "su2"
        return get_χ_su2_empty(r, m)
    elseif symmetry == "u1-dm"
        return get_χ_u1_dm_empty(r, m)
    end 
end