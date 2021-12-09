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