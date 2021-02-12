# define unitcell struct
struct unitcell
    # list of basis sites
    basis :: Vector{Vector{Float64}}
    # list of Bravais vectors
    vectors :: Vector{Vector{Float64}}
    # list of bonds for each basis site (in Bravais vectors and basis index)
    bonds :: Vector{Vector{Vector{Int64}}}
end

# load unitcells 
include("unitcell_lib/square.jl")
include("unitcell_lib/honeycomb.jl")
include("unitcell_lib/kagome.jl")
include("unitcell_lib/cubic.jl")
include("unitcell_lib/fcc.jl")
include("unitcell_lib/hyperhoneycomb.jl")
include("unitcell_lib/pyrochlore.jl")
include("unitcell_lib/diamond.jl")

# interface function to load unitcell 
function get_unitcell(
    name :: String
    )    :: unitcell 

    if name == "square"
        return get_unitcell_square()
    elseif name == "honeycomb"
        return get_unitcell_honeycomb()
    elseif name == "kagome"
        return get_unitcell_kagome() 
    elseif name == "cubic"
        return get_unitcell_cubic()
    elseif name == "fcc"
        return get_unitcell_fcc()
    elseif name == "hyperhoneycomb"
        return get_unitcell_hyperhoneycomb()
    elseif name == "pyrochlore"
        return get_unitcell_pyrochlore()
    elseif name == "diamond"
        return get_unitcell_diamond()
    else
        error("Unitcell $(name) unknown.")
    end 
end




