"""
    unitcell 

Struct containing the basis vectors, primitive translations and bonds for a lattice graph.
"""
struct unitcell
    basis   :: Vector{Vector{Float64}}
    vectors :: Vector{Vector{Float64}}
    bonds   :: Vector{Vector{Vector{Int64}}}
end

# load custom 2D unitcells 
include("unitcell_lib/square.jl")
include("unitcell_lib/honeycomb.jl")
include("unitcell_lib/kagome.jl")

# load custom 3D unitcells
include("unitcell_lib/cubic.jl")
include("unitcell_lib/fcc.jl")
include("unitcell_lib/hyperhoneycomb.jl")
include("unitcell_lib/pyrochlore.jl")
include("unitcell_lib/diamond.jl")

# print available lattices 
function lattice_avail() :: Nothing

    println()
    println("#--------------------- 2D Lattices ---------------------#")
    println("square")
    println("honeycomb")
    println("kagome")
    println()
    println("#--------------------- 3D Lattices ---------------------#")
    println("cubic")
    println("fcc")
    println("hyperhoneycomb")
    println("pyrochlore")
    println("diamond")
    println()

    return nothing 
end

"""
    get_unitcell(
        name :: String
        )    :: unitcell

Returns unitcell for lattice name. Use lattice_avail() to print available lattices.
"""
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




