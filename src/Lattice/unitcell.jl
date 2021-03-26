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
include("unitcell_lib/triangular.jl")

# load custom 3D unitcells
include("unitcell_lib/cubic.jl")
include("unitcell_lib/fcc.jl")
include("unitcell_lib/bcc.jl")
include("unitcell_lib/hyperhoneycomb.jl")
include("unitcell_lib/pyrochlore.jl")
include("unitcell_lib/diamond.jl")
include("unitcell_lib/hyperkagome.jl")

# print available lattices
function lattice_avail() :: Nothing

    println()
    println("#--------------------- 2D Lattices ---------------------#")
    println("square")
    println("honeycomb")
    println("kagome")
    println("triangular")
    println()
    println("#--------------------- 3D Lattices ---------------------#")
    println("cubic")
    println("fcc")
    println("bcc")
    println("hyperhoneycomb")
    println("pyrochlore")
    println("diamond")
    println("hyperkagome")
    println()

    return nothing
end

"""
    get_unitcell(
        name :: String
        )    :: unitcell

Returns unitcell for lattice name. Use `lattice_avail()` to print available lattices.
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
    elseif name == "triangular"
        return get_unitcell_triangular()
    elseif name == "cubic"
        return get_unitcell_cubic()
    elseif name == "fcc"
        return get_unitcell_fcc()
    elseif name == "bcc"
        return get_unitcell_bcc()
    elseif name == "hyperhoneycomb"
        return get_unitcell_hyperhoneycomb()
    elseif name == "pyrochlore"
        return get_unitcell_pyrochlore()
    elseif name == "diamond"
        return get_unitcell_diamond()
    elseif name == "hyperkagome"
        return get_unitcell_hyperkagome()
    else
        error("Unitcell $(name) unknown.")
    end
end
