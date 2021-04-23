"""
    unitcell

Struct containing the positions of basis sites, primitive translations and bonds for a lattice graph.
* `basis   :: Vector{Vector{Float64}}`       : position of basis sites in unitcell. basis[1] has to be the origin. 
* `vectors :: Vector{Vector{Float64}}`       : primitive translations of the lattice
* `bonds   :: Vector{Vector{Vector{Int64}}}` : bonds connecting basis sites
Use `get_unitcell` to load the unitcell for a specific lattice and `lattice_avail` to print available lattices.
"""
struct unitcell
    basis   :: Vector{Vector{Float64}}
    vectors :: Vector{Vector{Float64}}
    bonds   :: Vector{Vector{Vector{Int64}}}
end

# generate unitcell dummy 
function get_unitcell_empty()

    basis   = Vector{Vector{Float64}}(undef, 1)
    vectors = Vector{Vector{Float64}}(undef, 1)
    bonds   = Vector{Vector{Vector{Int64}}}(undef, 1)
    uc      = unitcell(basis, vectors, bonds)

    return uc 
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

Returns unitcell for lattice name. Use `lattice_avail` to print available lattices.
"""
function get_unitcell(
    name :: String
    )    :: unitcell

    uc = get_unitcell_empty()

    if name == "square"
        uc = get_unitcell_square()
    elseif name == "honeycomb"
        uc = get_unitcell_honeycomb()
    elseif name == "kagome"
        uc = get_unitcell_kagome()
    elseif name == "triangular"
        uc = get_unitcell_triangular()
    elseif name == "cubic"
        uc = get_unitcell_cubic()
    elseif name == "fcc"
        uc = get_unitcell_fcc()
    elseif name == "bcc"
        uc = get_unitcell_bcc()
    elseif name == "hyperhoneycomb"
        uc = get_unitcell_hyperhoneycomb()
    elseif name == "pyrochlore"
        uc = get_unitcell_pyrochlore()
    elseif name == "diamond"
        uc = get_unitcell_diamond()
    elseif name == "hyperkagome"
        uc = get_unitcell_hyperkagome()
    else
        error("Unitcell $(name) unknown.")
    end
end