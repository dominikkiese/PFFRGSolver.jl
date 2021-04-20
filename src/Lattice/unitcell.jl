"""
    unitcell 

Struct containing the positions of basis sites, primitive translations and bonds for a lattice graph.
* `basis   :: Vector{Vector{Float64}}`       : position of basis sites in unitcell
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
    elseif name == "cubic"
        uc = get_unitcell_cubic()
    elseif name == "fcc"
        uc = get_unitcell_fcc()
    elseif name == "hyperhoneycomb"
        uc = get_unitcell_hyperhoneycomb()
    elseif name == "pyrochlore"
        uc = get_unitcell_pyrochlore()
    elseif name == "diamond"
        uc = get_unitcell_diamond()
    else
        error("Unitcell $(name) unknown.")
    end 

    @assert norm(uc.basis[1]) < 1e-10 "Invalid unitcell definition, `basis[1]` must be located at the origin."

    return uc
end