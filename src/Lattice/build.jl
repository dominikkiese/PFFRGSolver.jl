"""
    Lattice

Struct containing the unitcell, sites and bonds of a lattice graph.
Additionally a set of sites to verify symmetry transformations is provided.
* `name       :: String`       : name of the lattice
* `size       :: Int64`        : bond truncation of the lattice
* `uc         :: Unitcell`     : unitcell of the lattice
* `test_sites :: Vector{Site}` : minimal set of test sites to verify symmetry transformations
* `sites      :: Vector{Site}` : list of sites in the lattice
* `bonds      :: Matrix{Bond}` : matrix encoding the interactions between arbitrary lattice sites
"""
struct Lattice
    name       :: String
    size       :: Int64
    uc         :: Unitcell
    test_sites :: Vector{Site}
    sites      :: Vector{Site}
    bonds      :: Matrix{Bond}
end

"""
    get_lattice(
        name    :: String,
        size    :: Int64
        ;
        verbose :: Bool = true
        )       :: Lattice

Returns lattice graph with maximum bond distance size from origin.
Use `lattice_avail` to print available lattices.
"""
function get_lattice(
    name    :: String,
    size    :: Int64
    ;
    verbose :: Bool = true
    )       :: Lattice

    if verbose
        println("Building lattice $(name) with maximum bond distance $(size) ...")
    end

    # get unitcell
    uc = get_unitcell(name)

    # get test sites
    test_sites, metric = get_test_sites(uc)

    # assure that the lattice is at least as large as the test set
    @assert metric <= size "Lattice is too small to perform symmetry reduction."

    # get list of sites
    sites = get_sites(size, uc)
    num   = length(sites)

    # get empty bond matrix
    bonds = Matrix{Bond}(undef, num, num)

    for i in 1 : num
        for j in 1 : num
            bonds[i, j] = get_bond_empty(i, j)
        end
    end

    # build lattice
    l = Lattice(name, size, uc, test_sites, sites, bonds)

    if verbose
        println("Done. Lattice has $(length(l.sites)) sites.")
    end

    return l
end

# load models
include("model_lib/model_heisenberg.jl")
include("model_lib/model_breathing.jl")

# print available models
function model_avail() :: Nothing

    println("#--------------------- SU(2) symmetric models ---------------------#")
    println("heisenberg")
    println("breathing")
    println()
    println("Documentation provided by ?init_model_<model_name>!.")

    return nothing
end

"""
    init_model!(
        name :: String,
        J    :: Vector{Vector{Float64}},
        l    :: Lattice
        )    :: Nothing

Initialize model on a given lattice by overwriting the respective bonds. Use `model_avail` to print available models.
Details about the layout of the coupling vector J can be found with `?init_model_<model_name>!`.
"""
function init_model!(
    name :: String,
    J    :: Vector{Vector{Float64}},
    l    :: Lattice
    )    :: Nothing

    if name == "heisenberg"
        init_model_heisenberg!(J, l)
    elseif name == "breathing"
        init_model_breathing!(J, l)
    else
        error("Model $(name) unknown.")
    end

    return nothing
end

"""
    get_site(
        vec :: Vector{Float64},
        l   :: Lattice
        )   :: Int64

Search for a site in lattice graph, returns respective index in l.sites or 0 in case of failure.
"""
function get_site(
    vec :: Vector{Float64},
    l   :: Lattice
    )   :: Int64

    index = 0

    for i in eachindex(l.sites)
        if norm(vec - l.sites[i].vec) <= 1e-8
            index = i
            break
        end
    end

    return index
end

"""
    get_bond(
        s1 :: Site,
        s2 :: Site,
        l  :: Lattice
        )  :: Bond

Returns bond between (s1, s2) from bond list of lattice graph.
"""
function get_bond(
    s1 :: Site,
    s2 :: Site,
    l  :: Lattice
    )  :: Bond

    # get indices of the sites
    i1 = get_site(s1.vec, l)
    i2 = get_site(s2.vec, l)

    # get bond from lattice bonds
    b = l.bonds[i1, i2]

    return b
end
