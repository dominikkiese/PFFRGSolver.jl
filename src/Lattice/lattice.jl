"""
    lattice

Struct containing the unitcell, sites and bonds of a lattice graph.
Additionally a set of sites to verify symmetry transformations is provided.
"""
struct lattice
    name       :: String
    size       :: Int64
    uc         :: unitcell
    test_sites :: Vector{site}
    sites      :: Vector{site}
    bonds      :: Matrix{bond}
end

"""
    get_lattice(
        name :: String,
        size :: Int64
        )    :: lattice

Returns lattice graph with maximum bond distance size from origin.
Use `lattice_avail()` to print available lattices.
"""
function get_lattice(
    name :: String,
    size :: Int64
    )    :: lattice

    println("Building lattice $(name) with maximum bond distance $(size), this may take a while ...")

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
    bonds = Matrix{bond}(undef, num, num)

    for i in 1 : num
        for j in 1 : num
            bonds[i, j] = get_bond_empty(i, j)
        end
    end

    # build lattice
    l = lattice(name, size, uc, test_sites, sites, bonds)

    println("Done. Lattice has $(length(l.sites)) sites.")

    return l
end

# load models
include("model_lib/model_heisenberg.jl")
include("model_lib/model_j1_j2_j3a_pyrochlore.jl")

# print available models
function model_avail() :: Nothing

    println()
    println("#--------------------- SU(2) symmetric models ---------------------#")
    println("heisenberg")
    println("j1_j2_j3a_pyrochlore")
    println()
    println("Documentation provided by `?init_model_<model_name>!`.")
    println()

    return nothing
end

"""
    init_model!(
        name :: String,
        J    :: Vector{<:Any},
        l    :: lattice
        )    :: Nothing

Init model on a given lattice by overwriting the respective bonds. Use `model_avail()` to print available models.
Details about the layout of the coupling vector J can be found with `?init_model_<model_name>!`.
"""
function init_model!(
    name :: String,
    J    :: Vector{<:Any},
    l    :: lattice
    )    :: Nothing

    if name == "heisenberg"
        init_model_heisenberg!(J, l)
    elseif name == "j1-j2-j3a-pyrochlore"
        init_model_j1_j2_j3a_pyrochlore!(J, l)
    else
        error("Model $(name) unknown.")
    end

    return nothing
end

"""
    get_site(
        vec :: Vector{Float64},
        l   :: lattice
        )   :: Int64

Search for a site in lattice graph, returns respective index in l.sites or 0 in case of failure.
"""
function get_site(
    vec :: Vector{Float64},
    l   :: lattice
    )   :: Int64

    index = 0

    for i in eachindex(l.sites)
        if norm(vec - l.sites[i].vec) <= 1e-10
            index = i
            break
        end
    end

    return index
end

"""
    get_bond(
        s1 :: site,
        s2 :: site,
        l  :: lattice
        )  :: bond

Returns bond between (s1, s2) from bond list of lattice graph.
"""
function get_bond(
    s1 :: site,
    s2 :: site,
    l  :: lattice
    )  :: bond

    # get indices of the sites
    i1 = get_site(s1.vec, l)
    i2 = get_site(s2.vec, l)

    # get bond from lattice bonds
    b = l.bonds[i1, i2]

    return b
end
