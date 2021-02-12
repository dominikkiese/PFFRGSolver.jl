# define lattice struct 
struct lattice
    # name of the lattice
    name :: String 
    # size of the lattice
    size :: Int64 
    # unitcell of the lattice 
    uc :: unitcell 
    # test sites for symmetry transformations 
    test_sites :: Vector{site}
    # lattice sites 
    sites :: Vector{site}
    # interactions between different sites
    bonds :: Matrix{bond}
end

# build lattice from unitcell 
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

# interface function to initialize models by name 
function init_model!(
    name :: String,
    J    :: Vector{Float64},
    l    :: lattice
    )    :: Nothing

    if name == "heisenberg"
        init_model_heisenberg!(J, l)
    elseif name == "j1-j2-j3a-pyrochlore"
        init_model_j1_j2_j3a_pyrochlore!(J, l)
    end

    return nothing
end

# search for site in lattice within numerical tolerance, return 0 in case of failure
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

# search for bond between two lattice sites
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
