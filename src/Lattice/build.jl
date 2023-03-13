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
        name      :: String,
        size      :: Int64
        ;
        verbose   :: Bool = true,
        euclidean :: Bool = false
        )         :: Lattice

Returns lattice graph with maximum bond distance size from origin (Euclidean distance in units of the nearest neighbor norm for euclidean == true).
Use `lattice_avail` to print available lattices.
"""
function get_lattice(
    name      :: String,
    size      :: Int64
    ;
    verbose   :: Bool = true,
    euclidean :: Bool = false
    )         :: Lattice

    if verbose
        if euclidean 
            println("Building lattice $(name) with maximum Euclidean distance $(size) ...")
        else
            println("Building lattice $(name) with maximum bond distance $(size) ...")
        end
    end

    # get unitcell
    uc = get_unitcell(name)

    # get test sites
    test_sites, metric = get_test_sites(uc, euclidean)

    # assure that the lattice is at least as large as the test set
    if euclidean
        @assert metric <= size * norm(get_vec(test_sites[1].int + uc.bonds[1][1], uc)) "Lattice is too small to perform symmetry reduction."
    else 
        @assert metric <= size "Lattice is too small to perform symmetry reduction."
    end

    # get list of sites
    sites = get_sites(size, uc, euclidean)
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

# helper function to increase size of test set if required by model
function grow_test_sites!(
    l         :: Lattice,
    metric    :: Int64
    ;
    euclidean :: Bool = false
    )         :: Nothing

    if euclidean 
        # determine the maximum real space distance of the current test set 
        norm_current = maximum(Float64[norm(s.vec) for s in l.test_sites])

        # determine nearest neighbor distance 
        nn_distance = norm(get_vec(l.sites[1].int + l.uc.bonds[1][1], l.uc))

        # ensure that the test set is not shrunk 
        if norm_current < metric * nn_distance
            println("   Increasing size of test set ...")
    
            # get new test sites with required euclidean distance
            test_sites_new = get_sites(metric, l.uc, euclidean)
    
            # add to current test set
            for s in test_sites_new
                if is_in(s, l.test_sites) == false
                    push!(l.test_sites, s)
                end
            end
    
            println("   Done. Lattice test sites have maximum euclidean distance $(metric).")
        end 
    else 
        # determine the maximum bond distance of the current test set
        metric_current = maximum(Int64[get_metric(l.test_sites[1], s, l.uc) for s in l.test_sites])

        # ensure that the test set is not shrunk
        if metric_current < metric
            println("   Increasing size of test set ...")

            # get new test sites with required bond distance
            test_sites_new = get_sites(metric, l.uc, euclidean)

            # add to current test set
            for s in test_sites_new
                if is_in(s, l.test_sites) == false
                    push!(l.test_sites, s)
                end
            end

            println("   Done. Lattice test sites have maximum bond distance $(metric).")
        end
    end

    return nothing
end

# load models
include("model_lib/model_heisenberg.jl")
include("model_lib/model_breathing.jl")
include("model_lib/model_triangular_dm_c3.jl")
include("model_lib/model_pyrochlore_hkg.jl")
include("model_lib/model_honeycomb_hkg.jl")
include("model_lib/model_pyrochlore_local.jl")


# print available models
function model_avail() :: Nothing

    println("##################")
    println("su2 models")
    println()
    println("heisenberg")
    println("breathing")
    println("pyrochlore-breathing-c3")
    println("pyrochlore_local")
    println("##################")

    println()

    println("##################")
    println("u1-dm models")
    println()
    println("triangular-dm-c3")
    println()
    println("pyrochlore-su2-hkg")
    println("honeycomb-hkg")
    println("##################")

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

Initialize model on a given lattice by modifying the respective bonds. Use `model_avail` to print available models.
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
    elseif name == "pyrochlore-breathing-c3"
        init_model_pyrochlore_breathing_c3!(J, l)
    elseif name == "triangular-dm-c3"
        init_model_triangular_dm_c3!(J, l)
    elseif name == "pyrochlore_hkg" 
        init_model_pyrochlore_hkg!(J, l)
    elseif name == "honeycomb_hkg"
        init_model_honeycomb_hkg!(J, l)
    elseif name == "pyrochlore-local"
        init_model_pyrochlore_local!(J, l)
    else
        error("Model $(name) unknown.")
    end
    
    return nothing
end

"""
    get_site(
        vec :: SVector{3, Float64},
        l   :: Lattice
        )   :: Int64

Search for a site in lattice graph, returns respective index in l.sites or 0 in case of failure.
"""
function get_site(
    vec :: SVector{3, Float64},
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