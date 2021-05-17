"""
    Site 

Struct containing the coordinates, parametrized via primitive translations plus basis index and in real space form, of a lattice site.
* `int :: Vector{Int64}`  : site given in term of primitive translation and basis index
* `vec :: Vector{Float64` : site given in cartesian coordinates
"""
struct Site 
    int :: Vector{Int64}
    vec :: Vector{Float64}
end

# transform int to vec representation 
function get_vec(
    int :: Vector{Int64},
    uc  :: unitcell
    )   :: Vector{Float64}

    vec   = int[1] .* uc.vectors[1]
    vec .+= int[2] .* uc.vectors[2]
    vec .+= int[3] .* uc.vectors[3]
    vec .+= uc.basis[int[4]]

    return vec 
end

# generate lattice sites from unitcell 
function get_sites(
    size :: Int64,
    uc   :: unitcell
    )    :: Vector{Site}

    # init buffers
    ints    = Vector{Int64}[Int64[0, 0, 0, 1]]
    current = copy(ints)
    touched = copy(ints)
    metric  = 0

    # iteratively add sites with bond distance 1 until required size is reached
    while metric < size
        # init list for new sites generated in this step
        new_ints = Vector{Int64}[]

        # add sites with bond distance 1 to new sites from last step
        for int in current
            for i in eachindex(uc.bonds[int[4]])
                new_int = int .+ uc.bonds[int[4]][i]

                # check if site was generated already
                if in(new_int, touched) == false
                    if in(new_int, ints) == false
                        push!(new_ints, new_int)
                        push!(ints, new_int)
                    end
                end
            end
        end
        
        # update lists and increment metric
        touched  = current
        current  = new_ints
        metric  += 1
    end

    # build sites
    sites = Site[Site(int, get_vec(int, uc)) for int in ints]

    return sites
end

"""
    get_metric(
        s1 :: Site,
        s2 :: Site,
        uc :: unitcell
        )  :: Int64

Compute the bond metric (i.e. minimal number of bonds required to connect two sites within the lattice graph) for a given unitcell.
""" 
function get_metric(
    s1 :: Site,
    s2 :: Site,
    uc :: unitcell
    )  :: Int64

    # init buffers
    current   = Vector{Int64}[s2.int]
    touched   = Vector{Int64}[s2.int]
    metric    = 0
    not_found = true

    # check if sites are identical
    if s1.int == s2.int
        not_found = false
    end

    # add sites with bond distance 1 around s2 until s1 is reached
    while not_found
        # increment metric
        metric += 1

        # init list for new sites generated in this step
        new_ints = Vector{Int64}[]

        # add sites with bond distance 1 to new sites from last step
        for int in current
            for i in eachindex(uc.bonds[int[4]])
                new_int = int .+ uc.bonds[int[4]][i]

                # check if site was touched already
                if in(new_int, touched) == false
                    push!(new_ints, new_int)
                end
            end
        end

        # if s1 is reached stop searching, otherwise update lists and continue
        if in(s1.int, new_ints)
            not_found = false
        else
            touched = current
            current = new_ints
        end
    end

    return metric
end

# check if Float64 item in list within numerical tolerance
function is_in(
    e    :: Float64,
    list :: Vector{Float64}
    )    :: Bool

    in = false

    for item in list
        if abs(item - e) <= 1e-8
            in = true
            break
        end
    end

    return in
end

"""
    get_nbs(
        n    :: Int64,
        s    :: Site,
        list :: Vector{Site}
        )    :: Vector{Int64}

Returns list of n-th nearest neigbors (Euclidean norm) for a given site s, assuming s in list.
"""
function get_nbs(
    n    :: Int64,
    s    :: Site,
    list :: Vector{Site}
    )    :: Vector{Int64}

    # init buffers
    dist        = Float64[]
    dist_unique = Float64[]

    # collect possible distances
    for item in list
        d = norm(item.vec - s.vec)
        if is_in(d, dist_unique) == false
            push!(dist_unique, d)
        end
        push!(dist, d)
    end

    # determine the n-th nearest neighbor distance
    dn = sort(dist_unique)[n + 1]

    # collect all sites with distance dn
    nbs = Int64[]

    for i in eachindex(dist)
        if abs(dn - dist[i]) <= 1e-8
            push!(nbs, i)
        end
    end

    return nbs
end

# check if site in list within numerical tolerance
function is_in(
    e    :: Site,
    list :: Vector{Site}
    )    :: Bool

    in = false

    for item in list
        if norm(item.vec - e.vec) <= 1e-8
            in = true
            break
        end
    end

    return in
end

# obtain minimal test set to verify symmetry transformations
function get_test_sites(
    uc :: unitcell
    )  :: Tuple{Vector{Site}, Int64}

    # init buffers
    test_sites = Site[]

    # add basis sites and connected neighbors to list
    for i in eachindex(uc.basis)
        b = Site(Int64[0, 0, 0, i], uc.basis[i])

        if is_in(b, test_sites) == false
            push!(test_sites, b)
        end
        
        for j in eachindex(uc.bonds[i])
            int = b.int .+ uc.bonds[i][j]
            bp  = Site(int, get_vec(int, uc))

            if is_in(bp, test_sites) == false
                push!(test_sites, bp)
            end
        end
    end

    # determine the maximum bond distance
    metric = maximum(Int64[get_metric(test_sites[1], s, uc) for s in test_sites])

    return test_sites, metric
end