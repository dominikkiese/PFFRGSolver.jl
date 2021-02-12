# define reduced lattice struct 
struct reduced_lattice 
    # irreducible lattice sites 
    sites :: Vector{site}
    # overlaps between sites 
    overlap :: Vector{Matrix{Int64}}
    # multiplicities of irreducible sites 
    mult :: Vector{Int64}
    # images of irreducible sites under site exchanges 
    exchange :: Vector{Int64}
    # projections of pairs of sites to irreducible sites 
    project :: Matrix{Int64}
end

# check if matrix in list of matrices within numerical tolerance
function is_in(
    e    :: Matrix{Float64},
    list :: Vector{Matrix{Float64}}
    )    :: Bool

    in = false

    for item in list
        if maximum(abs.(item .- e)) <= 1e-10
            in = true
            break
        end
    end

    return in
end

# rotate vector onto a reference vector using Rodrigues formula
function get_rotation(
    vec :: Vector{Float64},
    ref :: Vector{Float64}
    )   :: Matrix{Float64}

    # buffer geometric information
    a = vec ./ norm(vec)
    b = ref ./ norm(ref)
    n = cross(a, b)
    s = norm(n)
    c = dot(a, b)

    # allocate rotation matrix
    mat = Matrix{Float64}(I, 3, 3)

    # if vectors are antiparallel do inversion
    if s < 1e-10 && c < 0.0
        mat[1, 1] = -1.0
        mat[2, 2] = -1.0
        mat[3, 3] = -1.0
    end

    # if vectors are non-collinear use Rodrigues formula
    if s > 1e-10
        n    = n ./ s
        temp = zeros(Float64, 3, 3)

        temp[2, 1] =  n[3]
        temp[3, 1] = -n[2]
        temp[1, 2] = -n[3]
        temp[3, 2] =  n[1]
        temp[1, 3] =  n[2]
        temp[2, 3] = -n[1]

        mat .+= s .* temp .+ (1.0 - c) .* temp * temp
    end

    return mat
end

# try to rotate vector onto a reference vector around a given axis (return matrix of zeros in case of failure)
function get_rotation_for_axis(
    vec  :: Vector{Float64},
    ref  :: Vector{Float64},
    axis :: Vector{Float64}
    )    :: Matrix{Float64}

    # normalize vectors and axis
    a  = vec ./ norm(vec)
    b  = ref ./ norm(ref)
    ax = axis ./ norm(axis)

    # determine othogonal projections onto axis
    ovec = vec .- dot(vec, ax) .* ax
    ovec = ovec ./ norm(ovec)
    oref = ref .- dot(ref, ax) .* ax
    oref = oref ./ norm(oref)

    # buffer geometric information
    s = norm(cross(ovec, oref))
    c = dot(ovec, oref)

    # allocate rotation matrix
    mat = zeros(Float64, 3, 3)

    mat[1, 1] = ax[1]^2 + (1.0 - ax[1]^2) * c
    mat[2, 1] = ax[1] * ax[2] * (1.0 - c) + ax[3] * s
    mat[3, 1] = ax[1] * ax[3] * (1.0 - c) - ax[2] * s
    mat[1, 2] = ax[1] * ax[2] * (1.0 - c) - ax[3] * s
    mat[2, 2] = ax[2]^2 + (1.0 - ax[2]^2) * c
    mat[3, 2] = ax[2] * ax[3] * (1.0 - c) + ax[1] * s
    mat[1, 3] = ax[1] * ax[3] * (1.0 - c) + ax[2] * s
    mat[2, 3] = ax[2] * ax[3] * (1.0 - c) - ax[1] * s
    mat[3, 3] = ax[3]^2 + (1.0 - ax[3]^2) * c

    # check result
    if norm(mat * a .- b) > 1e-10
        mat .= 0.0
    end

    return mat
end

# try to obtain a rotation of (vec1, vec2) onto (ref1, ref2) (return matrix of zeros in case of failure)
function rotate(
    vec1 :: Vector{Float64},
    vec2 :: Vector{Float64},
    ref1 :: Vector{Float64},
    ref2 :: Vector{Float64}
    )    :: Matrix{Float64}

    # rotate vec1 onto ref1
    mat = get_rotation(vec1, ref1)

    # try to rotate vec2 around ref1 onto ref2
    mat = get_rotation_for_axis(mat * vec2, ref2, ref1) * mat

    return mat
end

# compute symmetry transformations which leave the origin of the lattice invariant
function get_trafos_orig(
    l :: lattice
    ) :: Vector{Matrix{Float64}}

    # allocate list for transformations, set reference site and its bonds
    trafos = Matrix{Float64}[]
    ref    = l.sites[1]
    con    = l.uc.bonds[1]

    # get a pair of non-collinear neighbors of origin
    for i1 in eachindex(con)
        for i2 in eachindex(con)
            # check that sites are unequal
            if i1 == i2
                continue
            end

            # get connected sites in Bravais representation
            int_i1 = ref.int .+ con[i1]
            int_i2 = ref.int .+ con[i2]

            # get connected sites in real space representation
            vec_i1 = get_vec(int_i1, l.uc)
            vec_i2 = get_vec(int_i2, l.uc)

            # check that sites are non-collinear
            if norm(cross(vec_i1, vec_i2)) <= 1e-10
                continue
            end

            # get site structs
            site_i1 = site(int_i1, vec_i1)
            site_i2 = site(int_i2, vec_i2)

            # get bonds
            bond_i1 = get_bond(ref, site_i1, l)
            bond_i2 = get_bond(ref, site_i2, l)

            # get another pair of non-collinear neighbors of origin
            for j1 in eachindex(con)
                for j2 in eachindex(con)
                    # check that sites are unequal
                    if j1 == j2
                        continue
                    end

                    # get connected sites in Bravais representation
                    int_j1 = ref.int .+ con[j1]
                    int_j2 = ref.int .+ con[j2]

                    # get connected sites in real space representation
                    vec_j1 = get_vec(int_j1, l.uc) 
                    vec_j2 = get_vec(int_j2, l.uc)

                    # check that sites are non-collinear
                    if norm(cross(vec_j1, vec_j2)) <= 1e-10
                        continue
                    end

                    # get site structs
                    site_j1 = site(int_j1, vec_j1)
                    site_j2 = site(int_j2, vec_j2)

                    # get bonds
                    bond_j1 = get_bond(ref, site_j1, l)
                    bond_j2 = get_bond(ref, site_j2, l)

                    # check that the bonds match
                    if are_equal(bond_i1, bond_j1) == false || are_equal(bond_i2, bond_j2) == false
                        continue
                    end

                    # try to obtain rotation
                    mat = rotate(site_i1.vec, site_i2.vec, site_j1.vec, site_j2.vec)

                    # check if successfull
                    if maximum(abs.(mat)) <= 1e-10
                        continue
                    end

                    # check if the trafo is already known
                    if is_in(mat, trafos)
                        continue
                    end

                    # verify transformation on test set
                    valid = true

                    for n in eachindex(l.test_sites)
                        # apply transformation to lattice site
                        mapped_vec = mat * l.test_sites[n].vec
                        orig_bond  = get_bond(ref, l.test_sites[n], l)
                        mapped_ind = get_site(mapped_vec, l)

                        # check if resulting site is in lattice and if the bonds match
                        if mapped_ind == 0
                            valid = false
                        else
                            valid = are_equal(orig_bond, get_bond(ref, l.sites[mapped_ind], l))
                        end

                        # break if test fails for one test site 
                        if valid == false 
                            break 
                        end
                    end

                    # save transformation
                    if valid
                        push!(trafos, mat)
                    end

                    # check if the trafo, combined with an inversion, is already known
                    if is_in(-mat, trafos)
                        continue
                    end

                    # verify transformation on test set
                    valid = true

                    for n in eachindex(l.test_sites)
                        # apply transformation, combined with an inversion, to lattice site
                        mapped_vec = -mat * l.test_sites[n].vec
                        orig_bond  = get_bond(ref, l.test_sites[n], l)
                        mapped_ind = get_site(mapped_vec, l)

                        # check if resulting site is in lattice and if the bonds match
                        if mapped_ind == 0
                            valid = false
                        else
                            valid = are_equal(orig_bond, get_bond(ref, l.sites[mapped_ind], l))
                        end

                        # break if test fails for one test site 
                        if valid == false 
                            break 
                        end
                    end

                    # save transformation
                    if valid
                        push!(trafos, -mat)
                    end
                end
            end
        end
    end

    return trafos
end

# compute reduced representation of the lattice
function get_reduced(
    l :: lattice
    ) :: Vector{Int64}

    # allocate a list of indices
    reduced = Int64[i for i in eachindex(l.sites)]

    # get transformations
    trafos = get_trafos_orig(l)

    # iterate over sites and try to find sites which are symmetry equivalent
    for i in 2 : length(reduced)
        # continue if index has already been replaced
        if reduced[i] < i
            continue
        end

        # apply all available transformations to current site and see where it is mapped
        for j in eachindex(trafos)
            mapped_vec = trafos[j] * l.sites[i].vec

            # check that site is not mapped to itself
            if norm(mapped_vec .- l.sites[i].vec) > 1e-10
                # determine image of site
                index = get_site(mapped_vec, l)

                # assert that the image is valid, otherwise this is not a valid transformation and our algorithm failed
                @assert index > 0 "Validity on test set could not be generalized."

                # replace site in list if bonds match
                if are_equal(l.bonds[1, i], get_bond(l.sites[1], l.sites[index], l))
                    reduced[index] = i
                end
            end
        end
    end

    return reduced
end

# compute symmetry transformations inside unitcell
function get_trafos_uc(
    l :: lattice
    ) :: Vector{Tuple{Matrix{Float64}, Bool}}

    # allocate list for transformations, set reference site and its bonds
    trafos = Vector{Tuple{Matrix{Float64}, Bool}}(undef, length(l.uc.basis) - 1)
    ref    = l.sites[1]
    con    = l.uc.bonds[1]

    # iterate over basis sites and find a symmetry for each of them
    for b in 2 : length(l.uc.basis)
        # set basis site and connections
        int       = Int64[0, 0, 0, b]
        basis     = site(int, get_vec(int, l.uc))
        con_basis = l.uc.bonds[b]

        # get a pair of non-collinear neighbors of basis
        for b1 in eachindex(con_basis)
            for b2 in eachindex(con_basis)
                # check that sites are unequal
                if b1 == b2
                    continue
                end

                # get connected sites in Bravais representation
                int_b1 = basis.int .+ con_basis[b1]
                int_b2 = basis.int .+ con_basis[b2]

                # get connected sites in real space representation
                vec_b1 = get_vec(int_b1, l.uc)
                vec_b2 = get_vec(int_b2, l.uc)

                # check that sites are non-collinear
                if norm(cross(vec_b1 .- basis.vec, vec_b2 .- basis.vec)) <= 1e-10
                    continue
                end

                # get site structs
                site_b1 = site(int_b1, vec_b1)
                site_b2 = site(int_b2, vec_b2)

                # get bonds
                bond_b1 = get_bond(basis, site_b1, l)
                bond_b2 = get_bond(basis, site_b2, l)

                # get a pair of non-collinear neigbors of origin
                for ref1 in eachindex(con)
                    for ref2 in eachindex(con)
                        # check that sites are unequal
                        if ref1 == ref2
                            continue
                        end

                        # get connected sites in Bravais representation
                        int_ref1 = ref.int .+ con[ref1]
                        int_ref2 = ref.int .+ con[ref2]

                        # get connected sites in real space representation
                        vec_ref1 = get_vec(int_ref1, l.uc)
                        vec_ref2 = get_vec(int_ref2, l.uc)

                        # check that sites are non-collinear
                        if norm(cross(vec_ref1, vec_ref2)) <= 1e-10
                            continue
                        end

                        # get site structs
                        site_ref1 = site(int_ref1, vec_ref1)
                        site_ref2 = site(int_ref2, vec_ref2)

                        # get bonds
                        bond_ref1 = get_bond(ref, site_ref1, l)
                        bond_ref2 = get_bond(ref, site_ref2, l)

                        # check that the bonds match
                        if are_equal(bond_b1, bond_ref1) == false || are_equal(bond_b2, bond_ref2) == false
                            continue
                        end

                        # try shift -> rotation
                        mat = rotate(site_b1.vec .- basis.vec, site_b2.vec .- basis.vec, site_ref1.vec, site_ref2.vec)

                        # check if successfull
                        if maximum(abs.(mat)) <= 1e-10
                            continue
                        end

                        # verify transformation on test set
                        valid = true

                        for n in eachindex(l.test_sites)
                            # test only those sites which are in range of basis
                            if get_metric(l.test_sites[n], basis, l.uc) <= l.size
                                # apply transformation to lattice site
                                mapped_vec = mat * (l.test_sites[n].vec .- basis.vec)
                                orig_bond  = get_bond(basis, l.test_sites[n], l)
                                mapped_ind = get_site(mapped_vec, l)

                                # check if resulting site is in lattice and if bonds match
                                if mapped_ind == 0
                                    valid = false
                                else
                                    valid = are_equal(orig_bond, get_bond(ref, l.sites[mapped_ind], l))
                                end

                                # break if test fails for one test site 
                                if valid == false
                                    break 
                                end
                            end
                        end

                        # save transformation
                        if valid
                            trafos[b - 1] = (mat, false)
                            break
                            break
                            break
                            break
                        end

                        # try inversion -> shift -> rotation
                        mat = rotate(-(site_b1.vec .- basis.vec), -(site_b2.vec .- basis.vec), site_ref1.vec, site_ref2.vec)

                        # check if successfull
                        if maximum(abs.(mat)) <= 1e-10
                            continue
                        end

                        # verify transformation on test set
                        valid = true

                        for n in eachindex(l.test_sites)
                            # test only those sites which are in range of basis
                            if get_metric(l.test_sites[n], basis, l.uc) <= l.size
                                # apply transformation to lattice site
                                mapped_vec = mat * (-l.test_sites[n].vec .+ basis.vec)
                                orig_bond  = get_bond(basis, l.test_sites[n], l)
                                mapped_ind = get_site(mapped_vec, l)
                                
                                # check if resulting site is in lattice and if bonds match
                                if mapped_ind == 0
                                    valid = false
                                else
                                    valid = are_equal(orig_bond, get_bond(ref, l.sites[mapped_ind], l))
                                end

                                # break if test fails for one test site 
                                if valid == false
                                    break 
                                end
                            end
                        end

                        # save transformation
                        if valid
                            trafos[b - 1] = (mat, true)
                            break
                            break
                            break
                            break
                        end
                    end
                end
            end
        end
    end

    return trafos
end

# compute mappings onto reduced lattice. Since this scales like number of sites cubed, it can be parallelized via multithreading
function get_mappings(
    l       :: lattice,
    reduced :: Vector{Int64}
    )       :: Matrix{Int64}

    # allocate matrix
    num = length(l.sites)
    mat = zeros(Int64, num, num)

    # get transformations
    trafos = get_trafos_uc(l)

    # compute entries of matrix
    @sync for i in eachindex(l.sites)
        @async Threads.@spawn begin
            # determine shift for second site
            si    = l.sites[i]
            b     = si.int[4]
            shift = get_vec(Int64[si.int[1], si.int[2], si.int[3], 1], l.uc)

            for j in eachindex(l.sites)
                sj         = l.sites[j]
                mapped_vec = sj.vec .- shift

                # perform transformation inside unitcell
                if b != 1
                    if trafos[b - 1][2]
                        mapped_vec = -1.0 .* mapped_vec
                        mapped_vec = mapped_vec .+ l.uc.basis[b]
                        mapped_vec = trafos[b - 1][1] * mapped_vec
                    else
                        mapped_vec = mapped_vec .- l.uc.basis[b]
                        mapped_vec = trafos[b - 1][1] * mapped_vec
                    end
                end

                # locate transformed site in lattice, if index = 0, metric between i and j exceeds lattice size
                index = get_site(mapped_vec, l)

                if index != 0
                    mat[i, j] = reduced[index]
                end
            end
        end
    end

    return mat
end

# compute irreducible sites in overlap of two sites
function get_overlap(
    l           :: lattice,
    reduced     :: Vector{Int64},
    irreducible :: Vector{Int64},
    mappings    :: Matrix{Int64}
    )           :: Vector{Matrix{Int64}}

    # allocate overlap
    overlap = Vector{Matrix{Int64}}(undef, length(irreducible))

    for i in eachindex(irreducible)
        # collect all sites in range of irreducible and origin
        temp = NTuple{2, Int64}[]

        for j in eachindex(l.sites)
            if mappings[irreducible[i], j] != 0
                push!(temp, (reduced[j], mappings[j, irreducible[i]]))
            end
        end

        # determine how often a certain pair occurs
        pairs = unique(temp)
        table = zeros(Int64, length(pairs), 3)

        for j in eachindex(pairs)
            # convert from original lattice index to new "irreducible" index
            table[j, 1] = findall(index -> index == pairs[j][1], irreducible)[1]
            table[j, 2] = findall(index -> index == pairs[j][2], irreducible)[1]

            # count multiplicity
            for k in eachindex(temp)
                if pairs[j] == temp[k]
                    table[j, 3] += 1
                end
            end
        end

        # save into overlap
        overlap[i] = table
    end

    return overlap
end

# compute multiplicity of irreducible sites
function get_mult(
    reduced     :: Vector{Int64},
    irreducible :: Vector{Int64}
    )           :: Vector{Int64}

    # allocate vector to store multiplicities
    mult = zeros(Float64, length(irreducible))

    for i in eachindex(mult)
        for j in eachindex(reduced)
            if reduced[j] == irreducible[i]
                mult[i] += 1
            end
        end
    end

    return mult
end

# compute mappings under site exchange
function get_exchange(
    irreducible :: Vector{Int64},
    mappings    :: Matrix{Int64}
    )           :: Vector{Int64}

    # allocate exchange list
    exchange = zeros(Int64, length(irreducible))

    # determine mappings under site exchange
    for i in eachindex(exchange)
        exchange[i] = findall(index -> index == mappings[irreducible[i], 1], irreducible)[1]
    end

    return exchange
end

# convert mapping table entries to irreducible site indices
function get_project(
    l           :: lattice,
    irreducible :: Vector{Int64},
    mappings    :: Matrix{Int64}
    )           :: Matrix{Int64}

    # allocate projections
    project = zeros(Int64, length(l.sites), length(l.sites))

    for i in eachindex(l.sites)
        for j in eachindex(l.sites)
            if mappings[i, j] != 0
                project[i, j] = findall(index -> index == mappings[i, j], irreducible)[1]
            end
        end
    end

    return project
end

# build reduced lattice
function get_reduced_lattice(
    l :: lattice
    ) :: reduced_lattice

    println("Performing symmetry reduction, this may take a while ...")

    # get reduced representation of lattice
    reduced     = get_reduced(l)
    irreducible = unique(reduced)
    sites       = site[site(l.sites[i].int, get_vec(l.sites[i].int, l.uc)) for i in irreducible]

    # get mapping table
    mappings = get_mappings(l, reduced)

    # get overlap
    overlap = get_overlap(l, reduced, irreducible, mappings)

    # get multiplicities
    mult = get_mult(reduced, irreducible)

    # get exchanges
    exchange = get_exchange(irreducible, mappings)

    # get projections
    project = get_project(l, irreducible, mappings)

    # build reduced lattice
    r = reduced_lattice(sites, overlap, mult, exchange, project)

    println("Done. Reduced lattice has $(length(r.sites)) sites.")

    return r
end


