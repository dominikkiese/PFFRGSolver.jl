"""
    Reduced_lattice

Struct containing symmetry irreducible sites of a lattice graph.
* `name     :: String`                  : name of the original lattice
* `size     :: Int64`                   : bond truncation of the original lattice
* `model    :: String`                  : name of the initialized model
* `J        :: Vector{Vector{Float64}}` : coupling vector of the initialized model
* `sites    :: Vector{Site}`            : list of symmetry irreducible sites
* `overlap  :: Vector{Matrix{Int64}}`   : pairs of irreducible sites with their respective multiplicity in range of origin and another irreducible site
* `mult     :: Vector{Int64}`           : multiplicities of irreducible sites
* `exchange :: Vector{Int64}`           : images of the pair (origin, irreducible site) under site exchange
* `project  :: Matrix{Int64}`           : projections of pairs (site1, site2) of the original lattice to pair (origin, irreducible site)
"""
struct Reduced_lattice
    name     :: String
    size     :: Int64
    model    :: String
    J        :: Vector{Vector{Float64}}
    sites    :: Vector{Site}
    overlap  :: Vector{Vector{Tuple{Mapping, Mapping, Int64}}}
    mult     :: Vector{Int64}
    exchange :: Vector{Mapping}
    localmap :: Vector{Mapping}
    project  :: Matrix{Mapping}
end

# check if matrix in list of matrices within numerical tolerance
function is_in(
    e    :: SMatrix{3, 3, Float64},
    list :: Vector{SMatrix{3, 3, Float64}}
    )    :: Bool

    in = false

    for item in list
        if maximum(abs.(item .- e)) <= 1e-8
            in = true
            break
        end
    end

    return in
end

# check if e is in "list", only comparing the first value (the rotations matrix)
# needed for combined real space and spin transformations
function is_in(
    e,
    list 
    )    :: Bool

    in = false

    for item in list
        if maximum(abs.(item[1] .- e[1])) <= 1e-8
            in = true
            break
        end
    end

    return in
end

# rotate vector onto a reference vector using Rodrigues formula
function get_rotation(
    vec :: SVector{3, Float64},
    ref :: SVector{3, Float64}
    )   :: SMatrix{3, 3, Float64}

    # buffer geometric information
    a = vec ./ norm(vec)
    b = ref ./ norm(ref)
    n = cross(a, b)
    s = norm(n)
    c = dot(a, b)

    # check if vectors are collinear
    if s < 1e-8
        # if vector are antiparallel do inversion
        if c < 0.0
            return SMatrix{3, 3, Float64}(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
        # if vectors are parallel return unity
        else 
            return SMatrix{3, 3, Float64}(+1.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, +1.0)
        end 
    # if vectors are non-collinear use Rodrigues formula
    else
        n    = n ./ s
        temp = SMatrix{3, 3, Float64}(0.0, n[3], -n[2], -n[3], 0.0, n[1], n[2], -n[1], 0.0)
        mat  = SMatrix{3, 3, Float64}(+1.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, +1.0)
        mat  = mat .+ s .* temp .+ (1.0 - c) .* temp * temp

        return mat
    end
end

# try to rotate vector onto a reference vector around a given axis (return matrix of zeros in case of failure)
function get_rotation_for_axis(
    vec  :: SVector{3, Float64},
    ref  :: SVector{3, Float64},
    axis :: SVector{3, Float64}
    )    :: SMatrix{3, 3, Float64}

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
    mat = SMatrix{3, 3, Float64}(ax[1]^2 + (1.0 - ax[1]^2) * c,
                                 ax[1] * ax[2] * (1.0 - c) + ax[3] * s,
                                 ax[1] * ax[3] * (1.0 - c) - ax[2] * s,
                                 ax[1] * ax[2] * (1.0 - c) - ax[3] * s,
                                 ax[2]^2 + (1.0 - ax[2]^2) * c,
                                 ax[2] * ax[3] * (1.0 - c) + ax[1] * s,
                                 ax[1] * ax[3] * (1.0 - c) + ax[2] * s,
                                 ax[2] * ax[3] * (1.0 - c) - ax[1] * s,
                                 ax[3]^2 + (1.0 - ax[3]^2) * c)

    # check if rotation works as expected
    if norm(mat * a .- b) > 1e-8
        # check if sense of rotation has to be inverted
        if norm(transpose(mat) * a .- b) < 1e-8
            return transpose(mat)
        # if algorithm fails return matrix of zeros
        else
            return SMatrix{3, 3, Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        end
    # if sanity check is passed return rotation matrix
    else 
        return mat 
    end 
end

# try to obtain a rotation of (vec1, vec2) onto (ref1, ref2) (return matrix of zeros in case of failure)
function rotate(
    vec1 :: SVector{3, Float64},
    vec2 :: SVector{3, Float64},
    ref1 :: SVector{3, Float64},
    ref2 :: SVector{3, Float64}
    )    :: SMatrix{3, 3, Float64}

    # rotate vec1 onto ref1
    mat = get_rotation(vec1, ref1)

    # try to rotate vec2 around ref1 onto ref2
    mat = get_rotation_for_axis(mat * vec2, ref2, ref1) * mat

    return mat
end

# map bonds on bi1 and bi2 onto bonds on bj1 and bj2 with spin permutations + sign changes
function get_spin_trafo(bi1, bi2, bj1, bj2)

    #Try even spin permutations (with sign changes that don't violoate the su(2) algebra)
    for signs in [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]] 
        for perm in [[1, 2, 3], [2, 3, 1], [3, 1, 2]]   
         
            mapped_bi1 = apply_spin_trafo(bi1, perm, signs)
            mapped_bi2 = apply_spin_trafo(bi2, perm, signs)

            if mapped_bi1 == bj1 && mapped_bi2 == bj2
                return perm, signs
            end
        end
    end
    
    #Try odd spin permutations (with sign changes that don't violoate the su(2) algebra)
    for signs in [[-1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]] 
        for perm in [[3, 2, 1], [2, 1, 3], [1, 3, 2]]

            mapped_bi1 = apply_spin_trafo(bi1, perm, signs)
            mapped_bi2 = apply_spin_trafo(bi2, perm, signs)

            if mapped_bi1 == bj1 && mapped_bi2 == bj2
                return perm, signs
            end
        end
    end

    #If no transformation is found, return zeros
    return zeros(3), ones(3)
end

#Apply spin transformation: First apply signs, then permute.
function apply_spin_trafo(b, perm, signs)
    mapped_b = similar(b)

    for μ in axes(b, 1), ν in axes(b, 2)
        μp, νp = perm[μ], perm[ν]
        mapped_b[μ, ν] = signs[μp] * signs[νp] * b[μp, νp]
    end
    
    return mapped_b
end

function link_spin_trafos(p1, s1, p2, s2)
    #combine permutations
    p = p1[p2]

    #As signs are always applied FIRST, need to apply inverse p1 to s2
    p1_inv = zeros(Int64, 3)
    
    for i in eachindex(p1)
        p1_inv[p1[i]] = i
    end

    #Adjust s2 for permutation and combine with s1
    s = s2[p1_inv] .* s1
    return p, s
end

# invert spin transformation (only needed for testing)
function invert_spin_trafo(perm, signs)
    inv_perm = zeros(Int64, 3)
    
    for i in eachindex(perm)
        inv_perm[perm[i]] = i
    end
    return SVector{3}(inv_perm), SVector{3}(signs[perm])
end


function invert_spin_trafo(trafo)
    perm, signs = invert_spin_trafo(trafo[end-1], trafo[end])
    
    if length(trafo) == 3
        return (trafo[1], perm, signs)
    else
        return (trafo[1], trafo[2], perm, signs)
    end
end


"""
    get_trafos_orig(
        l :: Lattice
        ) :: Vector{SMatrix{3, 3, Float64}}

Compute transformations which leave the origin of the lattice invariant (point group symmetries).
"""
function get_trafos_orig(
    l :: Lattice
    ) :: Vector{Tuple{SMatrix{3, 3, Float64}, SVector{3, Int64}, SVector{3, Int64}}}

    # allocate list for transformations, set reference site and its bonds
    trafos = Tuple{SMatrix{3, 3, Float64}, SVector{3, Int64}, SVector{3, Int64}}[]
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
            if norm(cross(vec_i1, vec_i2)) <= 1e-8
                continue
            end

            # get site structs
            site_i1 = Site(int_i1, vec_i1)
            site_i2 = Site(int_i2, vec_i2)

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
                    if norm(cross(vec_j1, vec_j2)) <= 1e-8
                        continue
                    end

                    # get site structs
                    site_j1 = Site(int_j1, vec_j1)
                    site_j2 = Site(int_j2, vec_j2)

                    # get bonds
                    bond_j1 = get_bond(ref, site_j1, l)
                    bond_j2 = get_bond(ref, site_j2, l)

                    # try to obtain rotation
                    mat = rotate(site_i1.vec, site_i2.vec, site_j1.vec, site_j2.vec)
                    # if successful, try to find spin transformation to map bonds
                    if maximum(abs.(mat)) > 1e-8
                        
                        #Try to map bonds by applying spin permutations on j1 and j2
                        #perm, signs = get_spin_trafo(bond_j1.exchange, bond_j2.exchange, bond_i1.exchange, bond_i2.exchange)
                        perm, signs = get_spin_trafo(bond_i1.exchange, bond_i2.exchange, bond_j1.exchange, bond_j2.exchange)


                        # if successful, verify rotation + spin transformation on test set before saving it
                        if iszero(perm) == false
                            trafo = (mat, perm, signs)
                            
                            # check if rotation is already known
                            if is_in(trafo, trafos) == false

                                valid = true

                                for n in eachindex(l.test_sites)
                                    # apply transformation to lattice site
                                    mapped_vec = mat * l.test_sites[n].vec
                                    mapped_ind = get_site(mapped_vec, l)

                                    # check if resulting site is in lattice
                                    if mapped_ind == 0
                                        valid = false
                                    # check if bonds match after spin transformations
                                    else
                                        # get bond between reference and test site
                                        orig_bond  = deepcopy(get_bond(ref, l.test_sites[n], l))
                                        # apply spin transformation
                                        orig_bond.exchange .= apply_spin_trafo(orig_bond.exchange, perm, signs)
                                        
                                        # get bond between the mapped sites
                                        mapped_bond = get_bond(ref, l.sites[mapped_ind], l)
                                        
                                        #check if exchange on both bonds agrees
                                        valid = are_equal(orig_bond, mapped_bond)
                                    end

                                    # break if test fails for one test site
                                    if valid == false
                                        break
                                    end
                                end

                                # save transformation
                                if valid
                                    push!(trafos, trafo)
                                end
                            end

                            trafo = (-mat, perm, signs)

                            # check if rotation combined with inversion is already known
                            if is_in(trafo, trafos) == false
                                # verify rotation
                                valid = true

                                for n in eachindex(l.test_sites)
                                    # apply transformation to lattice site
                                    mapped_vec = -mat * l.test_sites[n].vec
                                    mapped_ind = get_site(mapped_vec, l)

                                    # check if resulting site is in lattice 
                                    if mapped_ind == 0
                                        valid = false
                                    #Check if bonds match after spin transformations
                                    else
                                        # get bond between reference and test site
                                        orig_bond  = deepcopy(get_bond(ref, l.test_sites[n], l))

                                        # apply spin transformation
                                        orig_bond.exchange .= apply_spin_trafo(orig_bond.exchange, perm, signs)
                                        
                                        # get bond between the mapped sites
                                        mapped_bond = get_bond(ref, l.sites[mapped_ind], l)
                                        
                                        #check if exchange on both bonds agrees
                                        valid = are_equal(orig_bond, mapped_bond)
                                    end

                                    # break if test fails for one test site
                                    if valid == false
                                        break
                                    end
                                end

                                # save transformation
                                if valid
                                    push!(trafos, trafo)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return trafos
end

# compute reduced representation of the lattice
function get_reduced(
    l :: Lattice
    ) :: NTuple{2, Vector{Int64}}

    # allocate a list of indices
    reduced = Int64[i for i in eachindex(l.sites)]
    trafo_index = ones(Int64, length(l.sites))

    # get transformations
    trafos = get_trafos_orig(l)

    # iterate over sites and try to find sites which are symmetry equivalent
    for i in 2 : length(reduced)

        #only consider sites in range of origin (others may get mapped outsite of the lattice)
        if get_metric(l.sites[1], l.sites[i], l.uc) > l.size
            #Set mapping to zero to mark out-of-range sites
            reduced[i] = 0
            trafo_index[i] = 0 
            continue
        end

        # continue if index has already been replaced
        if reduced[i] < i
            continue
        end

        # apply all available transformations to current site and see where it is mapped
        for j in eachindex(trafos)
            m, perm, signs = trafos[j]

            mapped_vec = m * l.sites[i].vec

            # check that site is not mapped to itself
            if norm(mapped_vec .- l.sites[i].vec) > 1e-8
                # determine image of site
                index = get_site(mapped_vec, l)

                # assert that the image is valid, otherwise this is not a valid transformation and our algorithm failed
                @assert index > 0 "Validity on test set could not be generalized."
                
                # Apply spin transformation
                exchange = apply_spin_trafo(l.bonds[1, i].exchange, perm, signs)
                
                # replace site in list if bonds match
                if norm(l.bonds[1, index].exchange .- exchange) <= 1e-8
                    reduced[index] = i
                    trafo_index[index] = j
                end
            end
        end
    end

    return reduced, trafo_index
end

"""
    get_trafos_uc(
        l :: Lattice
        ) :: Vector{Tuple{SMatrix{3, 3, Float64}, Bool}}

Compute mappings of a lattice's basis sites to the origin.
The mappings consist of a transformation matrix and a boolean indicating if an inversion was used or not.
"""
function get_trafos_uc(
    l :: Lattice
    ) :: Vector{Tuple{SMatrix{3, 3, Float64}, Bool, SVector{3, Int64}, SVector{3, Int64}}}

    # allocate list for transformations, set reference site and its bonds
    trafos = Vector{Tuple{SMatrix{3, 3, Float64}, Bool, SVector{3, Int64}, SVector{3, Int64}}}(undef, length(l.uc.basis) - 1)
    ref    = l.sites[1]
    con    = l.uc.bonds[1]

    # iterate over basis sites and find a symmetry for each of them
    for b in 2 : length(l.uc.basis)
        # set basis site and connections
        int       = SVector{4, Int64}(0, 0, 0, b)
        basis     = Site(int, get_vec(int, l.uc))
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
                if norm(cross(vec_b1 .- basis.vec, vec_b2 .- basis.vec)) <= 1e-8
                    continue
                end

                # get site structs
                site_b1 = Site(int_b1, vec_b1)
                site_b2 = Site(int_b2, vec_b2)

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
                        if norm(cross(vec_ref1, vec_ref2)) <= 1e-8
                            continue
                        end

                        # get site structs
                        site_ref1 = Site(int_ref1, vec_ref1)
                        site_ref2 = Site(int_ref2, vec_ref2)

                        # get bonds
                        bond_ref1 = get_bond(ref, site_ref1, l)
                        bond_ref2 = get_bond(ref, site_ref2, l)

                        # try to map reference bonds onto bonds that get rotated (we want to map from the origin to the different basis sites)
                        perm, signs = get_spin_trafo(bond_ref1.exchange, bond_ref2.exchange, bond_b1.exchange, bond_b2.exchange)
                        
                        # if unsuccesfull continue with next bonds
                        if iszero(perm) == true
                            continue
                        end

                        # try shift -> rotation
                        mat = rotate(site_b1.vec .- basis.vec, site_b2.vec .- basis.vec, site_ref1.vec, site_ref2.vec)

                        # if successful, verify tranformation on test set before saving it
                        if maximum(abs.(mat)) > 1e-8
                            valid = true
                            
                            # Steps for validations: 
                            # 1. select bond between basis site and test site
                            # 2. rotate the vectors of basis site and test site
                            # 3. apply spin trafo to bond on which they are rotated. Compare with original bond.

                            for b_test in 1 : length(l.uc.basis)
                                # set basis site and connections
                                int_test       = SVector{4, Int64}(0, 0, 0, b_test)
                                basis_test     = Site(int, get_vec(int_test, l.uc))

                                for n in eachindex(l.test_sites)
                                    # test only those sites which are in range of basis
                                    if get_metric(l.test_sites[n], basis_test, l.uc) <= l.size

                                        # apply transformation to lattice test_sites
                                        mapped_vec_basis = mat * (basis_test.vec .- basis.vec)
                                        mapped_ind_basis = get_site(mapped_vec_basis, l)

                                        mapped_vec = mat * (l.test_sites[n].vec .- basis.vec)
                                        mapped_ind = get_site(mapped_vec, l)

                                        # check if resulting site is in lattice and if bonds match
                                        if mapped_ind == 0 || mapped_ind_basis == 0
                                            valid = false
                                        else
                                            # get bond between the unrotated sites
                                            orig_bond  = get_bond(basis_test, l.test_sites[n], l)

                                            # get bond between the sites on which the original sites are rotated
                                            mapped_bond = deepcopy(get_bond(l.sites[mapped_ind_basis], l.sites[mapped_ind], l))

                                            # apply spin transformation on the mapped bond
                                            mapped_bond.exchange .= apply_spin_trafo(mapped_bond.exchange, perm, signs) 

                                            # confirm if bonds are equal
                                            valid = are_equal(orig_bond, mapped_bond)
                                        end

                                        # break if test fails for one test site
                                        if valid == false
                                            break
                                            break
                                        end
                                    end
                                end
                            end

                            # save transformation
                            if valid
                                trafos[b - 1] = (mat, false, perm, signs)
                                break
                                break
                                break
                                break
                            end
                        end

                        # try shift -> inversion -> rotation
                        mat = rotate(-(site_b1.vec .- basis.vec), -(site_b2.vec .- basis.vec), site_ref1.vec, site_ref2.vec)

                        # if successful, verify tranformation on test set before saving it
                        if maximum(abs.(mat)) > 1e-8
                            valid = true

                            for b_test in 1 : length(l.uc.basis)
                                # set basis site and connections
                                int_test       = SVector{4, Int64}(0, 0, 0, b_test)
                                basis_test     = Site(int, get_vec(int_test, l.uc))

                                for n in eachindex(l.test_sites)
                                    # test only those sites which are in range of basis
                                    if get_metric(l.test_sites[n], basis_test, l.uc) <= l.size

                                        # apply transformation to lattice test_sites
                                        mapped_vec_basis = mat * (-basis_test.vec .+ basis.vec)
                                        mapped_ind_basis = get_site(mapped_vec_basis, l)

                                        mapped_vec = mat * (-l.test_sites[n].vec .+ basis.vec)
                                        mapped_ind = get_site(mapped_vec, l)

                                        # check if resulting site is in lattice and if bonds match
                                        if mapped_ind == 0 || mapped_ind_basis == 0
                                            valid = false
                                        else
                                            # get bond between the unrotated sites
                                            orig_bond  = get_bond(basis_test, l.test_sites[n], l)

                                            # get bond between the sites on which the original sites are rotated
                                            mapped_bond = deepcopy(get_bond(l.sites[mapped_ind_basis], l.sites[mapped_ind], l))

                                            # apply spin transformation on the mapped bond
                                            mapped_bond.exchange .= apply_spin_trafo(mapped_bond.exchange, perm, signs) 

                                            # confirm if bonds are equal
                                            valid = are_equal(orig_bond, mapped_bond)
                                        end

                                        # break if test fails for one test site
                                        if valid == false
                                            break
                                            break
                                        end
                                    end
                                end
                            end

                            # save transformation
                            if valid
                                trafos[b - 1] = (mat, true, perm, signs)
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
    end

    return trafos
end

# apply transformation (i, j) -> (i0, j*), where i0 is the origin, to site j
function apply_trafo(
    s      :: Site, 
    b      :: Int64,
    shift  :: SVector{3, Float64},
    trafos ,
    l      :: Lattice
    )      :: SVector{3, Float64}

    # check if equivalent to origin
    if b != 1
        # if inequivalent to origin use transformation inside unitcell
        if trafos[b - 1][2]
            return trafos[b - 1][1] * (shift .- s.vec .+ l.uc.basis[b])
        else 
            return trafos[b - 1][1] * (s.vec .- shift .- l.uc.basis[b])
        end 
    # if equivalent to origin, only shift along translation vectors needs to be performed
    else 
        return s.vec .- shift
    end
end


# compute mappings from reduced lattice to full lattice
# For all sites (i, j) returns m = mappings[i,j] with
# m[1]: irreducible sites, m[2]: spin permutations, m[3]: sign flips
function get_mappings(
    l           :: Lattice,
    reduced     :: Vector{Int64},
    trafo_index :: Vector{Int64},
    symmetry    :: String
    )           ::  Matrix{Mapping}
   
    # allocate matrix
    num = length(l.sites)

    mappings = Matrix{Mapping}(undef, num, num)

    # get transformations inside unitcell
    trafos_uc = get_trafos_uc(l)

    trafos_orig = get_trafos_orig(l)

    
    # get distances to origin
    dists = Float64[norm(l.sites[i].vec) for i in eachindex(l.sites)]
    d_max = maximum(dists[reduced .!= 0]) #Maximal distance for site in range of origin (according to choosen metric)

    # group sites in shells
    shell_dists = unique(trunc.(dists, digits = 8))
    shells      = Vector{Vector{Int64}}(undef, length(shell_dists))

    for i in eachindex(shells)
        shell = Int64[]

        for j in eachindex(dists)
            if abs(dists[j] - shell_dists[i]) < 1e-8
                push!(shell, j)
            end
        end

        shells[i] = shell
    end

    # compute entries of matrix
    Threads.@threads for i in eachindex(l.sites)
        # determine shift for second site
        si    = l.sites[i]
        b     = si.int[4]
        shift = get_vec(SVector{4, Int64}(si.int[1], si.int[2], si.int[3], 1), l.uc)

        for j in eachindex(l.sites)
            site = 0
            perm = [1, 2, 3]
            sign = [1, 1, 1]

            # apply symmetry transformation
            mapped_vec = apply_trafo(l.sites[j], b, shift, trafos_uc, l)

            # locate transformed site in lattice
            index = 0
            d_map = norm(mapped_vec)

            # check if transformed site can be in lattice
            if d_max - d_map > -1e-8
                # find respective shell
                shell = shells[argmin(abs.(shell_dists .- d_map))]

                # find matching site in shell
                for k in shell
                    if norm(mapped_vec .- l.sites[k].vec) < 1e-8
                        index = k
                        break
                    end
                end

                # map site from lattice to irreducible site if in range of origin
                if index != 0 && reduced[index] != 0
                    #Map to irreducible site
                    site = reduced[index]
                    
                    #Determine spin transformation for the mapping (combined origin and uc trafo)

                    #Origin transformation
                    trafo_orig = trafos_orig[trafo_index[index]]

                    #First site is equivalent to origin, no uc transformation
                    if b == 1
                        perm .= trafo_orig[2]
                        sign .= trafo_orig[3]
                    #First site is inequivalent to origin, link with uc transformation
                    else
                        trafo_uc = trafos_uc[b-1]
                        #p, s = link_spin_trafos(trafo_uc[3], trafo_uc[4], trafo_orig[2], trafo_orig[3])
                        p, s = link_spin_trafos(trafo_orig[2], trafo_orig[3], trafo_uc[3], trafo_uc[4])

                        sign .= s
                        perm .= p
                    end
                end
            end

            mappings[i, j] = trafoToMapping(site, perm, sign, symmetry)
        end
    end

    return mappings
end



# compute irreducible sites in overlap of two sites
function get_overlap(
    l           :: Lattice,
    reduced     :: Vector{Int64},
    irreducible :: Vector{Int64},
    mappings    :: Matrix{Mapping}
    )           :: Vector{Vector{Tuple{Mapping, Mapping, Int64}}}

    # allocate overlap
    overlap = Vector{Vector{Tuple{eltype(mappings), eltype(mappings), Int64}}}(undef, length(irreducible))
    for i in eachindex(irreducible)
    
        # collect all sites in range of irreducible and origin
        temp = NTuple{2, eltype(mappings)}[]
    
        for j in eachindex(l.sites)
            # Neglect sites out of range from origin
            if reduced[j] != 0
                
                #Neglect sites out of range from irreducible
                if mappings[irreducible[i], j].site != 0
                    push!(temp, (mappings[1, j], mappings[j, irreducible[i]]))
                end
            end
        end
    
        # determine how often a certain pair occurs
        pairs = unique(temp)
        table = Vector{Tuple{eltype(mappings), eltype(mappings), Int64}}(undef, length(pairs))
    
    
        for j in eachindex(pairs)
            pair = pairs[j]
    
            # count multiplicity
            multis = 0
            for k in eachindex(temp)
                if pair == temp[k]
                    multis += 1
                end
            end
            # convert from original lattice index to new "irreducible" index
            pair1 = Mapping(findfirst(index -> index == pair[1].site, irreducible), pair[1].components, pair[1].signs)
            pair2 = Mapping(findfirst(index -> index == pair[2].site, irreducible), pair[2].components, pair[2].signs)
            table[j] = (pair1, pair2, multis)
        end
        
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
    mappings    :: Matrix{Mapping}
    )           :: Vector{Mapping}

    # allocate exchange list
    exchange = Vector{Mapping}(undef, length(irreducible))

    # determine mappings under site exchange
    for i in eachindex(exchange)

        #Determine mapped site
        mapping = mappings[irreducible[i], 1]
        
        #Map to irreducible index
        site = findfirst(index -> index == mapping.site, irreducible)
        
        #Save mapped site and spin mappings
        exchange[i] = Mapping(site, mapping.components, mapping.signs)
    end

    return exchange
end

function get_local(
    irreducible :: Vector{Int64},
    mappings    :: Matrix{Mapping}
)
    # allocate projections
    return 
    
    return project
end


# convert mapping table entries to irreducible site indices
function get_project(
    l           :: Lattice,
    irreducible :: Vector{Int64},
    mappings    :: Matrix{Mapping}
    )           :: Matrix{Mapping}

    # allocate projections
    project = Matrix{Mapping}(undef, length(l.sites), length(l.sites))

    for i in eachindex(l.sites)
        for j in eachindex(l.sites)
            #Get mapping
            mapping = mappings[i, j]

            #Map site to irreducible site if in reduced lattice
            if mapping.site != 0
                site = findfirst(index -> index == mapping.site, irreducible)
                project[i, j] = Mapping(site, mapping.components, mapping.signs)
            else
                project[i, j] = mapping
            end
        end
    end

    return project
end

"""
    get_reduced_lattice(
        model   :: String,
        J       :: Vector{Vector{Float64}},
        l       :: Lattice
        ;
        verbose :: Bool = true
        )       :: Reduced_lattice

Compute symmetry reduced representation of a given lattice graph with spin interactions between sites.
The interactions are defined by passing a model's name and coupling vector.
"""
function get_reduced_lattice(
    model    :: String,
    J        :: Vector{Vector{Float64}},
    l        :: Lattice,
    symmetry :: String
    ;
    verbose :: Bool = true
    )       :: Reduced_lattice

    if verbose
        println("Performing symmetry reduction ...")
    end

    # initialize model by modifying bond matrix of lattice 
    init_model!(model, J, l)

    # get reduced representation of lattice
    reduced, trafo_index = get_reduced(l)
    irreducible = unique(reduced[reduced .!= 0])
    sites       = Site[Site(l.sites[i].int, get_vec(l.sites[i].int, l.uc)) for i in irreducible]

    # get mapping table
    mappings = get_mappings(l, reduced, trafo_index, symmetry)

    # get overlap
    overlap = get_overlap(l, reduced, irreducible, mappings)

    # get multiplicities
    mult = get_mult(reduced, irreducible)

    # get exchanges
    exchange = get_exchange(irreducible, mappings)

    #get local map (mappings of local vertex to reference site)
    localmap = [mappings[i, i] for i in irreducible]

    # get projections
    project = get_project(l, irreducible, mappings)

    # build reduced lattice
    r = Reduced_lattice(l.name, l.size, model, J, sites, overlap, mult, exchange, localmap, project)

    if verbose
        println("Done. Reduced lattice has $(length(r.sites)) sites.")
    end

    return r
end