"""
    save!(
        file :: HDF5.File,
        l    :: lattice
        )    :: Nothing

Save lattice to HDF5 file.
"""
function save!(
    file :: HDF5.File,
    l    :: lattice
    )    :: Nothing

    # save name and size
    file["lattice/name"] = l.name 
    file["lattice/size"] = l.size 

    # save basis sites 
    for i in eachindex(l.uc.basis)
        file["lattice/unitcell/basis/$(i)"] = l.uc.basis[i]
    end 

    # save Braivais vectors 
    for i in eachindex(l.uc.vectors)
        file["lattice/unitcell/vectors/$(i)"] = l.uc.vectors[i]
    end 

    # save bonds for basis sites
    for i in eachindex(l.uc.bonds)
        for j in eachindex(l.uc.bonds[i])
            file["lattice/unitcell/bonds/$(i)/$(j)"] = l.uc.bonds[i][j]
        end 
    end

    # save test sites 
    for i in eachindex(l.test_sites)
        file["lattice/test_sites/$(i)/int"] = l.test_sites[i].int
        file["lattice/test_sites/$(i)/vec"] = l.test_sites[i].vec 
    end 

    # save sites 
    for i in eachindex(l.sites)
        file["lattice/sites/$(i)/int"] = l.sites[i].int
        file["lattice/sites/$(i)/vec"] = l.sites[i].vec 
    end

    # save interactions
    for i in 1 : size(l.bonds, 1)
        for j in 1 : size(l.bonds, 2)
            for k in eachindex(l.bonds[i, j].exchange) 
                file["lattice/bonds/$(i)/$(j)/exchange/$(k)"] = l.bonds[i, j].exchange[k]
            end 
        end 
    end

    return nothing 
end

"""
    read_lattice(
        file :: HDF5.File,
        )    :: lattice 

Read lattice from HDF5 file.
"""
function read_lattice(
    file :: HDF5.File,
    )    :: lattice 

    # read name and size
    name = read(file, "lattice/name")
    size = read(file, "lattice/size")

    # read basis sites 
    num_basis = length(keys(file["lattice/unitcell/basis"]))
    basis     = Vector{Float64}[read(file, "lattice/unitcell/basis/$(i)") for i in 1 : num_basis]

    # read Bravais vectors 
    num_vectors = length(keys(file["lattice/unitcell/vectors"]))
    vectors     = Vector{Float64}[read(file, "lattice/unitcell/vectors/$(i)") for i in 1 : num_vectors]

    # read bonds for basis sites 
    bonds_uc = Vector{Vector{Vector{Int64}}}(undef, num_basis)

    for i in 1 : num_basis 
        num_bonds   = length(keys(file["lattice/unitcell/bonds/$(i)"]))
        bonds_uc[i] = Vector{Int64}[read(file, "lattice/unitcell/bonds/$(i)/$(j)") for j in 1 : num_bonds]
    end

    # build unitcell 
    uc = unitcell(basis, vectors, bonds_uc)

    # read test sites 
    num_test_sites = length(keys(file["lattice/test_sites"]))
    test_sites     = site[site(read(file, "lattice/test_sites/$(i)/int"), read(file, "lattice/test_sites/$(i)/vec")) for i in 1 : num_test_sites]

    # read sites 
    num_sites = length(keys(file["lattice/sites"]))
    sites     = site[site(read(file, "lattice/sites/$(i)/int"), read(file, "lattice/sites/$(i)/vec")) for i in 1 : num_sites]

    # read interactions 
    bonds_lattice = Matrix{bond}(undef, num_sites, num_sites)

    for i in 1 : num_sites 
        for j in 1 : num_sites 
            if haskey(file, "lattice/bonds/$(i)/$(j)/exchange")
                num_exchange        = length(keys(file["lattice/bonds/$(i)/$(j)/exchange"]))
                exchange            = Matrix{Float64}[read(file, "lattice/bonds/$(i)/$(j)/exchange/$(k)") for k in 1 : num_exchange]
                bonds_lattice[i, j] = bond((i, j), exchange)
            else
                bonds_lattice[i, j] = get_bond_empty(i, j)
            end
        end 
    end

    # build lattice 
    l = lattice(name, size, uc, test_sites, sites, bonds_lattice)

    return l 
end

"""
    save!(
        file :: HDF5.File,
        r    :: reduced_lattice
        )    :: Nothing 

Save reduced lattice to HDF5 file.
"""
function save!(
    file :: HDF5.File,
    r    :: reduced_lattice
    )    :: Nothing 

    # save reduced sites
    for i in eachindex(r.sites)
        file["reduced_lattice/sites/$(i)/int"] = r.sites[i].int 
        file["reduced_lattice/sites/$(i)/vec"] = r.sites[i].vec 
    end
        
    # save overlap 
    for i in eachindex(r.overlap)
        file["reduced_lattice/overlap/$(i)"] = r.overlap[i] 
    end 

    # save multiplicities, exchange and projections
    file["reduced_lattice/mult"]     = r.mult 
    file["reduced_lattice/exchange"] = r.exchange 
    file["reduced_lattice/project"]  = r.project

    return nothing 
end

"""
    read_reduced_lattice(
        file :: HDF5.File,
        )    :: reduced_lattice 

Read reduced lattice from HDF5 file.
"""
function read_reduced_lattice(
    file :: HDF5.File,
    )    :: reduced_lattice 

    # read lattice sites
    num_sites = length(keys(file["reduced_lattice/sites"]))
    sites     = site[site(read(file, "reduced_lattice/sites/$(i)/int"), read(file, "reduced_lattice/sites/$(i)/vec")) for i in 1 : num_sites]

    # read overlaps
    num_overlaps = length(keys(file["reduced_lattice/overlap"]))
    overlap      = Matrix{Int64}[read(file, "reduced_lattice/overlap/$(i)") for i in 1 : num_overlaps]

    # read multiplicities, exchange and projections
    mult     = read(file["reduced_lattice/mult"])
    exchange = read(file["reduced_lattice/exchange"])
    project  = read(file["reduced_lattice/project"])

    # build reduced lattice 
    r = reduced_lattice(sites, overlap, mult, exchange, project)

    return r 
end