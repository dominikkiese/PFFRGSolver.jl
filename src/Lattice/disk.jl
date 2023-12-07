# save reduced lattice to HDF5 file (implicitly saves original lattice too)
function save!(
    file      :: HDF5.File,
    r         :: Reduced_lattice,
    euclidean :: Bool
    )         :: Nothing

    # save name, size, model, building metric and coupling vector
    file["reduced_lattice/name"]      = r.name 
    file["reduced_lattice/size"]      = r.size 
    file["reduced_lattice/model"]     = r.model 
    file["reduced_lattice/euclidean"] = euclidean

    for i in eachindex(r.J)
        file["reduced_lattice/J/$(i)"] = r.J[i]
    end

    return nothing 
end

"""
    read_lattice(
        file :: HDF5.File,
        )    :: Tuple{Lattice, Reduced_lattice}

Construct lattice and reduced lattice from HDF5 file.
"""
function read_lattice(
    file :: HDF5.File,
    )    :: Tuple{Lattice, Reduced_lattice}

    # read name, size, model, building metric and coupling vector
    name      = read(file, "reduced_lattice/name")
    size      = read(file, "reduced_lattice/size")
    model     = read(file, "reduced_lattice/model")
    euclidean = read(file, "reduced_lattice/euclidean")
    J         = Vector{Float64}[]

    for i in eachindex(keys(file["reduced_lattice/J"]))
        push!(J, read(file, "reduced_lattice/J/$(i)"))
    end

    # build lattice and reduced lattice
    l = get_lattice(name, size, euclidean = euclidean, verbose = false)
    r = get_reduced_lattice(model, J, l, verbose = false)

    return l, r
end