function get_unitcell_square() :: Unitcell

    # define list of basis sites
    basis    = Vector{SVector{3, Float64}}(undef, 1)
    basis[1] = SVector{3, Float64}(0.0, 0.0, 0.0)

    # define list of Bravais vectors
    vectors    = Vector{SVector{3, Float64}}(undef, 3)
    vectors[1] = SVector{3, Float64}(1.0, 0.0, 0.0)
    vectors[2] = SVector{3, Float64}(0.0, 1.0, 0.0)
    vectors[3] = SVector{3, Float64}(0.0, 0.0, 0.0)

    # define list of bonds for each basis site
    bonds     = Vector{Vector{SVector{4, Int64}}}(undef, length(basis))
    bonds1    = Vector{SVector{4, Int64}}(undef, 4)
    bonds1[1] = SVector{4, Int64}( 1,  0,  0,  0)
    bonds1[2] = SVector{4, Int64}(-1,  0,  0,  0)
    bonds1[3] = SVector{4, Int64}( 0,  1,  0,  0)
    bonds1[4] = SVector{4, Int64}( 0, -1,  0,  0)
    bonds[1]  = bonds1

    # build unitcell
    uc = Unitcell(basis, vectors, bonds)

    return uc
end