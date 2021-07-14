function get_unitcell_honeycomb() :: Unitcell

    # define basis sites
    basis    = Vector{SVector{3, Float64}}(undef, 2)
    basis[1] = SVector{3, Float64}(0.0, 0.0, 0.0)
    basis[2] = SVector{3, Float64}(1.0, 0.0, 0.0)

    # define Bravais vectors
    vectors    = Vector{SVector{3, Float64}}(undef, 3)
    vectors[1] = SVector{3, Float64}(3.0 / 2.0,  sqrt(3.0) / 2.0, 0.0)
    vectors[2] = SVector{3, Float64}(3.0 / 2.0, -sqrt(3.0) / 2.0, 0.0)
    vectors[3] = SVector{3, Float64}(      0.0,              0.0, 0.0)

    # define bonds for basis sites
    bonds     = Vector{Vector{SVector{4, Int64}}}(undef, length(basis))
    bonds1    = Vector{SVector{4, Int64}}(undef, 3)
    bonds1[1] = SVector{4, Int64}( 0, -1,  0,  1)
    bonds1[2] = SVector{4, Int64}(-1,  0,  0,  1)
    bonds1[3] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds[1]  = bonds1
    bonds2    = Vector{SVector{4, Int64}}(undef, 3)
    bonds2[1] = SVector{4, Int64}( 0,  1,  0, -1)
    bonds2[2] = SVector{4, Int64}( 1,  0,  0, -1)
    bonds2[3] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds[2]  = bonds2

    # build unitcell
    uc = Unitcell(basis, vectors, bonds)

    return uc
end