function get_unitcell_hyperhoneycomb() :: Unitcell

    # define list of basis sites
    basis    = Vector{SVector{3, Float64}}(undef, 4)
    basis[1] = SVector{3, Float64}(0.0,  0.0, 0.0)
    basis[2] = SVector{3, Float64}(1.0,  1.0, 0.0)
    basis[3] = SVector{3, Float64}(1.0,  2.0, 1.0)
    basis[4] = SVector{3, Float64}(0.0, -1.0, 1.0)

    # define list of Bravais vectors
    vectors    = Vector{SVector{3, Float64}}(undef, 3)
    vectors[1] = SVector{3, Float64}(-1.0, 1.0, -2.0)
    vectors[2] = SVector{3, Float64}(-1.0, 1.0,  2.0)
    vectors[3] = SVector{3, Float64}( 2.0, 4.0,  0.0)

    # define list of bonds for each basis site
    bonds     = Vector{Vector{SVector{4, Int64}}}(undef, length(basis))
    bonds1    = Vector{SVector{4, Int64}}(undef, 3)
    bonds1[1] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds1[2] = SVector{4, Int64}( 0,  0,  0,  3)
    bonds1[3] = SVector{4, Int64}( 1,  0,  0,  3)
    bonds[1]  = bonds1
    bonds2    = Vector{SVector{4, Int64}}(undef, 3)
    bonds2[1] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds2[2] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds2[3] = SVector{4, Int64}( 0, -1,  0,  1)
    bonds[2]  = bonds2
    bonds3    = Vector{SVector{4, Int64}}(undef, 3)
    bonds3[1] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds3[2] = SVector{4, Int64}( 0,  1,  0, -1)
    bonds3[3] = SVector{4, Int64}( 0,  0,  1,  1)
    bonds[3]  = bonds3
    bonds4    = Vector{SVector{4, Int64}}(undef, 3)
    bonds4[1] = SVector{4, Int64}( 0,  0,  0, -3)
    bonds4[2] = SVector{4, Int64}(-1,  0,  0, -3)
    bonds4[3] = SVector{4, Int64}( 0,  0, -1, -1)
    bonds[4]  = bonds4

    # build unitcell
    uc = Unitcell(basis, vectors, bonds)

    return uc
end