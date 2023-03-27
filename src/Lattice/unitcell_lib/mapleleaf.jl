function get_unitcell_mapleleaf() :: Unitcell

    # define list of basis sites
    basis    = Vector{SVector{3, Float64}}(undef, 6)
    basis[1] = SVector{3, Float64}(0.0,             0.0, 0.0)
    basis[2] = SVector{3, Float64}(0.5*sqrt(3),    -0.5, 0.0)
    basis[3] = SVector{3, Float64}(0.5*sqrt(3),     0.5, 0.0)
    basis[4] = SVector{3, Float64}(sqrt(3),         0.0, 0.0)
    basis[5] = SVector{3, Float64}(sqrt(3),         1.0, 0.0)
    basis[6] = SVector{3, Float64}(1.5*sqrt(3),     0.5, 0.0)

    # define list of Bravais vectors
    vectors    = Vector{SVector{3, Float64}}(undef, 3)
    vectors[1] = SVector{3, Float64}(1.5 * sqrt(3), -0.5, 0.0)
    vectors[2] = SVector{3, Float64}(sqrt(3),        2.0, 0.0)
    vectors[3] = SVector{3, Float64}(0.0,            0.0, 0.0)

    a1 = [1.5*sqrt(3), -0.5]
    a2 = [sqrt(3), 2.0]


    # define list of bonds for each basis site
    bonds     = Vector{Vector{SVector{4, Int64}}}(undef, length(basis))
    bonds1    = Vector{SVector{4, Int64}}(undef, 5)
    bonds1[1] = SVector{4, Int64}(-1,  0,  0,  3)
    bonds1[2] = SVector{4, Int64}(-1,  0,  0,  5)
    bonds1[3] = SVector{4, Int64}( 0,  0,  0,  2)
    bonds1[4] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds1[5] = SVector{4, Int64}( 0, -1,  0,  4)
    bonds[1]  = bonds1
    bonds2    = Vector{SVector{4, Int64}}(undef, 5)
    bonds2[1] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds2[2] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds2[3] = SVector{4, Int64}( 0,  0,  0,  2)
    bonds2[4] = SVector{4, Int64}( 0, -1,  0,  4)
    bonds2[5] = SVector{4, Int64}( 0, -1,  0,  3)
    bonds[2]  = bonds2
    bonds3    = Vector{SVector{4, Int64}}(undef, 5)
    bonds3[1] = SVector{4, Int64}(-1,  0,  0,  3)
    bonds3[2] = SVector{4, Int64}( 0,  0,  0,  2)
    bonds3[3] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds3[4] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds3[5] = SVector{4, Int64}( 0,  0,  0, -2)
    bonds[3]  = bonds3
    bonds4    = Vector{SVector{4, Int64}}(undef, 5)
    bonds4[1] = SVector{4, Int64}( 0,  0,  0, -2)
    bonds4[2] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds4[3] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds4[4] = SVector{4, Int64}( 0,  0,  0,  2)
    bonds4[5] = SVector{4, Int64}( 1,  0,  0, -3)
    bonds[4]  = bonds4
    bonds5    = Vector{SVector{4, Int64}}(undef, 5)
    bonds5[1] = SVector{4, Int64}( 0,  1,  0, -4)
    bonds5[2] = SVector{4, Int64}( 0,  1,  0, -3)
    bonds5[3] = SVector{4, Int64}( 0,  0,  0,  1)
    bonds5[4] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds5[5] = SVector{4, Int64}( 0,  0,  0, -2)
    bonds[5]  = bonds5
    bonds6    = Vector{SVector{4, Int64}}(undef, 5)
    bonds6[1] = SVector{4, Int64}( 0,  0,  0, -1)
    bonds6[2] = SVector{4, Int64}( 0,  1,  0, -4)
    bonds6[3] = SVector{4, Int64}( 1,  0,  0, -3)
    bonds6[4] = SVector{4, Int64}( 1,  0,  0, -5)
    bonds6[5] = SVector{4, Int64}( 0,  0,  0, -2)
    bonds[6]  = bonds6

    # build unitcell
    uc = Unitcell(basis, vectors, bonds)

    return uc
end