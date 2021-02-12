function get_unitcell_pyrochlore() :: unitcell

    # define list of basis sites
    basis    = Vector{Vector{Float64}}(undef, 4)
    basis[1] = zeros(Float64, 3)
    basis[2] = Float64[0.0, 0.25, 0.25]
    basis[3] = Float64[0.25, 0.0, 0.25]
    basis[4] = Float64[0.25, 0.25, 0.0]

    # define list of Bravais vectors
    vectors    = Vector{Vector{Float64}}(undef, 3)
    vectors[1] = Float64[0.0, 0.5, 0.5]
    vectors[2] = Float64[0.5, 0.0, 0.5]
    vectors[3] = Float64[0.5, 0.5, 0.0]

    # define list of bonds for each basis site
    bonds     = Vector{Vector{Vector{Int64}}}(undef, length(basis))
    bonds1    = Vector{Vector{Int64}}(undef, 6)
    bonds1[1] = Int64[0, 0, 0, 1]
    bonds1[2] = Int64[0, 0, 0, 2]
    bonds1[3] = Int64[0, 0, 0, 3]
    bonds1[4] = Int64[-1, 0, 0, 1]
    bonds1[5] = Int64[0, -1, 0, 2]
    bonds1[6] = Int64[0, 0, -1, 3]
    bonds[1]  = bonds1
    bonds2    = Vector{Vector{Int64}}(undef, 6)
    bonds2[1] = Int64[0, 0, 0, -1]
    bonds2[2] = Int64[0, 0, 0, 1]
    bonds2[3] = Int64[0, 0, 0, 2]
    bonds2[4] = Int64[1, 0, 0, -1]
    bonds2[5] = Int64[1, -1, 0, 1]
    bonds2[6] = Int64[1, 0, -1, 2]
    bonds[2]  = bonds2
    bonds3    = Vector{Vector{Int64}}(undef, 6)
    bonds3[1] = Int64[0, 0, 0, -2]
    bonds3[2] = Int64[0, 0, 0, -1]
    bonds3[3] = Int64[0, 0, 0, 1]
    bonds3[4] = Int64[0, 1, 0, -2]
    bonds3[5] = Int64[-1, 1, 0, -1]
    bonds3[6] = Int64[0, 1, -1, 1]
    bonds[3]  = bonds3
    bonds4    = Vector{Vector{Int64}}(undef, 6)
    bonds4[1] = Int64[0, 0, 0, -3]
    bonds4[2] = Int64[0, 0, 0, -2]
    bonds4[3] = Int64[0, 0, 0, -1]
    bonds4[4] = Int64[0, 0, 1, -3]
    bonds4[5] = Int64[-1, 0, 1, -2]
    bonds4[6] = Int64[0, -1, 1, -1]
    bonds[4]  = bonds4

    # build unitcell
    uc = unitcell(basis, vectors, bonds)

    return uc
end