function get_unitcell_fcc() :: unitcell

    # define list of basis sites
    basis    = Vector{Vector{Float64}}(undef, 1)
    basis[1] = zeros(Float64, 3)

    # define list of Bravais vectors
    vectors    = Vector{Vector{Float64}}(undef, 3)
    vectors[1] = Float64[0.0, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0)]
    vectors[2] = Float64[1.0 / sqrt(2.0), 0.0, 1.0 / sqrt(2.0)]
    vectors[3] = Float64[1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0]

    # define list of bonds for each basis site
    bonds      = Vector{Vector{Vector{Int64}}}(undef, length(basis))
    bonds1     = Vector{Vector{Int64}}(undef, 12)
    bonds1[1]  = Int64[1, 0, 0, 0]
    bonds1[2]  = Int64[-1, 0, 0, 0]
    bonds1[3]  = Int64[0, 1, -1, 0]
    bonds1[4]  = Int64[0, -1, 1, 0]
    bonds1[5]  = Int64[0, 1, 0, 0]
    bonds1[6]  = Int64[0, -1, 0, 0]
    bonds1[7]  = Int64[1, 0, -1, 0]
    bonds1[8]  = Int64[-1, 0, 1, 0]
    bonds1[9]  = Int64[0, 0, 1, 0]
    bonds1[10] = Int64[0, 0, -1, 0]
    bonds1[11] = Int64[1, -1, 0, 0]
    bonds1[12] = Int64[-1, 1, 0, 0]
    bonds[1]   = bonds1

    # build unitcell
    uc = unitcell(basis, vectors, bonds)

    return uc
end