function get_unitcell_kagome() :: Unitcell

    # define list of basis sites
    basis    = Vector{Vector{Float64}}(undef, 3)
    basis[1] = zeros(Float64, 3)
    basis[2] = Float64[1.0, 0.0, 0.0]
    basis[3] = Float64[0.5, 0.5 * sqrt(3.0), 0.0]

    # define list of Bravais vectors
    vectors    = Vector{Vector{Float64}}(undef, 3)
    vectors[1] = Float64[2.0, 0.0, 0.0]
    vectors[2] = Float64[1.0, sqrt(3.0), 0.0]
    vectors[3] = zeros(Float64, 3)

    # define list of bonds for each basis site
    bonds     = Vector{Vector{Vector{Int64}}}(undef, length(basis))
    bonds1    = Vector{Vector{Int64}}(undef, 4)
    bonds1[1] = Int64[0, 0, 0, 1]
    bonds1[2] = Int64[0, 0, 0, 2]
    bonds1[3] = Int64[-1, 0, 0, 1]
    bonds1[4] = Int64[0, -1, 0, 2]
    bonds[1]  = bonds1
    bonds2    = Vector{Vector{Int64}}(undef, 4)
    bonds2[1] = Int64[0, 0, 0, -1]
    bonds2[2] = Int64[1, 0, 0, -1]
    bonds2[3] = Int64[0, 0, 0, 1]
    bonds2[4] = Int64[1, -1, 0, 1]
    bonds[2]  = bonds2
    bonds3    = Vector{Vector{Int64}}(undef, 4)
    bonds3[1] = Int64[0, 0, 0, -2]
    bonds3[2] = Int64[0, 0, 0, -1]
    bonds3[3] = Int64[0, 1, 0, -2]
    bonds3[4] = Int64[-1, 1, 0, -1]
    bonds[3]  = bonds3

    # build unitcell
    uc = Unitcell(basis, vectors, bonds)

    return uc
end