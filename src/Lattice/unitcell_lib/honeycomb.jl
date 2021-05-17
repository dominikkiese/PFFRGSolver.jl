function get_unitcell_honeycomb() :: Unitcell

    # define basis sites
    basis    = Vector{Vector{Float64}}(undef, 2)
    basis[1] = zeros(Float64, 3)
    basis[2] = Float64[1.0, 0.0, 0.0]

    # define Bravais vectors
    vectors    = Vector{Vector{Float64}}(undef, 3)
    vectors[1] = Float64[3.0 / 2.0, sqrt(3.0) / 2.0, 0.0]
    vectors[2] = Float64[3.0 / 2.0, -sqrt(3.0) / 2.0, 0.0]
    vectors[3] = zeros(Float64, 3)

    # define bonds for basis sites
    bonds     = Vector{Vector{Vector{Int64}}}(undef, length(basis))
    bonds1    = Vector{Vector{Int64}}(undef, 3)
    bonds1[1] = Int64[0, -1, 0, 1]
    bonds1[2] = Int64[-1, 0, 0, 1]
    bonds1[3] = Int64[0, 0, 0, 1]
    bonds[1]  = bonds1
    bonds2    = Vector{Vector{Int64}}(undef, 3)
    bonds2[1] = Int64[0, 1, 0, -1]
    bonds2[2] = Int64[1, 0, 0, -1]
    bonds2[3] = Int64[0, 0, 0, -1]
    bonds[2]  = bonds2

    # build unitcell
    uc = Unitcell(basis, vectors, bonds)

    return uc
end