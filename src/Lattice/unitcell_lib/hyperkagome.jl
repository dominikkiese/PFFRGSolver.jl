function get_unitcell_hyperkagome() :: unitcell

    # define list of basis sites
    basis    = Vector{Vector{Float64}}(undef, 12)
    basis[1] = Float64[7/2/sqrt(2), 1/2/sqrt(2), 1/2/sqrt(2)]
    basis[2] = Float64[5/2/sqrt(2), 3/2/sqrt(2), 1/2/sqrt(2)]
    basis[3] = Float64[7/2/sqrt(2), 3/2/sqrt(2), 3/2/sqrt(2)]
    basis[4] = Float64[7/2/sqrt(2), 5/2/sqrt(2), 5/2/sqrt(2)]
    basis[5] = Float64[5/2/sqrt(2), 5/2/sqrt(2), 7/2/sqrt(2)]
    basis[6] = Float64[3/2/sqrt(2), 3/2/sqrt(2), 7/2/sqrt(2)]
    basis[7] = Float64[3/2/sqrt(2), 1/2/sqrt(2), 5/2/sqrt(2)]
    basis[8] = Float64[1/2/sqrt(2), 1/2/sqrt(2), 7/2/sqrt(2)]
    basis[9] = Float64[5/2/sqrt(2), 7/2/sqrt(2), 5/2/sqrt(2)]
    basis[10] = Float64[3/2/sqrt(2), 7/2/sqrt(2), 3/2/sqrt(2)]
    basis[11] = Float64[1/2/sqrt(2), 7/2/sqrt(2), 1/2/sqrt(2)]
    basis[12] = Float64[1/2/sqrt(2), 5/2/sqrt(2), 3/2/sqrt(2)]

    # define list of Bravais vectors
    vectors    = Vector{Vector{Float64}}(undef, 3)
    vectors[1] = Float64[2*sqrt(2), 0.0,  0.0]
    vectors[2] = Float64[0.0, 2*sqrt(2), 0.0]
    vectors[3] = Float64[0.0, 0.0, 2*sqrt(2)]

    # define list of bonds for each basis site
    bonds     = Vector{Vector{Vector{Int64}}}(undef, length(basis))
    bonds1    = Vector{Vector{Int64}}(undef, 4)
    bonds1[1] = [0, 0, 0, 1]
    bonds1[2] = [0, 0, 0, 2]
    bonds1[3] = [1, 0, -1, 7]
    bonds1[4] = [1, -1, 0, 10]
    bonds[1] = bonds1
    bonds2    = Vector{Vector{Int64}}(undef, 4)
    bonds2[1] = [0, 0, 0, -1]
    bonds2[2] = [0, 0, 0, 1]
    bonds2[3] = [0, 0, -1, 3]
    bonds2[4] = [0, 0, -1, 4]
    bonds[2] = bonds2
    bonds3    = Vector{Vector{Int64}}(undef, 4)
    bonds3[1] = [0, 0, 0, -1]
    bonds3[2] = [0, 0, 0, -2]
    bonds3[3] = [0, 0, 0, 1]
    bonds3[4] = [1, 0, 0, 9]
    bonds[3] = bonds3
    bonds4    = Vector{Vector{Int64}}(undef, 4)
    bonds4[1] = [0, 0, 0, -1]
    bonds4[2] = [0, 0, 0, 1]
    bonds4[3] = [0, 0, 0, 5]
    bonds4[4] = [1, 0, 0, 8]
    bonds[4] = bonds4
    bonds5    = Vector{Vector{Int64}}(undef, 4)
    bonds5[1] = [0, 0, 0, -1]
    bonds5[2] = [0, 0, 0, 4]
    bonds5[3] = [0, 0, 0, 1]
    bonds5[4] = [0, 0, 1, -3]
    bonds[5] = bonds5
    bonds6    = Vector{Vector{Int64}}(undef, 4)
    bonds6[1] = [0, 0, 0, -1]
    bonds6[2] = [0, 0, 0, 1]
    bonds6[3] = [0, 0, 0, 2]
    bonds6[4] = [0, 0, 1, -4]
    bonds[6] = bonds6
    bonds7    = Vector{Vector{Int64}}(undef, 4)
    bonds7[1] = [0, 0, 0, -1]
    bonds7[2] = [0, 0, 0, 1]
    bonds7[3] = [0, -1, 0, 2]
    bonds7[4] = [0, -1, 0, 3]
    bonds[7] = bonds7
    bonds8    = Vector{Vector{Int64}}(undef, 4)
    bonds8[1] = [0, 0, 0, -2]
    bonds8[2] = [0, 0, 0, -1]
    bonds8[3] = [-1, 0, 1, -7]
    bonds8[4] = [0, -1, 1, 3]
    bonds[8] = bonds8
    bonds9    = Vector{Vector{Int64}}(undef, 4)
    bonds9[1] = [0, 0, 0, -5]
    bonds9[2] = [0, 0, 0, -4]
    bonds9[3] = [0, 1, 0, -2]
    bonds9[4] = [0, 0, 0, 1]
    bonds[9] = bonds9
    bonds10    = Vector{Vector{Int64}}(undef, 4)
    bonds10[1] = [0, 1, 0, -3]
    bonds10[2] = [0, 0, 0, -1]
    bonds10[3] = [0, 0, 0, 1]
    bonds10[4] = [0, 0, 0, 2]
    bonds[10] = bonds10
    bonds11    = Vector{Vector{Int64}}(undef, 4)
    bonds11[1] = [0, 0, 0, -1]
    bonds11[2] = [0, 0, 0, 1]
    bonds11[3] = [-1, 1, 0, -10]
    bonds11[4] = [0, 1, -1, -3]
    bonds[11] = bonds11
    bonds12    = Vector{Vector{Int64}}(undef, 4)
    bonds12[1] = [0, 0, 0, -2]
    bonds12[2] = [0, 0, 0, -1]
    bonds12[3] = [-1, 0, 0, -9]
    bonds12[4] = [-1, 0, 0, -8]
    bonds[12] = bonds12

    # build unitcell
    uc = unitcell(basis, vectors, bonds)

    return uc
end
