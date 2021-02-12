function get_bond_heisenberg(
    J :: Float64
    ) :: Matrix{Float64}

    mat = diagm(Float64[J, J, J])

    return mat
end