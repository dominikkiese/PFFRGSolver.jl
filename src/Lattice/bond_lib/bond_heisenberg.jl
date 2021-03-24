function add_bond_heisenberg!(
    J :: Float64,
    b :: bond
    ) :: Nothing

    b.exchange[1, 1] += J
    b.exchange[2, 2] += J
    b.exchange[3, 3] += J

    return nothing
end