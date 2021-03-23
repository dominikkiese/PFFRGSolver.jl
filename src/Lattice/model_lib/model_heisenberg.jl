"""
    init_model_heisenberg!(
        J :: Vector{Float64},
        l :: lattice
        ) :: Nothing

Init Heisenberg model on a given lattice by overwriting the respective bonds.
Here J[n] is the coupling to the n-th nearest neighbor (Euclidean norm).
"""
function init_model_heisenberg!(
    J :: Vector{Float64},
    l :: lattice
    ) :: Nothing

    # iterate over sites and add Heisenberg matrices to lattice bonds
    for i in eachindex(l.sites)
        for n in eachindex(J)
            nbs = get_nbs(n, l.sites[i], l.sites)
            
            for j in nbs
                add_bond_heisenberg!(J[n], l.bonds[i, j])
            end
        end
    end

    return nothing
end