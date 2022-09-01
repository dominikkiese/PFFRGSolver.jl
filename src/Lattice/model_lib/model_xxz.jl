"""
    init_model_xxz!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init xxz model on a given lattice by overwriting the respective bonds.
Here, J[n] is the coupling to the n-th nearest neighbor (Euclidean norm).
Symmetry inequivalent bonds are initialized uniformly
"""
function init_model_xxz!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # iterate over sites and add xxz couplings to lattice bonds
    for i in eachindex(l.sites)
       for n in eachindex(J)
            # find n-th nearest neighbors
            nbs = get_nbs(n, l.sites[i], l.sites)
            @assert length(J[n]) == 2 "J[$n] does not have two components (Jx, Jz)"
            # uniform initialization for n-th nearest neighbor, if no further couplings provided
            for j in nbs
                add_bond!(J[n][1], l.bonds[i, j], 1, 1)
                add_bond!(J[n][1], l.bonds[i, j], 2, 2)
                add_bond!(J[n][2], l.bonds[i, j], 3, 3)
            end
        end
    end

    return nothing
end