"""
    init_model_j1_j2_j3a_pyrochlore!(
        J :: Vector{Float64},
        l :: lattice
        ) :: Nothing

Init J1-J2-J3a model on the pyrochlore lattice by overwriting the respective bonds, where J = [J1, J2, J3a]. 
Here J1 and J2 are the nearest and next-nearest neighbor couplings (Euclidean norm) and J3a the next-nearest neighbor coupling along a tetraeder edge.
"""
function init_model_j1_j2_j3a_pyrochlore!(
    J :: Vector{Float64},
    l :: lattice
    ) :: Nothing

    @assert l.name == "pyrochlore" "Model requires pyrochlore lattice."
    @assert length(J) == 3 "Model requires J = [J1, J2, J3a]."

    # iterate over sites and add Heisenberg matrices to lattice bonds
    for i in eachindex(l.sites)
        # set coupling with nearest-neighbors to J1
        nbs1 = get_nbs(1, l.sites[i], l.sites)

        for j in nbs1
            push!(l.bonds[i, j].exchange, get_bond_heisenberg(J[1]))
        end 

        # set coupling with second nearest-neighbors to J2
        nbs2 = get_nbs(2, l.sites[i], l.sites)

        for j in nbs2
            push!(l.bonds[i, j].exchange, get_bond_heisenberg(J[2]))
        end 

        # set coupling with third nearest-neighbors along tetraeder edges to J3a
        nbs3 = get_nbs(3, l.sites[i], l.sites)

        for j in nbs3
            # compute bond distance
            dist = get_metric(l.sites[j], l.sites[i], l.uc)

            # set coupling to J3a only if bond metric is 2 (i.e. if the sites aline on tetraeder edge)
            if dist == 2
                push!(l.bonds[i, j].exchange, get_bond_heisenberg(J[3]))
            end
        end 
    end

    return nothing
end