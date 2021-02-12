# init Heisenberg model on a given lattice, here J[n] is the coupling to the n-th nearest neighbor 
function init_model_heisenberg!(
    J :: Vector{Float64},
    l :: lattice
    ) :: Nothing

    # iterate over sites and add Heisenberg matrices to lattice bonds
    for i in eachindex(l.sites)
        for n in eachindex(J)
            nbs = get_nbs(n, l.sites[i], l.sites)
            
            for j in nbs
                push!(l.bonds[i, j].exchange, get_bond_heisenberg(J[n]))
            end
        end
    end

    return nothing
end