"""
    init_model_heisenberg!(
        J :: Vector{<:Any},
        l :: lattice
        ) :: Nothing

Init Heisenberg model on a given lattice by overwriting the respective bonds.
Here J[n] is the coupling to the n-th nearest neighbor (Euclidean norm).
If there are m symmetry inequivalent n-th nearest neighbors, these are
    - uniformly intiliazed if J[n] is a single value
    - initilized in ascending bond distance form the origin, if J[n] is an array
      of length m
"""
function init_model_heisenberg!(
    J :: Vector{<:Any},
    l :: lattice
    ) :: Nothing

    # iterate over sites and add Heisenberg matrices to lattice bonds
    for i in eachindex(l.sites)
        for n in eachindex(J)
            nbs = get_nbs(n, l.sites[i], l.sites)


            if(length(J[n]) == 1)
                #uniform initialization for nth nearest neighbor
                for j in nbs
                    push!(l.bonds[i, j].exchange, get_bond_heisenberg(J[n]))
                end
            else
            	#initialize symmetry non-equivalent bonds interactions in ascending bond-length order
                #get bond distances of neighbors
                dist = Int64[]

                for j in nbs
                    push!(dist,get_metric(l.sites[j], l.sites[i], l.uc))
                end

                #filter out classes of bond distances
                nbkinds = sort(unique(dist))

                @assert length(J[n]) == length(nbkinds) "$(l.name) has $(length(nbkinds)) $(n)th nearest neighbors, but $(length(J[n])) couplings were supplied. \nPlease provide right number or just one for uniform initialization."

            	for nk in eachindex(nbkinds)
            	    #filter out neighbours with dist == nbkinds[nk]
            	    nknbs = nbs[findall(x -> x == nbkinds[nk], dist)]
                    for j in nknbs
                        push!(l.bonds[i, j].exchange, get_bond_heisenberg(J[n][nk]))
                    end
                end
            end
        end
    end

    return nothing
end
