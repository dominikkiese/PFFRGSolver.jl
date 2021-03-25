"""
    init_model_heisenberg!(
        J :: Vector{Vector{Float64}},
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
    J :: Vector{Vector{Float64}},
    l :: lattice
    ) :: Nothing

    # iterate over sites and add Heisenberg couplings to lattice bonds
    for i in eachindex(l.sites)
       for n in eachindex(J)
           # find n-th nearest neighbors
           nbs = get_nbs(n, l.sites[i], l.sites)

           # uniform initialization for n-th nearest neighbor, if no further couplings provided
           if length(J[n]) == 1
               for j in nbs
                   add_bond_heisenberg!(J[n][1], l.bonds[i, j])
               end
           # initialize symmetry non-equivalent bonds interactions in ascending bond-length order
           else
               # get bond distances of neighbors
               dist = Int64[get_metric(l.sites[j], l.sites[i], l.uc) for j in nbs]

               # filter out classes of bond distances
               nbkinds = sort(unique(dist))

               # sanity check
               @assert length(J[n]) == length(nbkinds) "$(l.name) has $(length(nbkinds)) inequivalent $(n)-th nearest neighbors, but $(length(J[n])) couplings were supplied."

               for nk in eachindex(nbkinds)
                   # filter out neighbors with dist == nbkinds[nk]
                   nknbs = nbs[findall(x -> x == nbkinds[nk], dist)]

                   for j in nknbs
                       add_bond_heisenberg!(J[n][nk], l.bonds[i, j])
                   end
               end
           end
       end
    end

    return nothing
end
