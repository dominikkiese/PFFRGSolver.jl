"""
    init_model_breathing!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init Heisenberg model on a breathing pyrochlore or kagome lattice by overwriting the respective bonds.
Here, J[n] is the coupling to the n-th nearest neighbor (Euclidean norm). J[1] has to be an array of
length 2, specifying the breathing, anisotropic nearest neighbor couplings.
If there are m symmetry inequivalent n-th nearest neighbors (n > 1), these are
* uniformly initialized if J[n] is a single value
* initialized in ascending bond distance from the origin, if J[n] is an array of length m
"""
function init_model_breathing!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    @assert l.name in ["kagome", "pyrochlore"] "Breathing model requires Pyrochlore or Kagome lattice."
    @assert length(J[1]) == 2 "Breathing model needs two nearest neighbor couplings."

    # iterate over sites and add Heisenberg couplings to lattice bonds
    for i in eachindex(l.sites)
        for n in eachindex(J)
            # find n-th nearest neighbors
            nbs = get_nbs(n, l.sites[i], l.sites)

            # treat nearest neighbors according to breathing
            if n == 1
                for j in nbs
                    # if bond is within unit cell, it belongs to first kind of breathing coupling
                    if (l.sites[j].int - l.sites[i].int)[1 : 3] == [0, 0, 0]
                        add_bond!(J[1][1], l.bonds[i, j], 1, 1)
                        add_bond!(J[1][1], l.bonds[i, j], 2, 2)
                        add_bond!(J[1][1], l.bonds[i, j], 3, 3)
                    # if it crosses unit cell boundary, initalizes with second kind
                    else
                        add_bond!(J[1][2], l.bonds[i, j], 1, 1)
                        add_bond!(J[1][2], l.bonds[i, j], 2, 2)
                        add_bond!(J[1][2], l.bonds[i, j], 3, 3)
                    end
                end
            # uniform initialization for n-th nearest neighbor, if no further couplings provided
            elseif length(J[n]) == 1
                for j in nbs
                    add_bond!(J[n][1], l.bonds[i, j], 1, 1)
                    add_bond!(J[n][1], l.bonds[i, j], 2, 2)
                    add_bond!(J[n][1], l.bonds[i, j], 3, 3)
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
                        add_bond!(J[n][nk], l.bonds[i, j], 1, 1)
                        add_bond!(J[n][nk], l.bonds[i, j], 2, 2)
                        add_bond!(J[n][nk], l.bonds[i, j], 3, 3)
                    end
                end
            end
        end
    end

    return nothing
end

"""
    init_model_pyrochlore_breathing_c3!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init Heisenberg model on a breathing pyrochlore lattice with broken C3 symmetry by overwriting the respective bonds.
Here, J[1] = [J1, J2, δ1, δ2] specifies the breathing nearest-neighbor couplings J1 (for up tetrahedra), 
J2 (for down tetrahedra) and δ1 (δ2), which quantifies the C3 breaking perturbation on up (down) tetrahedra.
J[n] (for n >= 2) specifies additional Heisenberg couplings, these are
* uniformly initialized if J[n] is a single value
* initialized in ascending bond distance from the origin, if J[n] is an array of length m
"""
function init_model_pyrochlore_breathing_c3!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    @assert l.name == "pyrochlore" "Model requires Pyrochlore lattice."
    @assert length(J[1]) == 4 "Model requires two nearest neighbor couplings and two C3 breaking perturbations."

    # iterate over sites and add Heisenberg couplings to lattice bonds
    for i in eachindex(l.sites)
        for n in eachindex(J)
            # find nearest neighbors
            nbs = get_nbs(n, l.sites[i], l.sites)

            # treat nearest neighbors according to breathing anisotropy
            if n == 1
                for j in nbs
                    # get basis indices 
                    idxs = (l.sites[j].int[4], l.sites[i].int[4])

                    # set coupling to J1 (± δ1) for up tetrahedra
                    if (l.sites[j].int - l.sites[i].int)[1 : 3] == [0, 0, 0]
                        # add δ on bonds connecting basis sites 1 and 2
                        if idxs == (1, 2) || idxs == (2, 1)
                            add_bond!(J[1][1] + J[1][3], l.bonds[i, j], 1, 1)
                            add_bond!(J[1][1] + J[1][3], l.bonds[i, j], 2, 2)
                            add_bond!(J[1][1] + J[1][3], l.bonds[i, j], 3, 3)
                        # add δ on bonds connecting basis sites 3 and 4
                        elseif idxs == (3, 4) || idxs == (4, 3)
                            add_bond!(J[1][1] + J[1][3], l.bonds[i, j], 1, 1)
                            add_bond!(J[1][1] + J[1][3], l.bonds[i, j], 2, 2)
                            add_bond!(J[1][1] + J[1][3], l.bonds[i, j], 3, 3)
                        # subtract δ for all other bonds
                        else 
                            add_bond!(J[1][1] - J[1][3], l.bonds[i, j], 1, 1)
                            add_bond!(J[1][1] - J[1][3], l.bonds[i, j], 2, 2)
                            add_bond!(J[1][1] - J[1][3], l.bonds[i, j], 3, 3)
                        end 
                    # set coupling to J2 (± δ2) for down tetrahedra
                    else
                        # add δ on bonds connecting basis sites 1 and 2
                        if idxs == (1, 2) || idxs == (2, 1)
                            add_bond!(J[1][2] + J[1][4], l.bonds[i, j], 1, 1)
                            add_bond!(J[1][2] + J[1][4], l.bonds[i, j], 2, 2)
                            add_bond!(J[1][2] + J[1][4], l.bonds[i, j], 3, 3)
                        # add δ on bonds connecting basis sites 3 and 4
                        elseif idxs == (3, 4) || idxs == (4, 3)
                            add_bond!(J[1][2] + J[1][4], l.bonds[i, j], 1, 1)
                            add_bond!(J[1][2] + J[1][4], l.bonds[i, j], 2, 2)
                            add_bond!(J[1][2] + J[1][4], l.bonds[i, j], 3, 3)
                        # ignore δ for all other bonds
                        else 
                            add_bond!(J[1][2] - J[1][4], l.bonds[i, j], 1, 1)
                            add_bond!(J[1][2] - J[1][4], l.bonds[i, j], 2, 2)
                            add_bond!(J[1][2] - J[1][4], l.bonds[i, j], 3, 3)
                        end 
                    end
                end
            # uniform initialization for n-th nearest neighbor, if no further couplings provided
            elseif length(J[n]) == 1
                for j in nbs
                    add_bond!(J[n][1], l.bonds[i, j], 1, 1)
                    add_bond!(J[n][1], l.bonds[i, j], 2, 2)
                    add_bond!(J[n][1], l.bonds[i, j], 3, 3)
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
                        add_bond!(J[n][nk], l.bonds[i, j], 1, 1)
                        add_bond!(J[n][nk], l.bonds[i, j], 2, 2)
                        add_bond!(J[n][nk], l.bonds[i, j], 3, 3)
                    end
                end
            end
        end
    end

    return nothing
end