"""
    init_model_mapleleaf_j1j2j3!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing
"""
function init_model_mapleleaf_j1j2j3!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "mapleleaf" "Model requires mapleleaf lattice."
    @assert length(J) == 3 "Model initialization only works for third-nearest neighbors."

    for n in eachindex(J)
        @assert length(J[n]) == 1 "Only Heisenberg couplings implemented"
    end

    J1, J2, J3 = J[1][1], J[2][1], J[3][1]

    # increase test set according to J 
    metric  = 5 #maximum(Int64[get_metric(l.sites[1], l.sites[i], l.uc) for i in max_nbs]) 
    grow_test_sites!(l, metric)

    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        #nearest neighbor bonds
        nbs1 = get_nbs(1, l.sites[i], l.sites)
        
        for j in nbs1
            add_bond!(J1, l.bonds[i, j], 1, 1)
            add_bond!(J1, l.bonds[i, j], 2, 2)
            add_bond!(J1, l.bonds[i, j], 3, 3)
        end

        #next-nearest neighbor bonds inside hexagons
        nbs2 = get_nbs(2, l.sites[i], l.sites)
        nn2pairs = [(1, 3), (1, 5),
                    (2, 4), (2, 6),
                    (3, 1), (3, 5),
                    (4, 2), (4, 6),
                    (5, 1), (5, 3),
                    (6, 2), (6, 4)]
        for j in nbs2
            ints = (l.sites[j].int[4], l.sites[i].int[4])
            if ints in nn2pairs
                add_bond!(J2, l.bonds[i, j], 1, 1)
                add_bond!(J2, l.bonds[i, j], 2, 2)
                add_bond!(J2, l.bonds[i, j], 3, 3)
            end
        end

        #next-nearest neighbor bonds inside hexagons
        nbs3 = get_nbs(3, l.sites[i], l.sites)
        nn3pairs = [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)]
        for j in nbs3
            ints = (l.sites[j].int[4], l.sites[i].int[4])
            if ints in nn3pairs
                add_bond!(J3, l.bonds[i, j], 1, 1)
                add_bond!(J3, l.bonds[i, j], 2, 2)
                add_bond!(J3, l.bonds[i, j], 3, 3)
            end
        end
    end
    
    return nothing
end