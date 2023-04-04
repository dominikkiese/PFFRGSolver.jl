"""
    init_model_mapleleaf_nn!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing
"""
function init_model_mapleleaf_nn!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "mapleleaf" "Model requires mapleleaf lattice."
    @assert length(J) == 1 "Model initialization only works for nearest neighbors."
    @assert length(J[1]) == 3 "Need J1, J2, J3 (kl, km and lm nearest-neighbor bonds)"
    
    J1, J2, J3 = J[1]

    # increase test set according to J 
    metric  = 5 #maximum(Int64[get_metric(l.sites[1], l.sites[i], l.uc) for i in max_nbs]) 
    grow_test_sites!(l, metric)


    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        #nearest neighbor bonds
        nbs1 = get_nbs(1, l.sites[i], l.sites)
        for j in nbs1
            ints = (l.sites[j].int[4], l.sites[i].int[4])
            
            #kl-coupling
            if abs.(ints[1]-ints[2]) in [1, 5]
                add_bond!(J1, l.bonds[i, j], 1, 1)
                add_bond!(J1, l.bonds[i, j], 2, 2)
                add_bond!(J1, l.bonds[i, j], 3, 3)
            #km-coupling
            elseif ints in [(1, 5), (5, 1), (3, 5), (5, 3), (1, 3), (3, 1),
                            (2, 6), (6, 2), (2, 4), (4, 2), (4, 6), (6, 4)]
                add_bond!(J2, l.bonds[i, j], 1, 1)
                add_bond!(J2, l.bonds[i, j], 2, 2)
                add_bond!(J2, l.bonds[i, j], 3, 3)
            #lm coupling (dimer)
            else
                add_bond!(J3, l.bonds[i, j], 1, 1)
                add_bond!(J3, l.bonds[i, j], 2, 2)
                add_bond!(J3, l.bonds[i, j], 3, 3)
            end
        end
    end    
    return nothing
end