""" 
init_model_pyrochlore_local!(
    J :: Vector{Float64},
    l :: Lattice
    ) :: Nothing

Init Heisenberg Kitaev Gamma Model on the Pyrochlore lattice in the local basis.
Following Shannon et al. (10.1103/PhysRevB.95.094422) the couplings J1, J2 and J3 define the coupling matrix on the 0-1 bond in the GLOBAL basis as
J_01 =  [J2 J4 J4
        -J4 J1 J3
        -J4 J3 J1]
All other bonds in the global basis can be obtained via the corresponding lattice rotation.
To make the environment on all sites equivalent, this model transforms these into the LOCAL basis, which is also defined in Shannon et al. 
The correlations are, therefore, also in the local basis. The model here is only defined for nearest neighbors.
"""


function init_model_pyrochlore_local!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "pyrochlore" "Model requires pyrochlore lattice."
    @assert length(J) == 1 "only nearest neighbors are regarded"
    @assert length(J[1]) == 3 "each interaction has to be specified." 

    # iterate over sites and add respective couplings to lattice bonds
    for i in eachindex(l.sites)

        # get nearest neighbor sites 
        nbs = get_nbs(1, l.sites[i], l.sites)

        # determine couplings 
        J1, J2, J3 = J[1][1], J[1][2], J[1][3]

        for j in nbs 

            # get basis indices
            idxs = (l.sites[j].int[4], l.sites[i].int[4])

            # add Kitaev x-interaction to the respective bonds + the respective Gamma and Heisenberg interactions
            if idxs == (1, 2) || idxs == (2, 1) || idxs == (3, 4) || idxs == (4, 3)
                #diagonal 
                add_bond!(1/3 * (-J1 + 2 * J2 - J3), l.bonds[i, j], 1, 1)
                add_bond!(-J1 + J3, l.bonds[i, j], 2, 2)
                add_bond!(1/3 * (-2*J1 + J2 - 2 * J3), l.bonds[i, j], 3, 3)
                
                #offdiagonal
                add_bond!(-(1/3) * sqrt(2) * (J1 + J2 + J3), l.bonds[i, j], 1, 3)
                add_bond!(-(1/3) * sqrt(2) * (J1 + J2 + J3), l.bonds[i, j], 3, 1)

            end

            # add Kitaev y-interaction to the respective bonds + the respective Gamma and Heisenberg interactions
            if idxs == (1, 3) || idxs == (3, 1) || idxs == (2, 4) || idxs == (4, 2)
                #diagonal
                add_bond!(1/6 * (-5 * J1 + J2 + 4 * J3), l.bonds[i, j], 1, 1)
                add_bond!(1/2 * (-J1 + J2), l.bonds[i, j], 2, 2)
                add_bond!(1/3 * (-2 *J1 + J2 - 2 * J3), l.bonds[i, j], 3, 3)
                
                #offdiagonal
                add_bond!(-(J1 + J2 - 2 * J3)/(2 * sqrt(3)), l.bonds[i, j], 1, 2)
                add_bond!(-(J1 + J2 - 2 * J3)/(2 * sqrt(3)), l.bonds[i, j], 2, 1)
                
                add_bond!((J1 + J2 + J3)/(3 * sqrt(2)), l.bonds[i, j], 1, 3)
                add_bond!((J1 + J2 + J3)/(3 * sqrt(2)), l.bonds[i, j], 3, 1)
                
                add_bond!(-(J1 + J2 + J3)/sqrt(6), l.bonds[i, j], 2, 3)
                add_bond!(-(J1 + J2 + J3)/sqrt(6), l.bonds[i, j], 3, 2)

            end

            # add Kitaev z-interaction to the respective bonds + the respective Gamma and Heisenberg interactions
            if idxs == (1, 4) || idxs == (4, 1) || idxs == (2, 3) || idxs == (3, 2)
                #diagonal
                add_bond!(1/6 * (-5 * J1 + J2 + 4 * J3), l.bonds[i, j], 1, 1)
                add_bond!(1/2 * (-J1 + J2), l.bonds[i, j], 2, 2)
                add_bond!(1/3 * (-2 *J1 + J2 - 2 * J3), l.bonds[i, j], 3, 3)
                
                #offdiagonal
                add_bond!((J1 + J2 - 2 * J3)/(2 * sqrt(3)), l.bonds[i, j], 1, 2)
                add_bond!((J1 + J2 - 2 * J3)/(2 * sqrt(3)), l.bonds[i, j], 2, 1) 

                add_bond!((J1 + J2 + J3)/(3 * sqrt(2)), l.bonds[i, j], 1, 3)
                add_bond!((J1 + J2 + J3)/(3 * sqrt(2)), l.bonds[i, j], 3, 1)

                add_bond!( (J1 + J2 + J3)/sqrt(6), l.bonds[i, j], 2, 3)               
                add_bond!(((J1 + J2 + J3)/sqrt(6)), l.bonds[i, j], 3, 2)

            end 
        end 
    end

    return nothing 
end 