""" 
init_model_hkg!(
    J :: Vector{Float64},
    l :: Lattice
    ) :: Nothing

Init Heisenberg Kitaev Gamma Model on the Pyrochlore lattice. Bonds are directionaly occupied considering also off-diagonal interactions.
Here J = [H, Jxx, Jyy, Jzz, Γxy, Γxz, Γyz]. 
* `H`   : Heisenberg interactions
* `Jxx` : Kitaev x-Bond diagonal 
* `Jyy` : Kitaev y-Bond diagonal
* `Jzz` : Kitaev z-Bond diagonal
* `Γyz` : Gamma x-Bond offdiagonal
* `Γxz` : Gamma y-Bond offdiagonal
* `Γxy` : Gamma z-Bond offdiagonal

Note: offdiagonals are symmetric i.e: Γyz = Γzy, Γxz = Γzx, Γxy = Γyx 
The model here is only defined for nearest neighbors.
"""


function init_model_pyrochlore_hkg!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "pyrochlore" "Model requires pyrochlore lattice."
    @assert length(J) == 1 "only nearest neighbors are regarded"
    @assert length(J[1]) == 7 "each interaction has to be specified." 

    # iterate over sites and add respective couplings to lattice bonds
    for i in eachindex(l.sites)

        # get nearest neighbor sites 
        nbs = get_nbs(1, l.sites[i], l.sites)

        # determine couplings 
        H, Jxx, Jyy, Jzz, Γxy, Γxz, Γyz = J[1][1], J[1][2], J[1][3], J[1][4], J[1][5], J[1][6], J[1][7]

        for j in nbs 

            # get basis indices
            idxs = (l.sites[j].int[4], l.sites[i].int[4])

            # add Kitaev x-interaction to the respective bonds + the respective Gamma and Heisenberg interactions
            if idxs == (1, 2) || idxs == (2, 1) || idxs == (3, 4) || idxs == (4, 3)

                add_bond!(Jxx, l.bonds[i, j], 1, 1)
                add_bond!(H, l.bonds[i, j], 1, 1)
                add_bond!(H, l.bonds[i, j], 2, 2)
                add_bond!(H, l.bonds[i, j], 3, 3)
                add_bond!(Γyz, l.bonds[i, j], 2, 3)
                add_bond!(Γyz, l.bonds[i, j], 3, 2)

            end

            # add Kitaev y-interaction to the respective bonds + the respective Gamma and Heisenberg interactions
            if idxs == (1, 3) || idxs == (3, 1) || idxs == (2, 4) || idxs == (4, 2)

                add_bond!(Jyy, l.bonds[i, j], 2, 2)
                add_bond!(H, l.bonds[i, j], 1, 1)
                add_bond!(H, l.bonds[i, j], 2, 2)
                add_bond!(H, l.bonds[i, j], 3, 3)
                add_bond!(Γxz, l.bonds[i, j], 1, 3)
                add_bond!(Γxz, l.bonds[i, j], 3, 1)

            end

            # add Kitaev z-interaction to the respective bonds + the respective Gamma and Heisenberg interactions
            if idxs == (1, 4) || idxs == (4, 1) || idxs == (2, 3) || idxs == (3, 2)

                add_bond!(H, l.bonds[i, j], 1, 1)
                add_bond!(H, l.bonds[i, j], 2, 2)
                add_bond!(H, l.bonds[i, j], 3, 3)
                add_bond!(Jzz, l.bonds[i, j], 3, 3)
                add_bond!(Γxy, l.bonds[i, j], 1, 2)
                add_bond!(Γxy, l.bonds[i, j], 2, 1)

            end 
        end 
    end

    return nothing 
end 