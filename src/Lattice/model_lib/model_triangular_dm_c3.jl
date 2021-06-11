"""
    init_model_triangular_dm_c3!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init spin model with out-of-plane Dzyaloshinskii-Moriya (DM) interactions on the triangular lattice. 
The DM interaction is chosen such that it switches sign on every second nearest-neighbor bond, thus breaking
the sixfold rotation symmetry of the triangular lattice down to threefold.
Here, J = [[Jxx], [Jzz], [Jxy]], with
* `Jxx` : coupling between the xx and yy spin components
* `Jzz` : coupling between the zz spin components
* `Jxy` : coupling between the xy spin components (aka DM interaction)
"""
function init_model_triangular_dm_c3!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity check 
    @assert l.name == "triangular" "Model requires triangular lattice."

    # determine couplings 
    Jxx, Jzz, Jxy = J[1][1], J[2][1], J[3][1]

    # buffer bonds with negative phase 
    bonds_Φm = (Int64[0, 1, 0, 0], Int64[-1, 0, 0, 0], Int64[1, -1, 0, 0])

    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        # find nearest neighbor sites 
        nbs = get_nbs(1, l.sites[i], l.sites)

        for j in nbs 
            # disentanle bond dependence of DM coupling
            if l.sites[j].int .- l.sites[i].int in bonds_Φm
                add_bond!(-Jxy, l.bonds[i, j], 1, 2)
                add_bond!(+Jxy, l.bonds[i, j], 2, 1)
            else
                add_bond!(+Jxy, l.bonds[i, j], 1, 2)
                add_bond!(-Jxy, l.bonds[i, j], 2, 1)
            end

            # add other couplings to bond
            add_bond!(Jxx, l.bonds[i, j], 1, 1)
            add_bond!(Jxx, l.bonds[i, j], 2, 2)
            add_bond!(Jzz, l.bonds[i, j], 3, 3)
        end 
    end

    return nothing
end
