"""
    init_model_honeycomb_kitaev!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init spin model with Kitaev interactions on the honeycomb lattice.
Here, J[1] = [Jxx, Jyy, Jzz] (only nearest neighbor couplings), with
* `Jxx` : nearest neighbor coupling between the xx spin components
* `Jyy` : nearest neighbor coupling between the yy spin components
* `Jzz` : nearest neighbor coupling between the zz spin components
"""
function init_model_honeycomb_kitaev!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "honeycomb" "Model requires honeycomb lattice."
    @assert length(J) == 1 "Model allows only nearest-neighbor couplings."
    @assert length(J[1]) == 3 "Model requires three nearest-neighbor couplings Jxx, Jyy and Jzz."

    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        # find nearest neighbors
        nbs = get_nbs(1, l.sites[i], l.sites)

        # determine couplings 
        Jxx, Jyy, Jzz = J[1][1], J[1][2], J[1][3]

        # iterate over neighbors and overwrite bond matrices
        for j in nbs
            int = l.sites[j].int .- l.sites[i].int

            if in(int, ([0, 0, 0, 1], [0, 0, 0, -1]))
                add_bond!(Jxx, l.bonds[i, j], 1, 1)
            elseif in(int, ([0, -1, 0, 1], [0, 1, 0, -1]))
                add_bond!(Jyy, l.bonds[i, j], 2, 2)
            else 
                add_bond!(Jzz, l.bonds[i, j], 3, 3)
            end
        end
    end

    return nothing
end