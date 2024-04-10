"""
    init_model_pyrochlore_heisenberg_dipolar!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init dipolar Heisenberg model on the pyrochlore lattice by overwriting the respective bonds.
Here, J[1] = [J, D] (see PRB 62.488), with 
* `J` : nearest neighbor Heisenberg coupling
* `D` : dipolar Heisenberg coupling
"""
function init_model_pyrochlore_heisenberg_dipolar!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "pyrochlore" "Model requires pyrochlore lattice."
    @assert length(J) == 1 && length(J[1]) == 2 "Only nearest neighbor Heisenberg and dipolar couplings allowed."

    # iterate over sites and add Heisenberg couplings to lattice bonds
    for i in eachindex(l.sites)
        # find nearest neighbors
        nbs = get_nbs(1, l.sites[i], l.sites)
        rnn = norm(get_vec(l.sites[i].int, l.uc) - get_vec(l.sites[nbs[1]].int, l.uc))

        for j in nbs
            add_bond!(J[1][1], l.bonds[i, j], 1, 1)
            add_bond!(J[1][1], l.bonds[i, j], 2, 2)
            add_bond!(J[1][1], l.bonds[i, j], 3, 3)
        end

        # add dipolar couplings to lattice bonds
        for j in eachindex(l.sites)
            if i != j
                ratio = (rnn / norm(get_vec(l.sites[i].int, l.uc) - get_vec(l.sites[j].int, l.uc)))^3
                add_bond!(ratio * J[1][2], l.bonds[i, j], 1, 1)
                add_bond!(ratio * J[1][2], l.bonds[i, j], 2, 2)
                add_bond!(ratio * J[1][2], l.bonds[i, j], 3, 3)
            end
        end
    end

    return nothing
end