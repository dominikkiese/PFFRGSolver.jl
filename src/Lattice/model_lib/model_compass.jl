"""
    init_model_square_compass!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init spin model with compass interactions on the square lattice.
Here, J[1] = [Jxx, Jzz] (only nearest neighbor couplings), with
* `Jxx` : nearest neighbor coupling between the xx spin components
* `Jzz` : nearest neighbor coupling between the zz spin components
"""
function init_model_square_compass!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "square" "Model requires square lattice."
    @assert length(J) == 1 "Model allows only nearest-neighbor couplings."
    @assert length(J[1]) == 2 "Model requires two nearest-neighbor couplings Jxx and Jzz."

    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        # find nearest neighbors
        nbs = get_nbs(1, l.sites[i], l.sites)

        # determine couplings 
        Jxx, Jzz = J[1][1], J[1][2]

        # iterate over neighbors and overwrite bond matrices
        for j in nbs
            int = l.sites[j].int .- l.sites[i].int

            if in(int, ([0, 1, 0, 0], [0, -1, 0, 0]))
                add_bond!(Jxx, l.bonds[i, j], 1, 1)
            else 
                add_bond!(Jzz, l.bonds[i, j], 3, 3)
            end
        end
    end

    return nothing
end

"""
    init_model_honeycomb_compass!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init spin model with compass interactions on the honeycomb lattice.
Here, J[1] = [Jxx, Jzz] (only nearest neighbor couplings), with
* `Jxx` : nearest neighbor coupling between the xx spin components
* `Jzz` : nearest neighbor coupling between the zz spin components
"""
function init_model_honeycomb_compass!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "honeycomb" "Model requires honeycomb lattice."
    @assert length(J) == 1 "Model allows only nearest-neighbor couplings."
    @assert length(J[1]) == 2 "Model requires two nearest-neighbor couplings Jxx and Jzz."

    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        # find nearest neighbors
        nbs = get_nbs(1, l.sites[i], l.sites)

        # determine couplings 
        Jxx, Jzz = J[1][1], J[1][2]

        # iterate over neighbors and overwrite bond matrices
        for j in nbs
            int = l.sites[j].int .- l.sites[i].int

            if in(int, ([0, 0, 0, 1], [0, 0, 0, -1]))
                add_bond!(Jxx, l.bonds[i, j], 1, 1)
            else 
                add_bond!(Jzz, l.bonds[i, j], 3, 3)
            end
        end
    end

    return nothing
end