"""
    init_model_tWSe2!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init effective spin model for twisted WSe2 on the triangular lattice.
Here, J = [[J], [Φ]], where J is the effective coupling between nearest-neighbors and Φ the inversion symmetry breaking phase.
"""
function init_model_tWSe2!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # determine bond independent couplings 
    J, Φ = J[1][1], J[2][1]
    Jxx  = J * cos(2.0 * Φ)

    # buffer bonds with negative phase 
    bonds_Φm = (Int64[0, 1, 0, 0], Int64[-1, 0, 0, 0], Int64[1, -1, 0, 0])

    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        # find nearest neighbor sites 
        nbs = get_nbs(1, l.sites[i], l.sites)

        for j in nbs 
            # determine bond dependent DM coupling
            D = J * sin(2.0 * Φ)

            if l.sites[j].int .- l.sites[i].int in bonds_Φm
                D *= -1.0
            end

            # add couplings to bonds
            add_bond!(Jxx, l.bonds[i, j], 1, 1)
            add_bond!(Jxx, l.bonds[i, j], 2, 2)
            add_bond!(  J, l.bonds[i, j], 3, 3)
            add_bond!( +D, l.bonds[i, j], 1, 2)
            add_bond!( -D, l.bonds[i, j], 2, 1)
        end 
    end

    return nothing
end
