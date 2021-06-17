"""
    init_model_triangular_dm_c3!(
        J :: Vector{Vector{Float64}},
        l :: Lattice
        ) :: Nothing

Init spin model with out-of-plane Dzyaloshinskii-Moriya (DM) interactions on the triangular lattice. 
The DM interaction is chosen such that it switches sign between associated neighboring bonds, thus breaking the sixfold rotation symmetry of the triangular lattice down to threefold.
Here, J[n] = [Jxx, Jzz, Jxy] (n <= 3), with
* `Jxx` : n-th nearest neighbor coupling between the xx and yy spin components
* `Jzz` : n-th nearest neighbor coupling between the zz spin components
* `Jxy` : n-th nearest neighbor coupling between the xy spin components (aka DM interaction)
"""
function init_model_triangular_dm_c3!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "triangular" "Model requires triangular lattice."
    @assert length(J) <= 3 "Model initialization only works up to third-nearest neighbors."

    for n in eachindex(J)
        @assert length(J[n]) == 3 "Jxx, Jzz and Jxy all need to be specified for J[$(n)]."
    end

    # increase test set according to J 
    max_nbs = get_nbs(length(J), l.sites[1], l.sites)
    metric  = maximum(Int64[get_metric(l.sites[1], l.sites[i], l.uc) for i in max_nbs])
    grow_test_sites!(l, metric)

    # save reference vectors to ensure uniform initialization 
    ref_vecs = Vector{Float64}[]

    # iterate over sites and add couplings to lattice bonds
    for i in eachindex(l.sites)
        for n in eachindex(J)
            # find n-th nearest neighbors
            nbs = get_nbs(n, l.sites[i], l.sites)

            # determine couplings 
            Jxx, Jzz, Jxy = J[n][1], J[n][2], J[n][3]

            # determine (and save) reference vector 
            if length(ref_vecs) < n
                push!(ref_vecs, l.sites[nbs[1]].vec .- l.sites[i].vec)
            end

            ref_vec = ref_vecs[n]

            # iterate over neighbors and overwrite bond matrices
            for j in nbs
                vec = l.sites[j].vec .- l.sites[i].vec
                p   = round(dot(ref_vec, vec) / (norm(ref_vec) * norm(vec)), digits = 8)
                
                # disentanle bond dependence of DM coupling, preserving C3 but not C6 symmetry
                if acos(p) % (2.0 * pi / 3.0) < 1e-8
                    add_bond!(+Jxy, l.bonds[i, j], 1, 2)
                    add_bond!(-Jxy, l.bonds[i, j], 2, 1)
                else
                    add_bond!(-Jxy, l.bonds[i, j], 1, 2)
                    add_bond!(+Jxy, l.bonds[i, j], 2, 1)
                end

                # add other couplings to bond
                add_bond!(Jxx, l.bonds[i, j], 1, 1)
                add_bond!(Jxx, l.bonds[i, j], 2, 2)
                add_bond!(Jzz, l.bonds[i, j], 3, 3)
            end
        end 
    end

    return nothing
end
