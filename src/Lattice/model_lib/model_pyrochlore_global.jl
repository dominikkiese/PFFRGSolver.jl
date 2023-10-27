""" 
init_model_pyrochlore_global!(
    J :: Vector{Float64},
    l :: Lattice
    ) :: Nothing

Init Heisenberg Kitaev Model on the Pyrochlore lattice in the global basis. In order for basis sites to be equivalent in this basis, we only consider
J1 and J2, which following to Shannon et al. (10.1103/PhysRevB.95.094422) define the coupling matrix on the 0-1 bond in the GLOBAL basis as
J_01 =  [J2  0  0
          0 J1  0
          0  0 J1]
All other bonds in the global basis can be obtained via the corresponding lattice rotation.
The model here is only defined for nearest neighbors.
"""
function init_model_pyrochlore_global!(
    J :: Vector{Vector{Float64}},
    l :: Lattice
    ) :: Nothing

    # sanity checks
    @assert l.name == "pyrochlore" "Model requires pyrochlore lattice."
    @assert length(J) == 1 "only nearest neighbors are regarded"
    @assert length(J[1]) == 4 "each interaction has to be specified." 

    # iterate over sites and add respective couplings to lattice bonds
    for i in eachindex(l.sites)

        # get nearest neighbor sites 
        nbs = get_nbs(1, l.sites[i], l.sites)

        # determine couplings 
        J1, J2, J3, J4 = J[1]

        for j in nbs 

            # get basis indices
            idxs = (l.sites[i].int[4], l.sites[j].int[4])

            # add Kitaev x-interaction to the respective bonds + the respective Gamma and Heisenberg interactions
            if idxs == (1, 2)
                l.bonds[i, j].exchange .+= [
                     J2  J4  J4;
                    -J4  J1  J3;
                    -J4  J3  J1
                ]
            end
            if idxs == (2, 1)
                l.bonds[i, j].exchange .+= [
                     J2  -J4  -J4;
                    J4  J1  J3;
                    J4  J3  J1
                ]
            end

            if idxs == (3, 4)
                l.bonds[i, j].exchange .+= [
                     J2  -J4   J4;
                     J4   J1  -J3;
                    -J4  -J3   J1
                ]
            end
            if idxs == (4, 3)
                l.bonds[i, j].exchange .+= [
                     J2  J4   -J4;
                     -J4   J1  -J3;
                    J4  -J3   J1
                ]
            end

            if idxs == (1, 3) 
                l.bonds[i, j].exchange .+= [
                    J1  -J4 J3;
                    J4  J2  J4;
                    J3  -J4 J1    
                ]
            end
            if idxs == (3, 1)
                l.bonds[i, j].exchange .+= [
                    J1  J4 J3;
                    -J4  J2  -J4;
                    J3  J4 J1    
                ]
            end

            if idxs == (2, 4)
                l.bonds[i, j].exchange .+= [
                    J1  J4  -J3;
                    -J4 J2  J4;
                    -J3 -J4 J1
                ]
            end
            if idxs == (4, 2)
                l.bonds[i, j].exchange .+= [
                    J1  -J4  -J3;
                    J4 J2  -J4;
                    -J3 J4 J1
                ]
            end

            if idxs == (1, 4)
                l.bonds[i, j].exchange .+= [
                    J1  J3  -J4;
                    J3  J1  -J4;
                    J4  J4  J2
                ]
            end
            if idxs == (4, 1)
                l.bonds[i, j].exchange .+= [
                    J1  J3  J4;
                    J3  J1  J4;
                    -J4  -J4  J2
                ]
            end 
                        
            if idxs == (2, 3)
                l.bonds[i, j].exchange .+= [
                    J1  -J3 J4;
                    -J3 J1  -J4;
                    -J4 J4  J2
                ]
            end 
            if idxs == (3, 2)
                l.bonds[i, j].exchange .+= [
                    J1  -J3 -J4;
                    -J3 J1  J4;
                    J4 -J4  J2
                ]
            end 
        end 
    end

    return nothing 
end 