"""
    test_lattices() :: Nothing

Tests, that available lattice implementations produce the right number of sites
"""
function test_lattices() :: Nothing

    lattice_names = ["square",
                     "honeycomb",
                     "kagome",
                     "triangular",
                     "cubic",
                     "fcc",
                     "bcc",
                     "hyperhoneycomb",
                     "hyperkagome",
                     "pyrochlore",
                     "diamond"]
    
    # for each lattice give [bond distance to test, expected number of sites, minimal euclidean size for comprison, maximal euclidean distance for comparison]
    testsizes = [[8,145,2,4],
                 [8,109,2,4],
                 [8,163,2,4],
                 [8,217,2,4],
                 [8,833,2,4],
                 [8,2057,2,4],
                 [8,1241,2,4],
                 [8,319,3,4],
                 [8,415,6,7],
                 [8,1029,2,4],
                 [8,525,2,4]
                 ]
    # run tests for lattice implementations with bond distance norm
    @testset "lattices " begin
        for i in eachindex(lattice_names)
            # generate testdata
            current_name  = lattice_names[i]
            l           = get_lattice(current_name, testsizes[i][1], verbose = false)

            # check that implementations give right number of sites
            @testset "$current_name sitenumber" begin
                @test length(l.sites) == testsizes[i][2]
            end

            # check that euclidean gives same number of sites as sites cut out from larger lattice
            @testset "$current_name euclidean" begin
                for j in testsizes[i][3]:testsizes[i][4]
                    l_bond      = get_lattice(current_name, ceil(Int64, 2.5 * j), verbose = false)
                    l_euclidean = get_lattice(current_name, j,     verbose = false, euclidean = true)
                    nn_distance = norm(l.sites[2].vec)
                    filtered_sites = filter!(x -> norm(x.vec) <= j * nn_distance, l_bond.sites)
                    @test length(l_euclidean.sites) == length(filtered_sites)
                end
            end
        end
    end

    return nothing
end
