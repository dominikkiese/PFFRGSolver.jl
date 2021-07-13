"""
    test_FM() :: Nothing 

Computes structure factors for ferromagnetic correlations on different lattices and checks if the maximum resides at the Γ point.
"""
function test_FM() :: Nothing

    # init test dummy
    k = get_momenta((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (10, 10, 10))

    # initialize ferromagnetic test correlations
    @testset "FM corr" begin 
        for name in ["square", "cubic", "kagome", "hyperkagome"]
            l = get_lattice(name, 6, verbose = false)
            r = get_reduced_lattice("heisenberg", [[0.0]], l, verbose = false)
            χ = Float64[exp(-norm(r.sites[i].vec)) for i in eachindex(r.sites)]
            s = compute_structure_factor(χ, k, l, r)
            
            @test norm(k[:, argmax(abs.(s))]) ≈ 0.0 
        end 
    end 

    return nothing 
end

"""
    test_observable() :: Nothing 

Consistency checks for current implementation of observables.
"""
function test_observable() :: Nothing 

    # test if ferromagnetic correlations are Fourier transformed correctly 
    test_FM() 

    return nothing 
end