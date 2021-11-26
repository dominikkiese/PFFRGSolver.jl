"""
    get_observable_timers() :: Nothing 

Test performance of current observable implementation by Fourier transforming test correlations 
on a few 2D and 3D lattices with 50 x 50 x 0 (2D) or 50 x 50 x 50 (3D) momenta.
"""
function get_observable_timers() :: Nothing

    # init test dummys
    l1   = get_lattice("square",      6, verbose = false); r1 = get_reduced_lattice("heisenberg", [[0.0]], l1, verbose = false)
    l2   = get_lattice("cubic",       6, verbose = false); r2 = get_reduced_lattice("heisenberg", [[0.0]], l2, verbose = false)
    l3   = get_lattice("kagome",      6, verbose = false); r3 = get_reduced_lattice("heisenberg", [[0.0]], l3, verbose = false)
    l4   = get_lattice("hyperkagome", 6, verbose = false); r4 = get_reduced_lattice("heisenberg", [[0.0]], l4, verbose = false)
    χ1   = Float64[exp(-norm(r1.sites[i].vec)) for i in eachindex(r1.sites)]
    χ2   = Float64[exp(-norm(r2.sites[i].vec)) for i in eachindex(r2.sites)]
    χ3   = Float64[exp(-norm(r3.sites[i].vec)) for i in eachindex(r3.sites)]
    χ4   = Float64[exp(-norm(r4.sites[i].vec)) for i in eachindex(r4.sites)]
    k_2D = get_momenta((0.0, 1.0 * pi), (0.0, 1.0 * pi), (0.0, 0.0), (50, 50, 0))
    k_3D = get_momenta((0.0, 1.0 * pi), (0.0, 1.0 * pi), (0.0, 1.0 * pi), (50, 50, 50))

    # init timer
    to = TimerOutput()

    @timeit to "=> Fourier transform" begin
        for rep in 1 : 10
            @timeit to "-> square"      compute_structure_factor(χ1, k_2D, l1, r1)
            @timeit to "-> cubic"       compute_structure_factor(χ2, k_3D, l2, r2)
            @timeit to "-> kagome"      compute_structure_factor(χ3, k_2D, l3, r3)
            @timeit to "-> hyperkagome" compute_structure_factor(χ4, k_3D, l4, r4)
        end 
    end

    show(to)

    return nothing 
end