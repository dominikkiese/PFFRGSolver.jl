"""
    get_lattice_timers() :: Nothing

Function to test current performance of lattice implementation for L = 6.
"""
function get_lattice_timers() :: Nothing 

    println()

    # init timer
    to = TimerOutput()

    # init list of lattices 
    lattices = String["square", "honeycomb", "kagome", "cubic", "fcc", "hyperhoneycomb", "pyrochlore", "diamond"]

    # time lattice building
    for name in lattices
        @timeit to "=> " * name begin 
            @timeit to "-> build"  l = get_lattice(name, 6, verbose = false)
            @timeit to "-> reduce" r = get_reduced_lattice(l, verbose = false)
        end  
    end 

    show(to)
    println()

    return nothing 
end









    