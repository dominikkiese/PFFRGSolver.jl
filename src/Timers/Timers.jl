"""
    get_PFFRG_timers() :: Nothing 

Test current performance of FRG solver.
"""
function get_PFFRG_timers() :: Nothing 

    println()
    println("Testing performance of FRG solver ...")
    println()

    # time lattice
    println("Testing lattice performance ...") 
    println()
    get_lattice_timers()
    println()
    println()

    # time frequencies
    println("Testing frequency performance ...") 
    println()
    get_frequency_timers()
    println()
    println()

    # time action
    println("Testing action performance ...") 
    println()
    get_action_timers()
    println()
    println()

    # time flow equations
    println("Testing flow performance ...") 
    get_flow_timers()
    println()
    println()

    # time observables
    println("Testing observable performance ...") 
    println()
    get_observable_timers() 
    println()
    println()

    println("Done.")

    return nothing 
end