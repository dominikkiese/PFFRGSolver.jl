using PFFRG 

println()
println("Testing performance of FRG solver ...")
println()

# time lattice
println("Testing lattice performance ...") 
get_lattice_timers()
println()

# time frequencies
println("Testing frequency performance ...") 
get_frequency_timers()
println()

# time action
println("Testing action performance ...") 
get_action_timers()
println()

# time flow equations
println("Testing flow performance ...") 
get_flow_timers()
println()

# time observables
println("Testing observable performance ...") 
get_observable_timers() 
println()

println("Done.")
println()