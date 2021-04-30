"""
    test_PFFRG() :: Nothing 

Run consistency checks for FRG solver.
"""
function test_PFFRG() :: Nothing 

    println()
    println("Running tests for FRG solver ...")
    println()

    # test frequency implementation
    println("Running frequency tests ...")
    test_frequencies()

    # test action implementation 
    println("Running action tests ...")
    test_action()

    # test flow implementation 
    println("Running flow tests ...") 
    test_flow()

    # test observable implementation 
    println("Running observable tests ...")
    test_observable()

    println("Done.")
    println()

    return nothing 
end