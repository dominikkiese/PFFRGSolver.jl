function test_PFFRG() :: Nothing 

    # test frequency implementation
    println()
    println("Testing frequency implementation ...") 
    test_frequencies()
    println("Done.")
    println()

    # test action implementation 
    println("Testing action implementation ...") 
    test_action()
    println("Done.")
    println()

    # test flow implementation 
    #println("Testing flow implementation ...") 
    #test_flow()
    #println("Done.")

    # test observable implementation 
    println("Testing observable implementation ...") 
    test_observable()
    println("Done.")
    println()

    return nothing 
end