"""
    get_frequency_timers() :: Nothing 

Test performance of current frequency implementation for meshes with 30 positive frequencies.
"""
function get_frequency_timers() :: Nothing 

    println()

    # init test dummys
    list = get_mesh(rand(), 1.0, 30, 0.4)
    m    = mesh(31, 31, 31, list, list, list, list, list, list, list)
    w    = rand() 
    v    = rand()
    vp   = rand()

    # init timer
    to = TimerOutput()

    # time single interpolation
    @timeit to "=> single interpolation" begin
        for rep in 1 : 5
            @timeit to "-> index search" get_indices(w, list)
            @timeit to "-> param build"  get_param(w, list)
        end
    end

    # time buffer building 
    @timeit to "=> buffer building" begin
        for rep in 1 : 5
            @timeit to "-> s channel" get_buffer_su2_s(w, v, vp, m)
            @timeit to "-> t channel" get_buffer_su2_t(w, v, vp, m)
            @timeit to "-> u channel" get_buffer_su2_u(w, v, vp, m)
        end
    end
    
    show(to)
    println()

    return nothing
end