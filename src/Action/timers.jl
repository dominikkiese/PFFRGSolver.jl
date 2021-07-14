"""
    get_action_timers() : Nothing 

Test performance of current interpolation routines on the hyperkagome lattice with L = 6.
"""
function get_action_timers() :: Nothing 

    # init test dummys (taking action_su2 as an example)
    list = get_mesh(rand(), 1.0, 30, 0.4)
    m    = Mesh(31, 31, 31, list, list, list, list, list, list, list)
    p1   = get_param(rand(), list)
    p2   = get_param(rand(), list)
    p3   = get_param(rand(), list)
    l    = get_lattice("hyperkagome", 6, verbose = false)
    r    = get_reduced_lattice("heisenberg", [[1.0]], l, verbose = false)
    a    = get_action_empty("su2", r, m); init_action!(l, r, a)
    temp = zeros(Float64, length(r.sites), 1, 1)

    # fill self energy with random values 
    a.Σ .= rand(31)

    # fill s channel with random values
    a.Γ[1].ch_s.q3 .= rand(length(r.sites), 31, 31, 31)

    # init timer 
    to = TimerOutput()

    # time self energy interpolation and extrapolation 
    val   = rand() 
    outer = m.σ[end] + val

    @timeit to "=> interpolation / extrapolation Σ" begin 
        for rep in 1 : 5
            @timeit to "-> interpolation" get_Σ(val, m, a)
            @timeit to "-> extrapolation" get_Σ(outer, m, a)
        end 
    end

    # deref channel and generate temp view 
    ch    = a.Γ[1].ch_s
    vtemp = view(temp, :, 1, 1)

    # time q3 interpolations
    @timeit to "=> interpolation q3 kernel" begin 
        # time sequential routine
        for rep in 1 : 5
            @timeit to "-> sequential" begin 
                for i in 1 : size(temp, 1)
                    temp[i, 1, 1] = get_q3(i, p1, p2, p3, ch)
                end 
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 5
            @timeit to "-> vectorized" begin 
                get_q3_avx!(r, p1, p2, p3, ch, vtemp)
            end 
        end 
    end

    # time q2_2 interpolations
    @timeit to "=> interpolation q2_2 kernel" begin 
        # time sequential routine
        for rep in 1 : 5
            @timeit to "-> sequential" begin 
                for i in 1 : size(temp, 1)
                    temp[i, 1, 1] = get_q2_2(i, p1, p3, ch)
                end 
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 5
            @timeit to "-> vectorized" begin 
                get_q2_2_avx!(r, p1, p3, ch, vtemp)
            end 
        end 
    end

    # time q2_1 interpolations
    @timeit to "=> interpolation q2_1 kernel" begin 
        # time sequential routine
        for rep in 1 : 5
            @timeit to "-> sequential" begin 
                for i in 1 : size(temp, 1)
                    temp[i, 1, 1] = get_q2_1(i, p1, p2, ch)
                end
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 5
            @timeit to "-> vectorized" begin 
                get_q2_1_avx!(r, p1, p2, ch, vtemp)
            end 
        end 
    end

    # time q1 interpolations
    @timeit to "=> interpolation q1 kernel" begin 
        # time sequential routine
        for rep in 1 : 5
            @timeit to "-> sequential" begin 
                for i in 1 : size(temp, 1)
                    temp[i, 1, 1] = get_q1(i, p1, ch)
                end 
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 5
            @timeit to "-> vectorized" begin 
                get_q1_avx!(r, p1, ch, vtemp)
            end 
        end 
    end
    
    show(to)

    return nothing 
end