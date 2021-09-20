"""
    get_action_timers() : Nothing 

Test performance of current interpolation routines.
"""
function get_action_timers() :: Nothing 

    # fix some dummy parameters
    initial = 50.0
    num_σ   = 50 
    num_Ω   = 15 
    num_ν   = 10
    p_σ     = 0.3
    p_Ω     = 0.3
    p_ν     = 0.5

    # generate action dummy for square lattice Heisenberg model
    l = get_lattice("square", 10, verbose = false)
    r = get_reduced_lattice("heisenberg", [[1.0]], l, verbose = false)
    m = get_mesh("su2", initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    a = get_action_empty("su2", r, m)
    init_action!(l, r, a)

    # fill self energy with random values and ensure antisymmetry
    a.Σ    .= rand(Float64, m.num_σ)
    a.Σ[1]  = 0.0

    # fill s channel with random values
    a.Γ[1].ch_s.q3 .= rand(length(r.sites), m.num_Ω, m.num_ν, m.num_ν)

    # generate dummy buffer and temp view
    temp  = zeros(Float64, length(r.sites), 1, 1)
    vtemp = view(temp, :, 1, 1)
    ch    = a.Γ[1].ch_s
    
    # generate dummy interpolation points / parameters 
    val   = rand() * m.σ[end]
    outer = m.σ[end] + val
    p1    = get_param(rand() * m.Ωs[1][end], m.Ωs[1])
    p2    = get_param(rand() * m.νs[1][end], m.νs[1])
    p3    = get_param(rand() * m.νs[1][end], m.νs[1])
    
    # init timer 
    to = TimerOutput()

    # time self energy interpolation and extrapolation 
    @timeit to "=> interpolation / extrapolation Σ" begin 
        for rep in 1 : 100
            @timeit to "-> interpolation" get_Σ(val, m, a)
            @timeit to "-> extrapolation" get_Σ(outer, m, a)
        end 
    end

    # time q3 interpolations
    @timeit to "=> interpolation q3 kernel" begin 
        # time sequential routine
        for rep in 1 : 100
            @timeit to "-> sequential" begin 
                for site in eachindex(r.sites)
                    temp[site, 1, 1] = get_q3(site, p1, p2, p3, ch)
                end 
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 100
            @timeit to "-> vectorized" begin 
                get_q3_avx!(r, p1, p2, p3, ch, vtemp, false, 1.0)
            end 
        end 
    end

    # time q2_2 interpolations
    @timeit to "=> interpolation q2_2 kernel" begin 
        # time sequential routine
        for rep in 1 : 100
            @timeit to "-> sequential" begin 
                for site in eachindex(r.sites)
                    temp[site, 1, 1] = get_q2_2(site, p1, p3, ch)
                end 
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 100
            @timeit to "-> vectorized" begin 
                get_q2_2_avx!(r, p1, p3, ch, vtemp, false, 1.0)
            end 
        end 
    end

    # time q2_1 interpolations
    @timeit to "=> interpolation q2_1 kernel" begin 
        # time sequential routine
        for rep in 1 : 100
            @timeit to "-> sequential" begin 
                for site in eachindex(r.sites)
                    temp[site, 1, 1] = get_q2_1(site, p1, p2, ch)
                end
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 100
            @timeit to "-> vectorized" begin 
                get_q2_1_avx!(r, p1, p2, ch, vtemp, false, 1.0)
            end 
        end 
    end

    # time q1 interpolations
    @timeit to "=> interpolation q1 kernel" begin 
        # time sequential routine
        for rep in 1 : 100
            @timeit to "-> sequential" begin 
                for site in eachindex(r.sites)
                    temp[site, 1, 1] = get_q1(site, p1, ch)
                end 
            end 
        end 
        
        # time vectorized routine
        for rep in 1 : 100
            @timeit to "-> vectorized" begin 
                get_q1_avx!(r, p1, ch, vtemp, false, 1.0)
            end 
        end 
    end
    
    show(to)

    return nothing 
end