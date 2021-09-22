"""
    get_frequency_timers() :: Nothing 

Test performance of current frequency implementation for meshes with 30 positive frequencies.
"""
function get_frequency_timers() :: Nothing 

    # fix some dummy parameters
    initial = 50.0
    num_σ   = 50 
    num_Ω   = 15
    num_ν   = 10
    p_σ     = 0.3
    p_Ω     = 0.3
    p_ν     = 0.5
    w       = 1.5 * initial
    v       = 1.5 * initial
    vp      = 1.5 * initial

    # build test list and meshes
    list    = get_mesh(0.1, 1.0, num_σ, p_σ)
    m_su2   = get_mesh("su2",   initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    m_u1_dm = get_mesh("u1-dm", initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)

    # init timer
    to = TimerOutput()

    # time single param search
    @timeit to "=> single param search" begin
        for rep in 1 : 100
            @timeit to "-> index search" get_indices(w, list)
            @timeit to "-> param build"  get_param(w, list)
        end
    end

    # time buffer building 
    @timeit to "=> su2 buffer build" begin
        for rep in 1 : 100
            @timeit to "-> s channel" get_buffers_s(w, v, vp, m_su2)
            @timeit to "-> t channel" get_buffers_t(w, v, vp, m_su2)
            @timeit to "-> u channel" get_buffers_u(w, v, vp, m_su2)
        end
    end

    @timeit to "=> u1-dm buffer build" begin
        for rep in 1 : 100
            @timeit to "-> s channel" get_buffers_s(w, v, vp, m_u1_dm)
            @timeit to "-> t channel" get_buffers_t(w, v, vp, m_u1_dm)
            @timeit to "-> u channel" get_buffers_u(w, v, vp, m_u1_dm)
        end
    end
    
    show(to)

    return nothing
end