"""
    test_frequencies() :: Nothing

Run checks for current frequency implementation.
"""
function test_frequencies() :: Nothing
    
    # fix some dummy parameters
    initial = 50.0
    num_σ   = 50 
    num_Ω   = 15
    num_ν   = 10
    p_σ     = 0.3
    p_Ω     = 0.3
    p_ν     = 0.5
    w       = rand() * initial
    v       = rand() * initial
    vp      = rand() * initial

    # build test list and meshes
    list    = get_mesh(0.1, 1.0, num_σ, p_σ)
    m_su2   = get_mesh("su2",   initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    m_u1_dm = get_mesh("u1-dm", initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    
    # test list bounds
    @testset "list bounds" begin 
        @test list[1]   ≈ 0.0       
        @test list[end] ≈ 1.0 
    end 

    # test frequency buffers
    @testset "su2 buffers" begin 
        @testset "s channel" begin 
            b = get_buffer_s(rand(1 : 2), w,     v,  vp, m_su2); @test b.kernel == 4 
            b = get_buffer_s(rand(1 : 2), w,   Inf,  vp, m_su2); @test b.kernel == 3 
            b = get_buffer_s(rand(1 : 2), w,     v, Inf, m_su2); @test b.kernel == 2
            b = get_buffer_s(rand(1 : 2), w,   Inf, Inf, m_su2); @test b.kernel == 1 
            b = get_buffer_s(rand(1 : 2), Inf, Inf, Inf, m_su2); @test b.kernel == 0
        end

        @testset "t channel" begin 
            b = get_buffer_t(rand(1 : 2), w,     v,  vp, m_su2); @test b.kernel == 4 
            b = get_buffer_t(rand(1 : 2), w,   Inf,  vp, m_su2); @test b.kernel == 3 
            b = get_buffer_t(rand(1 : 2), w,     v, Inf, m_su2); @test b.kernel == 2
            b = get_buffer_t(rand(1 : 2), w,   Inf, Inf, m_su2); @test b.kernel == 1 
            b = get_buffer_t(rand(1 : 2), Inf, Inf, Inf, m_su2); @test b.kernel == 0
        end

        @testset "u channel" begin 
            b = get_buffer_u(rand(1 : 2), w,     v,  vp, m_su2); @test b.kernel == 4 
            b = get_buffer_u(rand(1 : 2), w,   Inf,  vp, m_su2); @test b.kernel == 3 
            b = get_buffer_u(rand(1 : 2), w,     v, Inf, m_su2); @test b.kernel == 2
            b = get_buffer_u(rand(1 : 2), w,   Inf, Inf, m_su2); @test b.kernel == 1 
            b = get_buffer_u(rand(1 : 2), Inf, Inf, Inf, m_su2); @test b.kernel == 0
        end
    end

    @testset "u1-dm buffers" begin 
        @testset "s channel" begin 
            b = get_buffer_s(rand(1 : 6), w,     v,  vp, m_u1_dm); @test b.kernel == 4 
            b = get_buffer_s(rand(1 : 6), w,   Inf,  vp, m_u1_dm); @test b.kernel == 3 
            b = get_buffer_s(rand(1 : 6), w,     v, Inf, m_u1_dm); @test b.kernel == 2
            b = get_buffer_s(rand(1 : 6), w,   Inf, Inf, m_u1_dm); @test b.kernel == 1 
            b = get_buffer_s(rand(1 : 6), Inf, Inf, Inf, m_u1_dm); @test b.kernel == 0
        end

        @testset "t channel" begin 
            b = get_buffer_t(rand(1 : 6), w,     v,  vp, m_u1_dm); @test b.kernel == 4 
            b = get_buffer_t(rand(1 : 6), w,   Inf,  vp, m_u1_dm); @test b.kernel == 3 
            b = get_buffer_t(rand(1 : 6), w,     v, Inf, m_u1_dm); @test b.kernel == 2
            b = get_buffer_t(rand(1 : 6), w,   Inf, Inf, m_u1_dm); @test b.kernel == 1 
            b = get_buffer_t(rand(1 : 6), Inf, Inf, Inf, m_u1_dm); @test b.kernel == 0
        end

        @testset "u channel" begin 
            b = get_buffer_u(rand(1 : 6), w,     v,  vp, m_u1_dm); @test b.kernel == 4 
            b = get_buffer_u(rand(1 : 6), w,   Inf,  vp, m_u1_dm); @test b.kernel == 3 
            b = get_buffer_u(rand(1 : 6), w,     v, Inf, m_u1_dm); @test b.kernel == 2
            b = get_buffer_u(rand(1 : 6), w,   Inf, Inf, m_u1_dm); @test b.kernel == 1 
            b = get_buffer_u(rand(1 : 6), Inf, Inf, Inf, m_u1_dm); @test b.kernel == 0
        end
    end

    return nothing 
end