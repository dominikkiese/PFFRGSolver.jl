"""
    test_frequencies() :: Nothing

Run consistency checks for current frequency implementation for meshes with 30 positive frequencies.
"""
function test_frequencies() :: Nothing 

    # init test dummys
    list = get_mesh(rand(), 1.0, 30, 0.4)
    m    = Mesh(31, 31, 31, 31, list, list, list, list, list, list)
    w    = rand() 
    v    = rand()
    vp   = rand()

    # test list bounds
    @testset "list bounds" begin 
        @test list[1]   ≈ 0.0       
        @test list[end] ≈ 1.0 
    end

    # test frequency buffers
    @testset "buffers" begin 
        @testset "s channel" begin 
            b = get_buffer_s(w,     v,  vp, m); @test b.kernel == 4 
            b = get_buffer_s(w,   Inf,  vp, m); @test b.kernel == 3 
            b = get_buffer_s(w,     v, Inf, m); @test b.kernel == 2
            b = get_buffer_s(w,   Inf, Inf, m); @test b.kernel == 1 
            b = get_buffer_s(Inf, Inf, Inf, m); @test b.kernel == 0
        end

        @testset "t channel" begin 
            b = get_buffer_t(w,     v,  vp, m); @test b.kernel == 4 
            b = get_buffer_t(w,   Inf,  vp, m); @test b.kernel == 3 
            b = get_buffer_t(w,     v, Inf, m); @test b.kernel == 2
            b = get_buffer_t(w,   Inf, Inf, m); @test b.kernel == 1 
            b = get_buffer_t(Inf, Inf, Inf, m); @test b.kernel == 0
        end

        @testset "u channel" begin 
            b = get_buffer_u(w,     v,  vp, m); @test b.kernel == 4 
            b = get_buffer_u(w,   Inf,  vp, m); @test b.kernel == 3 
            b = get_buffer_u(w,     v, Inf, m); @test b.kernel == 2
            b = get_buffer_u(w,   Inf, Inf, m); @test b.kernel == 1 
            b = get_buffer_u(Inf, Inf, Inf, m); @test b.kernel == 0
        end
    end

    return nothing 
end