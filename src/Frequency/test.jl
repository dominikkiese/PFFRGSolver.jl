"""
    test_frequencies() :: Nothing

Run consistency checks for current frequency implementation for meshes with 30 positive frequencies.
"""
function test_frequencies() :: Nothing 

    # init test dummys
    list = get_mesh(rand(), 10.0, 30, 0.4)
    m    = Mesh(31, 31, 31, list, [list], [list], [list], [list], [list], [list])
    w    = rand() 
    v    = rand()
    vp   = rand()

    # test list bounds
    @testset "list bounds" begin 
        @test list[1]   ≈ 0.0       
        @test list[end] ≈ 10.0 
    end

    # test frequency buffers
    @testset "buffers and kernels" begin 
        @testset "s channel" begin 
            b = get_buffer_s(1, w,     v,  vp, m); @test b.kernel == 4 
            b = get_buffer_s(1, w,   Inf,  vp, m); @test b.kernel == 3 
            b = get_buffer_s(1, w,     v, Inf, m); @test b.kernel == 2
            b = get_buffer_s(1, w,   Inf, Inf, m); @test b.kernel == 1 
            b = get_buffer_s(1, Inf, Inf, Inf, m); @test b.kernel == 0
        end

        @testset "t channel" begin 
            b = get_buffer_t(1, w,     v,  vp, m); @test b.kernel == 4 
            b = get_buffer_t(1, w,   Inf,  vp, m); @test b.kernel == 3 
            b = get_buffer_t(1, w,     v, Inf, m); @test b.kernel == 2
            b = get_buffer_t(1, w,   Inf, Inf, m); @test b.kernel == 1 
            b = get_buffer_t(1, Inf, Inf, Inf, m); @test b.kernel == 0
        end

        @testset "u channel" begin 
            b = get_buffer_u(1, w,     v,  vp, m); @test b.kernel == 4 
            b = get_buffer_u(1, w,   Inf,  vp, m); @test b.kernel == 3 
            b = get_buffer_u(1, w,     v, Inf, m); @test b.kernel == 2
            b = get_buffer_u(1, w,   Inf, Inf, m); @test b.kernel == 1 
            b = get_buffer_u(1, Inf, Inf, Inf, m); @test b.kernel == 0
        end
    end

    return nothing 
end