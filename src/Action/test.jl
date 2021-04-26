"""
    test_action() :: Nothing 

Run consistency checks for available action implementations.
"""
function test_action() :: Nothing

    println()

    # init test dummys 
    list   = get_mesh(rand(), 1.0, 30, 0.4)
    m      = mesh(31, 31, 31, list, list, list, list, list, list, list)
    w_idx  = rand(1 : 31) 
    v_idx  = rand(1 : 31)
    vp_idx = rand(1 : 31)

    # run tests for action_sun
    @testset "action sun" begin
        # generate action dummy for square lattice Heisenberg model
        l = get_lattice("square", 6, verbose = false)
        init_model!("heisenberg", [[1.0]], l)
        r = get_reduced_lattice(l, verbose = false)
        a = get_action_empty("sun", r, m)
        init_action!(l, r, a)
        
        # test if bare action is correctly initialized
        @testset "initialization" begin 
            @test norm(a.Γ[1].bare) ≈ 1.0 
            @test norm(a.Γ[2].bare) ≈ 0.0
        end

        # fill self energy with random values 
        a.Σ .= rand(31)

        # fill channels with random values
        for i in eachindex(a.Γ)
            a.Γ[i].ch_s.q3 .= rand(length(r.sites), 31, 31, 31)
            a.Γ[i].ch_t.q3 .= rand(length(r.sites), 31, 31, 31)
            a.Γ[i].ch_u.q3 .= rand(length(r.sites), 31, 31, 31)
        end

        # set asymptotic limits
        limits!(a)

        # test self energy interpolation and extrapolation 
        @testset "interpolation / extrapolation Σ" begin
            outer = m.σ[end] + rand()
            @test get_Σ(m.σ[w_idx], m, a) ≈ a.Σ[w_idx]
            @test get_Σ(outer, m, a)      ≈ a.Σ[end] * m.σ[end] / outer
        end

        # prepare buffer for vectorized routines
        temp = zeros(Float64, length(r.sites), 2, 1)

        # test interpolations in s channel
        @testset "interpolations s channel" begin 
            # build buffers
            bs_q3   = get_buffer_sun_s(m.Ωs[w_idx], m.νs[v_idx], m.νs[vp_idx], m)
            bs_q2_2 = get_buffer_sun_s(m.Ωs[w_idx],         Inf, m.νs[vp_idx], m)
            bs_q2_1 = get_buffer_sun_s(m.Ωs[w_idx], m.νs[v_idx],          Inf, m)
            bs_q1   = get_buffer_sun_s(m.Ωs[w_idx],         Inf,          Inf, m)
            bs_bare = get_buffer_sun_s(        Inf,         Inf,          Inf, m)

            # test sequential routine
            @testset "sequential" begin
                idx = rand(1 : length(r.sites))

                for i in eachindex(a.Γ)
                    @test get_vertex(idx, bs_q3,   a.Γ[i], 1) ≈ a.Γ[i].ch_s.q3[idx, w_idx, v_idx, vp_idx]
                    @test get_vertex(idx, bs_q2_2, a.Γ[i], 1) ≈ a.Γ[i].ch_s.q2_2[idx, w_idx, vp_idx]
                    @test get_vertex(idx, bs_q2_1, a.Γ[i], 1) ≈ a.Γ[i].ch_s.q2_1[idx, w_idx, v_idx]
                    @test get_vertex(idx, bs_q1,   a.Γ[i], 1) ≈ a.Γ[i].ch_s.q1[idx, w_idx]
                    @test get_vertex(idx, bs_bare, a.Γ[i], 1) ≈ 0.0
                end 
            end
            
            # test vectorized routine
            @testset "vectorized" begin
                for i in eachindex(a.Γ)
                    temp .= 0.0; get_vertex_avx!(r, bs_q3,   a.Γ[i], 1, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q3[:, w_idx, v_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, bs_q2_2, a.Γ[i], 1, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q2_2[:, w_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, bs_q2_1, a.Γ[i], 1, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q2_1[:, w_idx, v_idx]
                    temp .= 0.0; get_vertex_avx!(r, bs_q1,   a.Γ[i], 1, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q1[:, w_idx]
                    temp .= 0.0; get_vertex_avx!(r, bs_bare, a.Γ[i], 1, view(temp, :, i, 1)); @test norm(temp[:, i, 1]) ≈ 0.0
                end 
            end
        end

        # test interpolations in t channel
        @testset "interpolations t channel" begin 
            # build buffers
            bt_q3   = get_buffer_sun_t(m.Ωt[w_idx], m.νt[v_idx], m.νt[vp_idx], m)
            bt_q2_2 = get_buffer_sun_t(m.Ωt[w_idx],         Inf, m.νt[vp_idx], m)
            bt_q2_1 = get_buffer_sun_t(m.Ωt[w_idx], m.νt[v_idx],          Inf, m)
            bt_q1   = get_buffer_sun_t(m.Ωt[w_idx],         Inf,          Inf, m)
            bt_bare = get_buffer_sun_t(        Inf,         Inf,          Inf, m)

            # test sequential routine
            @testset "sequential" begin
                idx = rand(1 : length(r.sites))

                for i in eachindex(a.Γ)
                    @test get_vertex(idx, bt_q3,   a.Γ[i], 2) ≈ a.Γ[i].ch_t.q3[idx, w_idx, v_idx, vp_idx]
                    @test get_vertex(idx, bt_q2_2, a.Γ[i], 2) ≈ a.Γ[i].ch_t.q2_2[idx, w_idx, vp_idx]
                    @test get_vertex(idx, bt_q2_1, a.Γ[i], 2) ≈ a.Γ[i].ch_t.q2_1[idx, w_idx, v_idx]
                    @test get_vertex(idx, bt_q1,   a.Γ[i], 2) ≈ a.Γ[i].ch_t.q1[idx, w_idx]
                    @test get_vertex(idx, bt_bare, a.Γ[i], 2) ≈ 0.0
                end 
            end
            
            # test vectorized routine
            @testset "vectorized" begin
                for i in eachindex(a.Γ)
                    temp .= 0.0; get_vertex_avx!(r, bt_q3,   a.Γ[i], 2, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q3[:, w_idx, v_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, bt_q2_2, a.Γ[i], 2, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q2_2[:, w_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, bt_q2_1, a.Γ[i], 2, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q2_1[:, w_idx, v_idx]
                    temp .= 0.0; get_vertex_avx!(r, bt_q1,   a.Γ[i], 2, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q1[:, w_idx]
                    temp .= 0.0; get_vertex_avx!(r, bt_bare, a.Γ[i], 2, view(temp, :, i, 1)); @test norm(temp[:, i, 1]) ≈ 0.0
                end 
            end
        end

        # test interpolations in u channel
        @testset "interpolations u channel" begin 
            # build buffers
            bu_q3   = get_buffer_sun_u(m.Ωu[w_idx], m.νu[v_idx], m.νu[vp_idx], m)
            bu_q2_2 = get_buffer_sun_u(m.Ωu[w_idx],         Inf, m.νu[vp_idx], m)
            bu_q2_1 = get_buffer_sun_u(m.Ωu[w_idx], m.νu[v_idx],          Inf, m)
            bu_q1   = get_buffer_sun_u(m.Ωu[w_idx],         Inf,          Inf, m)
            bu_bare = get_buffer_sun_u(        Inf,         Inf,          Inf, m)

            # test sequential routine
            @testset "sequential" begin
                idx = rand(1 : length(r.sites))

                for i in eachindex(a.Γ)
                    @test get_vertex(idx, bu_q3,   a.Γ[i], 3) ≈ a.Γ[i].ch_u.q3[idx, w_idx, v_idx, vp_idx]
                    @test get_vertex(idx, bu_q2_2, a.Γ[i], 3) ≈ a.Γ[i].ch_u.q2_2[idx, w_idx, vp_idx]
                    @test get_vertex(idx, bu_q2_1, a.Γ[i], 3) ≈ a.Γ[i].ch_u.q2_1[idx, w_idx, v_idx]
                    @test get_vertex(idx, bu_q1,   a.Γ[i], 3) ≈ a.Γ[i].ch_u.q1[idx, w_idx]
                    @test get_vertex(idx, bu_bare, a.Γ[i], 3) ≈ 0.0
                end 
            end
            
            # test vectorized routine
            @testset "vectorized" begin
                for i in eachindex(a.Γ)
                    temp .= 0.0; get_vertex_avx!(r, bu_q3,   a.Γ[i], 3, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q3[:, w_idx, v_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, bu_q2_2, a.Γ[i], 3, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q2_2[:, w_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, bu_q2_1, a.Γ[i], 3, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q2_1[:, w_idx, v_idx]
                    temp .= 0.0; get_vertex_avx!(r, bu_q1,   a.Γ[i], 3, view(temp, :, i, 1)); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q1[:, w_idx]
                    temp .= 0.0; get_vertex_avx!(r, bu_bare, a.Γ[i], 3, view(temp, :, i, 1)); @test norm(temp[:, i, 1]) ≈ 0.0
                end 
            end
        end
    end 

    println()

    return nothing 
end                        