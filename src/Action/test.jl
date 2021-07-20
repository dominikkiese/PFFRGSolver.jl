"""
    test_action() :: Nothing

Run consistency checks for available action implementations.
"""
function test_action() :: Nothing

    # init dummy grid
    list   = get_mesh(rand(), 10.0, 30, 0.4)
    w_idx  = rand(1 : 31)
    v_idx  = rand(1 : 31)
    vp_idx = rand(1 : 31)

    # init test points for interpolations
    ws  = rand() * list[end]
    vs  = rand() * list[end]
    vsp = rand() * list[end]

    wt  = rand() * list[end]
    vt  = rand() * list[end]
    vtp = rand() * list[end]

    wu  = rand() * list[end]
    vu  = rand() * list[end]
    vup = rand() * list[end]

    # run tests for action_su2
    @testset "action su2" begin
        # generate dummy mesh
        listp = Vector{Float64}[list, list]
        m     = Mesh(31, 31, 31, list, listp, listp, listp, listp, listp, listp)

        # generate action dummy for square lattice Heisenberg model
        l = get_lattice("square", 6, verbose = false)
        r = get_reduced_lattice("heisenberg", [[1.0]], l, verbose = false)
        a = get_action_empty("su2", r, m)
        init_action!(l, r, a)

        # test if bare action is correctly initialized
        @testset "initialization" begin
            @test norm(a.Γ[1].bare) ≈ 0.25
            @test norm(a.Γ[2].bare) ≈ 0.0
        end

        # fill self energy with random values
        a.Σ .= rand(31)

        # ensure antisymmetry
        a.Σ[1] = 0.0

        # generate coefficients for test function
        a111 = rand(3, length(a.Γ), length(r.sites))
        a110 = rand(3, length(a.Γ), length(r.sites))
        a101 = rand(3, length(a.Γ), length(r.sites))
        a011 = rand(3, length(a.Γ), length(r.sites))
        a100 = rand(3, length(a.Γ), length(r.sites))
        a010 = rand(3, length(a.Γ), length(r.sites))
        a001 = rand(3, length(a.Γ), length(r.sites))
        a000 = rand(3, length(a.Γ), length(r.sites))

        # define testfunction (linear in all frequencies, should interpolate exactly)
        f(ch, i, s, w, v, vp) = a111[ch, i, s] * w * v * vp +
                                a110[ch, i, s] * w * v      +
                                a101[ch, i, s] * w     * vp +
                                a011[ch, i, s]     * v * vp +
                                a100[ch, i, s] * w          +
                                a010[ch, i, s]     * v      +
                                a001[ch, i, s]         * vp +
                                a000[ch, i, s]

        # fill channels with values of testfunction
        for i in eachindex(a.Γ)
            for s in 1 : length(r.sites)
                for iw in 1 : m.num_Ω
                    for iv in 1 : m.num_ν
                        for ivp in 1 : m.num_ν
                            a.Γ[i].ch_s.q3[s, iw, iv, ivp] = f(1, i, s, list[iw], list[iv], list[ivp])
                            a.Γ[i].ch_t.q3[s, iw, iv, ivp] = f(2, i, s, list[iw], list[iv], list[ivp])
                            a.Γ[i].ch_u.q3[s, iw, iv, ivp] = f(3, i, s, list[iw], list[iv], list[ivp])
                        end
                    end
                end
            end
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
            @testset "on mesh" begin
                # test sequential routine
                @testset "sequential" begin
                    idx = rand(1 : length(r.sites))

                    for i in eachindex(a.Γ)
                        bs_q3   = get_buffer_s(i, list[w_idx], list[v_idx], list[vp_idx], m)
                        bs_q2_2 = get_buffer_s(i, list[w_idx],         Inf, list[vp_idx], m)
                        bs_q2_1 = get_buffer_s(i, list[w_idx], list[v_idx],          Inf, m)
                        bs_q1   = get_buffer_s(i, list[w_idx],         Inf,          Inf, m)
                        bs_bare = get_buffer_s(i,         Inf,         Inf,          Inf, m)

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
                        bs_q3   = get_buffer_s(i, list[w_idx], list[v_idx], list[vp_idx], m)
                        bs_q2_2 = get_buffer_s(i, list[w_idx],         Inf, list[vp_idx], m)
                        bs_q2_1 = get_buffer_s(i, list[w_idx], list[v_idx],          Inf, m)
                        bs_q1   = get_buffer_s(i, list[w_idx],         Inf,          Inf, m)
                        bs_bare = get_buffer_s(i,         Inf,         Inf,          Inf, m)

                        temp .= 0.0; get_vertex_avx!(r, bs_q3,   a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q3[:, w_idx, v_idx, vp_idx]
                        temp .= 0.0; get_vertex_avx!(r, bs_q2_2, a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q2_2[:, w_idx, vp_idx]
                        temp .= 0.0; get_vertex_avx!(r, bs_q2_1, a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q2_1[:, w_idx, v_idx]
                        temp .= 0.0; get_vertex_avx!(r, bs_q1,   a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_s.q1[:, w_idx]
                        temp .= 0.0; get_vertex_avx!(r, bs_bare, a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test norm(temp[:, i, 1]) ≈ 0.0
                    end
                end
            end

            @testset "interpolated" begin
                # test sequential routine
                @testset "sequential" begin
                    idx = rand(1 : length(r.sites))

                    for i in eachindex(a.Γ)
                        bs_q3   = get_buffer_s(i,  ws,  vs, vsp, m)
                        bs_q2_2 = get_buffer_s(i,  ws, Inf, vsp, m)
                        bs_q2_1 = get_buffer_s(i,  ws,  vs, Inf, m)
                        bs_q1   = get_buffer_s(i,  ws, Inf, Inf, m)
                        bs_bare = get_buffer_s(i, Inf, Inf, Inf, m)

                        @test get_vertex(idx, bs_q3,   a.Γ[i], 1) ≈ f(1, i, idx, ws, vs, vsp)
                        @test get_vertex(idx, bs_q2_2, a.Γ[i], 1) ≈ f(1, i, idx, ws, list[end], vsp)
                        @test get_vertex(idx, bs_q2_1, a.Γ[i], 1) ≈ f(1, i, idx, ws, vs, list[end])
                        @test get_vertex(idx, bs_q1,   a.Γ[i], 1) ≈ f(1, i, idx, ws, list[end], list[end])
                        @test get_vertex(idx, bs_bare, a.Γ[i], 1) ≈ 0.0
                    end
                end

                # test vectorized routine
                @testset "vectorized" begin
                    for i in eachindex(a.Γ)
                        bs_q3   = get_buffer_s(i,  ws,  vs, vsp, m)
                        bs_q2_2 = get_buffer_s(i,  ws, Inf, vsp, m)
                        bs_q2_1 = get_buffer_s(i,  ws,  vs, Inf, m)
                        bs_q1   = get_buffer_s(i,  ws, Inf, Inf, m)
                        bs_bare = get_buffer_s(i, Inf, Inf, Inf, m)

                        temp .= 0.0; get_vertex_avx!(r, bs_q3,   a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(1, i, j, ws, vs, vsp)              for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bs_q2_2, a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(1, i, j, ws, list[end], vsp)       for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bs_q2_1, a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(1, i, j, ws, vs, list[end])        for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bs_q1,   a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(1, i, j, ws, list[end], list[end]) for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bs_bare, a.Γ[i], 1, view(temp, :, i, 1), false, 1.0); @test norm(temp[:, i, 1]) ≈ 0.0
                    end
                end
            end
        end

        # test interpolations in t channel
        @testset "interpolations t channel" begin
            @testset "on mesh" begin
                # test sequential routine
                @testset "sequential" begin
                    idx = rand(1 : length(r.sites))

                    for i in eachindex(a.Γ)
                        bt_q3   = get_buffer_t(i, list[w_idx], list[v_idx], list[vp_idx], m)
                        bt_q2_2 = get_buffer_t(i, list[w_idx],         Inf, list[vp_idx], m)
                        bt_q2_1 = get_buffer_t(i, list[w_idx], list[v_idx],          Inf, m)
                        bt_q1   = get_buffer_t(i, list[w_idx],         Inf,          Inf, m)
                        bt_bare = get_buffer_t(i,         Inf,         Inf,          Inf, m)

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
                        bt_q3   = get_buffer_t(i, list[w_idx], list[v_idx], list[vp_idx], m)
                        bt_q2_2 = get_buffer_t(i, list[w_idx],         Inf, list[vp_idx], m)
                        bt_q2_1 = get_buffer_t(i, list[w_idx], list[v_idx],          Inf, m)
                        bt_q1   = get_buffer_t(i, list[w_idx],         Inf,          Inf, m)
                        bt_bare = get_buffer_t(i,         Inf,         Inf,          Inf, m)
                        
                        temp .= 0.0; get_vertex_avx!(r, bt_q3,   a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q3[:, w_idx, v_idx, vp_idx]
                        temp .= 0.0; get_vertex_avx!(r, bt_q2_2, a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q2_2[:, w_idx, vp_idx]
                        temp .= 0.0; get_vertex_avx!(r, bt_q2_1, a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q2_1[:, w_idx, v_idx]
                        temp .= 0.0; get_vertex_avx!(r, bt_q1,   a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_t.q1[:, w_idx]
                        temp .= 0.0; get_vertex_avx!(r, bt_bare, a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test norm(temp[:, i, 1]) ≈ 0.0
                    end
                end
            end

            @testset "interpolated" begin
                # test sequential routine
                @testset "sequential" begin
                    idx = rand(1 : length(r.sites))

                    for i in eachindex(a.Γ)
                        bt_q3   = get_buffer_t(i,  wt,  vt, vtp, m)
                        bt_q2_2 = get_buffer_t(i,  wt, Inf, vtp, m)
                        bt_q2_1 = get_buffer_t(i,  wt,  vt, Inf, m)
                        bt_q1   = get_buffer_t(i,  wt, Inf, Inf, m)
                        bt_bare = get_buffer_t(i, Inf, Inf, Inf, m)

                        @test get_vertex(idx, bt_q3,   a.Γ[i], 2) ≈ f(2, i, idx, wt, vt, vtp)
                        @test get_vertex(idx, bt_q2_2, a.Γ[i], 2) ≈ f(2, i, idx, wt, list[end], vtp)
                        @test get_vertex(idx, bt_q2_1, a.Γ[i], 2) ≈ f(2, i, idx, wt, vt, list[end])
                        @test get_vertex(idx, bt_q1,   a.Γ[i], 2) ≈ f(2, i, idx, wt, list[end], list[end])
                        @test get_vertex(idx, bt_bare, a.Γ[i], 2) ≈ 0.0
                    end
                end

                # test vectorized routine
                @testset "vectorized" begin
                    for i in eachindex(a.Γ)
                        bt_q3   = get_buffer_t(i,  wt,  vt, vtp, m)
                        bt_q2_2 = get_buffer_t(i,  wt, Inf, vtp, m)
                        bt_q2_1 = get_buffer_t(i,  wt,  vt, Inf, m)
                        bt_q1   = get_buffer_t(i,  wt, Inf, Inf, m)
                        bt_bare = get_buffer_t(i, Inf, Inf, Inf, m)
                        
                        temp .= 0.0; get_vertex_avx!(r, bt_q3,   a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(2, i, j, wt, vt, vtp)              for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bt_q2_2, a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(2, i, j, wt, list[end], vtp)       for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bt_q2_1, a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(2, i, j, wt, vt, list[end])        for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bt_q1,   a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(2, i, j, wt, list[end], list[end]) for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bt_bare, a.Γ[i], 2, view(temp, :, i, 1), false, 1.0); @test norm(temp[:, i, 1]) ≈ 0.0
                    end
                end
            end
        end

        # test interpolations in u channel
        @testset "interpolations u channel" begin
            @testset "on mesh" begin
                # test sequential routine
                @testset "sequential" begin
                    idx = rand(1 : length(r.sites))

                    for i in eachindex(a.Γ)
                        bu_q3   = get_buffer_u(i, list[w_idx], list[v_idx], list[vp_idx], m)
                        bu_q2_2 = get_buffer_u(i, list[w_idx],         Inf, list[vp_idx], m)
                        bu_q2_1 = get_buffer_u(i, list[w_idx], list[v_idx],          Inf, m)
                        bu_q1   = get_buffer_u(i, list[w_idx],         Inf,          Inf, m)
                        bu_bare = get_buffer_u(i,         Inf,         Inf,          Inf, m)

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
                        bu_q3   = get_buffer_u(i, list[w_idx], list[v_idx], list[vp_idx], m)
                        bu_q2_2 = get_buffer_u(i, list[w_idx],         Inf, list[vp_idx], m)
                        bu_q2_1 = get_buffer_u(i, list[w_idx], list[v_idx],          Inf, m)
                        bu_q1   = get_buffer_u(i, list[w_idx],         Inf,          Inf, m)
                        bu_bare = get_buffer_u(i,         Inf,         Inf,          Inf, m)

                        temp .= 0.0; get_vertex_avx!(r, bu_q3,   a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q3[:, w_idx, v_idx, vp_idx]
                        temp .= 0.0; get_vertex_avx!(r, bu_q2_2, a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q2_2[:, w_idx, vp_idx]
                        temp .= 0.0; get_vertex_avx!(r, bu_q2_1, a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q2_1[:, w_idx, v_idx]
                        temp .= 0.0; get_vertex_avx!(r, bu_q1,   a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ a.Γ[i].ch_u.q1[:, w_idx]
                        temp .= 0.0; get_vertex_avx!(r, bu_bare, a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test norm(temp[:, i, 1]) ≈ 0.0
                    end
                end
            end

            @testset "interpolated" begin
                # test sequential routine
                @testset "sequential" begin
                    idx = rand(1 : length(r.sites))

                    for i in eachindex(a.Γ)
                        bu_q3   = get_buffer_u(i,  wu,  vu, vup, m)
                        bu_q2_2 = get_buffer_u(i,  wu, Inf, vup, m)
                        bu_q2_1 = get_buffer_u(i,  wu,  vu, Inf, m)
                        bu_q1   = get_buffer_u(i,  wu, Inf, Inf, m)
                        bu_bare = get_buffer_u(i, Inf, Inf, Inf, m)

                        @test get_vertex(idx, bu_q3,   a.Γ[i], 3) ≈ f(3, i, idx, wu, vu, vup)
                        @test get_vertex(idx, bu_q2_2, a.Γ[i], 3) ≈ f(3, i, idx, wu, list[end], vup)
                        @test get_vertex(idx, bu_q2_1, a.Γ[i], 3) ≈ f(3, i, idx, wu, vu, list[end])
                        @test get_vertex(idx, bu_q1,   a.Γ[i], 3) ≈ f(3, i, idx, wu, list[end], list[end])
                        @test get_vertex(idx, bu_bare, a.Γ[i], 3) ≈ 0.0
                    end
                end

                # test vectorized routine
                @testset "vectorized" begin
                    for i in eachindex(a.Γ)
                        bu_q3   = get_buffer_u(i,  wu,  vu, vup, m)
                        bu_q2_2 = get_buffer_u(i,  wu, Inf, vup, m)
                        bu_q2_1 = get_buffer_u(i,  wu,  vu, Inf, m)
                        bu_q1   = get_buffer_u(i,  wu, Inf, Inf, m)
                        bu_bare = get_buffer_u(i, Inf, Inf, Inf, m)

                        temp .= 0.0; get_vertex_avx!(r, bu_q3,   a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(3, i, j, wu, vu, vup)              for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bu_q2_2, a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(3, i, j, wu, list[end], vup)       for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bu_q2_1, a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(3, i, j, wu, vu, list[end])        for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bu_q1,   a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test temp[:, i, 1]       ≈ [f(3, i, j, wu, list[end], list[end]) for j in 1 : length(r.sites)]
                        temp .= 0.0; get_vertex_avx!(r, bu_bare, a.Γ[i], 3, view(temp, :, i, 1), false, 1.0); @test norm(temp[:, i, 1]) ≈ 0.0
                    end
                end
            end
        end
    end

    return nothing
end
