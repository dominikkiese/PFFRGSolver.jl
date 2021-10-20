# run interpolation checks
function test_interpolations(
    r :: Reduced_lattice,
    m :: Mesh,
    a :: Action
    ) :: Nothing

    # init discrete test points for interpolations
    σ_idx  = rand(1 : m.num_σ)
    w_idx  = rand(1 : m.num_Ω)
    v_idx  = rand(1 : m.num_ν)
    vp_idx = rand(1 : m.num_ν)

    # init continous test points for interpolations
    w  = rand() * m.Ωs[1][end]
    v  = rand() * m.νs[1][end]
    vp = rand() * m.νs[1][end]

    # fill self energy with random values and ensure antisymmetry
    a.Σ    .= rand(Float64, m.num_σ)
    a.Σ[1]  = 0.0

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
    f(ch, comp, site, w, v, vp) = a111[ch, comp, site] * w * v * vp +
                                  a110[ch, comp, site] * w * v      +
                                  a101[ch, comp, site] * w     * vp +
                                  a011[ch, comp, site]     * v * vp +
                                  a100[ch, comp, site] * w          +
                                  a010[ch, comp, site]     * v      +
                                  a001[ch, comp, site]         * vp +
                                  a000[ch, comp, site]

    # fill channels with values of testfunction
    for comp in eachindex(a.Γ)
        for site in eachindex(r.sites)
            for iw in 1 : m.num_Ω
                for iv in 1 : m.num_ν
                    for ivp in 1 : m.num_ν
                        a.Γ[comp].ch_s.q3[site, iw, iv, ivp] = f(1, comp, site, m.Ωs[comp][iw], m.νs[comp][iv], m.νs[comp][ivp])
                        a.Γ[comp].ch_t.q3[site, iw, iv, ivp] = f(2, comp, site, m.Ωt[comp][iw], m.νt[comp][iv], m.νt[comp][ivp])
                        a.Γ[comp].ch_u.q3[site, iw, iv, ivp] = f(3, comp, site, m.Ωs[comp][iw], m.νs[comp][iv], m.νs[comp][ivp])
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
        @test get_Σ(m.σ[σ_idx], m, a) ≈ a.Σ[σ_idx]
        @test get_Σ(outer, m, a)      ≈ a.Σ[end] * m.σ[end] / outer
    end

    # prepare buffer for vectorized routines
    temp = zeros(Float64, length(r.sites), length(a.Γ), 1)

    # test interpolations in s channel
    @testset "interpolations s channel" begin
        @testset "on mesh" begin
            @testset "sequential" begin
                for comp in eachindex(a.Γ)
                    idx    = rand(1 : length(r.sites))
                    b_q3   = get_buffers_s(m.Ωs[comp][w_idx], m.νs[comp][v_idx], m.νs[comp][vp_idx], m)
                    b_q2_2 = get_buffers_s(m.Ωs[comp][w_idx],               Inf, m.νs[comp][vp_idx], m)
                    b_q2_1 = get_buffers_s(m.Ωs[comp][w_idx], m.νs[comp][v_idx],                Inf, m)
                    b_q1   = get_buffers_s(m.Ωs[comp][w_idx],               Inf,                Inf, m)
                    b_bare = get_buffers_s(              Inf,               Inf,                Inf, m)

                    @test get_vertex(idx, b_q3[comp],   a.Γ[comp], 1) ≈ a.Γ[comp].ch_s.q3[idx, w_idx, v_idx, vp_idx]
                    @test get_vertex(idx, b_q2_2[comp], a.Γ[comp], 1) ≈ a.Γ[comp].ch_s.q2_2[idx, w_idx, vp_idx]
                    @test get_vertex(idx, b_q2_1[comp], a.Γ[comp], 1) ≈ a.Γ[comp].ch_s.q2_1[idx, w_idx, v_idx]
                    @test get_vertex(idx, b_q1[comp],   a.Γ[comp], 1) ≈ a.Γ[comp].ch_s.q1[idx, w_idx]
                    @test get_vertex(idx, b_bare[comp], a.Γ[comp], 1) ≈ 0.0
                end
            end

            @testset "vectorized" begin
                for comp in eachindex(a.Γ)
                    b_q3   = get_buffers_s(m.Ωs[comp][w_idx], m.νs[comp][v_idx], m.νs[comp][vp_idx], m)
                    b_q2_2 = get_buffers_s(m.Ωs[comp][w_idx],               Inf, m.νs[comp][vp_idx], m)
                    b_q2_1 = get_buffers_s(m.Ωs[comp][w_idx], m.νs[comp][v_idx],                Inf, m)
                    b_q1   = get_buffers_s(m.Ωs[comp][w_idx],               Inf,                Inf, m)
                    b_bare = get_buffers_s(              Inf,               Inf,                Inf, m)

                    temp .= 0.0; get_vertex_avx!(r, b_q3[comp],   a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_s.q3[:, w_idx, v_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_2[comp], a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_s.q2_2[:, w_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_1[comp], a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_s.q2_1[:, w_idx, v_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q1[comp],   a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_s.q1[:, w_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_bare[comp], a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test norm(temp[:, comp, 1]) ≈ 0.0
                end
            end
        end

        @testset "interpolated" begin
            @testset "sequential" begin
                for comp in eachindex(a.Γ)
                    idx    = rand(1 : length(r.sites))
                    b_q3   = get_buffers_s(  w,   v,  vp, m)
                    b_q2_2 = get_buffers_s(  w, Inf,  vp, m)
                    b_q2_1 = get_buffers_s(  w,   v, Inf, m)
                    b_q1   = get_buffers_s(  w, Inf, Inf, m)
                    b_bare = get_buffers_s(Inf, Inf, Inf, m)

                    @test get_vertex(idx, b_q3[comp],   a.Γ[comp], 1) ≈ f(1, comp, idx, w, v, vp)
                    @test get_vertex(idx, b_q2_2[comp], a.Γ[comp], 1) ≈ f(1, comp, idx, w, m.νs[comp][end], vp)
                    @test get_vertex(idx, b_q2_1[comp], a.Γ[comp], 1) ≈ f(1, comp, idx, w, v, m.νs[comp][end])
                    @test get_vertex(idx, b_q1[comp],   a.Γ[comp], 1) ≈ f(1, comp, idx, w, m.νs[comp][end], m.νs[comp][end])
                    @test get_vertex(idx, b_bare[comp], a.Γ[comp], 1) ≈ 0.0
                end
            end

            @testset "vectorized" begin
                for comp in eachindex(a.Γ)
                    b_q3   = get_buffers_s(  w,   v,  vp, m)
                    b_q2_2 = get_buffers_s(  w, Inf,  vp, m)
                    b_q2_1 = get_buffers_s(  w,   v, Inf, m)
                    b_q1   = get_buffers_s(  w, Inf, Inf, m)
                    b_bare = get_buffers_s(Inf, Inf, Inf, m)

                    temp .= 0.0; get_vertex_avx!(r, b_q3[comp],   a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(1, comp, j, w, v, vp)                            for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_2[comp], a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(1, comp, j, w, m.νs[comp][end], vp)              for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_1[comp], a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(1, comp, j, w, v, m.νs[comp][end])               for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q1[comp],   a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(1, comp, j, w, m.νs[comp][end], m.νs[comp][end]) for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_bare[comp], a.Γ[comp], 1, view(temp, :, comp, 1), false, 1.0); @test norm(temp[:, comp, 1]) ≈ 0.0
                end
            end
        end
    end

    # test interpolations in t channel
    @testset "interpolations t channel" begin
        @testset "on mesh" begin
            @testset "sequential" begin
                for comp in eachindex(a.Γ)
                    idx    = rand(1 : length(r.sites))
                    b_q3   = get_buffers_t(m.Ωt[comp][w_idx], m.νt[comp][v_idx], m.νt[comp][vp_idx], m)
                    b_q2_2 = get_buffers_t(m.Ωt[comp][w_idx],               Inf, m.νt[comp][vp_idx], m)
                    b_q2_1 = get_buffers_t(m.Ωt[comp][w_idx], m.νt[comp][v_idx],                Inf, m)
                    b_q1   = get_buffers_t(m.Ωt[comp][w_idx],               Inf,                Inf, m)
                    b_bare = get_buffers_t(              Inf,               Inf,                Inf, m)

                    @test get_vertex(idx, b_q3[comp],   a.Γ[comp], 2) ≈ a.Γ[comp].ch_t.q3[idx, w_idx, v_idx, vp_idx]
                    @test get_vertex(idx, b_q2_2[comp], a.Γ[comp], 2) ≈ a.Γ[comp].ch_t.q2_2[idx, w_idx, vp_idx]
                    @test get_vertex(idx, b_q2_1[comp], a.Γ[comp], 2) ≈ a.Γ[comp].ch_t.q2_1[idx, w_idx, v_idx]
                    @test get_vertex(idx, b_q1[comp],   a.Γ[comp], 2) ≈ a.Γ[comp].ch_t.q1[idx, w_idx]
                    @test get_vertex(idx, b_bare[comp], a.Γ[comp], 2) ≈ 0.0
                end
            end

            @testset "vectorized" begin
                for comp in eachindex(a.Γ)
                    b_q3   = get_buffers_t(m.Ωt[comp][w_idx], m.νt[comp][v_idx], m.νt[comp][vp_idx], m)
                    b_q2_2 = get_buffers_t(m.Ωt[comp][w_idx],               Inf, m.νt[comp][vp_idx], m)
                    b_q2_1 = get_buffers_t(m.Ωt[comp][w_idx], m.νt[comp][v_idx],                Inf, m)
                    b_q1   = get_buffers_t(m.Ωt[comp][w_idx],               Inf,                Inf, m)
                    b_bare = get_buffers_t(              Inf,               Inf,                Inf, m)

                    temp .= 0.0; get_vertex_avx!(r, b_q3[comp],   a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_t.q3[:, w_idx, v_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_2[comp], a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_t.q2_2[:, w_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_1[comp], a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_t.q2_1[:, w_idx, v_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q1[comp],   a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_t.q1[:, w_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_bare[comp], a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test norm(temp[:, comp, 1]) ≈ 0.0
                end
            end
        end

        @testset "interpolated" begin
            @testset "sequential" begin
                for comp in eachindex(a.Γ)
                    idx    = rand(1 : length(r.sites))
                    b_q3   = get_buffers_t(  w,   v,  vp, m)
                    b_q2_2 = get_buffers_t(  w, Inf,  vp, m)
                    b_q2_1 = get_buffers_t(  w,   v, Inf, m)
                    b_q1   = get_buffers_t(  w, Inf, Inf, m)
                    b_bare = get_buffers_t(Inf, Inf, Inf, m)

                    @test get_vertex(idx, b_q3[comp],   a.Γ[comp], 2) ≈ f(2, comp, idx, w, v, vp)
                    @test get_vertex(idx, b_q2_2[comp], a.Γ[comp], 2) ≈ f(2, comp, idx, w, m.νt[comp][end], vp)
                    @test get_vertex(idx, b_q2_1[comp], a.Γ[comp], 2) ≈ f(2, comp, idx, w, v, m.νt[comp][end])
                    @test get_vertex(idx, b_q1[comp],   a.Γ[comp], 2) ≈ f(2, comp, idx, w, m.νt[comp][end], m.νt[comp][end])
                    @test get_vertex(idx, b_bare[comp], a.Γ[comp], 2) ≈ 0.0
                end
            end

            @testset "vectorized" begin
                for comp in eachindex(a.Γ)
                    b_q3   = get_buffers_t(  w,   v,  vp, m)
                    b_q2_2 = get_buffers_t(  w, Inf,  vp, m)
                    b_q2_1 = get_buffers_t(  w,   v, Inf, m)
                    b_q1   = get_buffers_t(  w, Inf, Inf, m)
                    b_bare = get_buffers_t(Inf, Inf, Inf, m)

                    temp .= 0.0; get_vertex_avx!(r, b_q3[comp],   a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(2, comp, j, w, v, vp)                            for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_2[comp], a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(2, comp, j, w, m.νt[comp][end], vp)              for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_1[comp], a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(2, comp, j, w, v, m.νt[comp][end])               for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q1[comp],   a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(2, comp, j, w, m.νt[comp][end], m.νt[comp][end]) for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_bare[comp], a.Γ[comp], 2, view(temp, :, comp, 1), false, 1.0); @test norm(temp[:, comp, 1]) ≈ 0.0
                end
            end
        end
    end

    # test interpolations in u channel
    @testset "interpolations u channel" begin
        @testset "on mesh" begin
            @testset "sequential" begin
                for comp in eachindex(a.Γ)
                    idx    = rand(1 : length(r.sites))
                    b_q3   = get_buffers_u(m.Ωs[comp][w_idx], m.νs[comp][v_idx], m.νs[comp][vp_idx], m)
                    b_q2_2 = get_buffers_u(m.Ωs[comp][w_idx],               Inf, m.νs[comp][vp_idx], m)
                    b_q2_1 = get_buffers_u(m.Ωs[comp][w_idx], m.νs[comp][v_idx],                Inf, m)
                    b_q1   = get_buffers_u(m.Ωs[comp][w_idx],               Inf,                Inf, m)
                    b_bare = get_buffers_u(              Inf,               Inf,                Inf, m)

                    @test get_vertex(idx, b_q3[comp],   a.Γ[comp], 3) ≈ a.Γ[comp].ch_u.q3[idx, w_idx, v_idx, vp_idx]
                    @test get_vertex(idx, b_q2_2[comp], a.Γ[comp], 3) ≈ a.Γ[comp].ch_u.q2_2[idx, w_idx, vp_idx]
                    @test get_vertex(idx, b_q2_1[comp], a.Γ[comp], 3) ≈ a.Γ[comp].ch_u.q2_1[idx, w_idx, v_idx]
                    @test get_vertex(idx, b_q1[comp],   a.Γ[comp], 3) ≈ a.Γ[comp].ch_u.q1[idx, w_idx]
                    @test get_vertex(idx, b_bare[comp], a.Γ[comp], 3) ≈ 0.0
                end
            end

            @testset "vectorized" begin
                for comp in eachindex(a.Γ)
                    b_q3   = get_buffers_u(m.Ωs[comp][w_idx], m.νs[comp][v_idx], m.νs[comp][vp_idx], m)
                    b_q2_2 = get_buffers_u(m.Ωs[comp][w_idx],               Inf, m.νs[comp][vp_idx], m)
                    b_q2_1 = get_buffers_u(m.Ωs[comp][w_idx], m.νs[comp][v_idx],                Inf, m)
                    b_q1   = get_buffers_u(m.Ωs[comp][w_idx],               Inf,                Inf, m)
                    b_bare = get_buffers_u(              Inf,               Inf,                Inf, m)

                    temp .= 0.0; get_vertex_avx!(r, b_q3[comp],   a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_u.q3[:, w_idx, v_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_2[comp], a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_u.q2_2[:, w_idx, vp_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_1[comp], a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_u.q2_1[:, w_idx, v_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_q1[comp],   a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ a.Γ[comp].ch_u.q1[:, w_idx]
                    temp .= 0.0; get_vertex_avx!(r, b_bare[comp], a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test norm(temp[:, comp, 1]) ≈ 0.0
                end
            end
        end

        @testset "interpolated" begin
            @testset "sequential" begin
                for comp in eachindex(a.Γ)
                    idx    = rand(1 : length(r.sites))
                    b_q3   = get_buffers_u(  w,   v,  vp, m)
                    b_q2_2 = get_buffers_u(  w, Inf,  vp, m)
                    b_q2_1 = get_buffers_u(  w,   v, Inf, m)
                    b_q1   = get_buffers_u(  w, Inf, Inf, m)
                    b_bare = get_buffers_u(Inf, Inf, Inf, m)

                    @test get_vertex(idx, b_q3[comp],   a.Γ[comp], 3) ≈ f(3, comp, idx, w, v, vp)
                    @test get_vertex(idx, b_q2_2[comp], a.Γ[comp], 3) ≈ f(3, comp, idx, w, m.νs[comp][end], vp)
                    @test get_vertex(idx, b_q2_1[comp], a.Γ[comp], 3) ≈ f(3, comp, idx, w, v, m.νs[comp][end])
                    @test get_vertex(idx, b_q1[comp],   a.Γ[comp], 3) ≈ f(3, comp, idx, w, m.νs[comp][end], m.νs[comp][end])
                    @test get_vertex(idx, b_bare[comp], a.Γ[comp], 3) ≈ 0.0
                end
            end

            @testset "vectorized" begin
                for comp in eachindex(a.Γ)
                    b_q3   = get_buffers_u(  w,   v,  vp, m)
                    b_q2_2 = get_buffers_u(  w, Inf,  vp, m)
                    b_q2_1 = get_buffers_u(  w,   v, Inf, m)
                    b_q1   = get_buffers_u(  w, Inf, Inf, m)
                    b_bare = get_buffers_u(Inf, Inf, Inf, m)

                    temp .= 0.0; get_vertex_avx!(r, b_q3[comp],   a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(3, comp, j, w, v, vp)                            for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_2[comp], a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(3, comp, j, w, m.νs[comp][end], vp)              for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q2_1[comp], a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(3, comp, j, w, v, m.νs[comp][end])               for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_q1[comp],   a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test temp[:, comp, 1]       ≈ Float64[f(3, comp, j, w, m.νs[comp][end], m.νs[comp][end]) for j in eachindex(r.sites)]
                    temp .= 0.0; get_vertex_avx!(r, b_bare[comp], a.Γ[comp], 3, view(temp, :, comp, 1), false, 1.0); @test norm(temp[:, comp, 1]) ≈ 0.0
                end
            end
        end
    end

    return nothing 
end

"""
    test_action() :: Nothing

Run checks for available action implementations by benchmarking interpolation routines.
"""
function test_action() :: Nothing

    # fix some dummy parameters
    initial = 50.0
    num_σ   = 50 
    num_Ω   = 15
    num_ν   = 10
    p_σ     = 0.3
    p_Ω     = 0.3
    p_ν     = 0.5

    # run tests for action_su2
    @testset "action su2" begin
        # generate action dummy for square lattice Heisenberg model
        l = get_lattice("square", 6, verbose = false)
        r = get_reduced_lattice("heisenberg", [[1.0]], l, verbose = false)
        m = get_mesh("su2", initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
        a = get_action_empty("su2", r, m)
        init_action!(l, r, a)

        # test if bare action is correctly initialized
        @testset "initialization" begin
            @test norm(a.Γ[1].bare) ≈ 0.25
            @test norm(a.Γ[2].bare) ≈ 0.0
        end

        # test interpolations 
        test_interpolations(r, m, a)
    end 

    # run tests for action_u1_dm
    @testset "action u1-dm" begin
        # generate action dummy for triangular lattice dm-c3 model
        l = get_lattice("triangular", 6, verbose = false)
        r = get_reduced_lattice("triangular-dm-c3", [[1.0, 1.0, 1.0]], l, verbose = false)
        m = get_mesh("u1-dm", initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
        a = get_action_empty("u1-dm", r, m)
        init_action!(l, r, a)
    
        # test if bare action is correctly initialized
        @testset "initialization" begin
            @test norm(a.Γ[1].bare) ≈ sqrt(0.125)
            @test norm(a.Γ[2].bare) ≈ sqrt(0.125)
            @test norm(a.Γ[3].bare) ≈ sqrt(0.125)
            @test norm(a.Γ[4].bare) ≈ 0.0
            @test norm(a.Γ[5].bare) ≈ 0.0
            @test norm(a.Γ[6].bare) ≈ 0.0
        end

        # test interpolations 
        test_interpolations(r, m, a)
    end 

    return nothing
end