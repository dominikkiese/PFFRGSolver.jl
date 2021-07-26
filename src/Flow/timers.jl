"""
    get_flow_timers() :: Nothing

Time current implementation of flow equations by running the different integration kernels.
"""
function get_flow_timers() :: Nothing

    # init test dummys
    list = get_mesh(rand(), 1.0, 30, 0.4)
    m    = Mesh(31, 31, 31, 31, 31, list, list, list, list, list, list, list)
    Λ    = rand()
    v    = rand()
    dv   = rand()
    wc   = rand()
    vc   = rand()
    vcp  = rand()

    # init timer
    to = TimerOutput()

    # time evals of integration kernels for action_su2
    @timeit to "=> action_su2" begin 
        # generate action dummy for hyperkagome lattice Heisenberg model
        l    = get_lattice("hyperkagome", 6, verbose = false)
        r    = get_reduced_lattice("heisenberg", [[1.0]], l, verbose = false)
        a    = get_action_empty("su2", r, m); init_action!(l, r, a)
        ap   = get_action_empty("su2", r, m)
        buff = zeros(Float64, 2, length(r.sites))
        temp = zeros(Float64, length(r.sites), 2, 2)

        # fill self energy with random values
        a.Σ  .= rand(31)
        ap.Σ .= rand(31)

        # fill channels with random values
        for i in eachindex(a.Γ)
            a.Γ[i].ch_s.q3 .= rand(length(r.sites), 31, 31, 31)
            a.Γ[i].ch_t.q3 .= rand(length(r.sites), 31, 31, 31)
            a.Γ[i].ch_u.q3 .= rand(length(r.sites), 31, 31, 31)

            ap.Γ[i].ch_s.q3 .= rand(length(r.sites), 31, 31, 31)
            ap.Γ[i].ch_t.q3 .= rand(length(r.sites), 31, 31, 31)
            ap.Γ[i].ch_u.q3 .= rand(length(r.sites), 31, 31, 31)
        end

        # set asymptotic limits
        limits!(a)
        limits!(ap)

        for rep in 1 : 5
            # time parquet equations
            @timeit to "=> parquet" begin
                # time SDE
                @timeit to "-> SDE" compute_Σ_kernel(Λ, v, wc, r, m, a, (1e-5, 1e-3))

                # time BSE in all channels
                @timeit to "-> BSE s channel" compute_s_BSE!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, temp)
                @timeit to "-> BSE t channel" compute_t_BSE!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, temp)
                @timeit to "-> BSE u channel" compute_u_BSE!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, temp)
            end

            # time flow equations
            @timeit to "=> FRG" begin
                # time self energy
                @timeit to "-> Σ one loop" compute_dΣ_kernel(Λ, v, wc, r, m, a)
                @timeit to "-> Σ corr 1"   compute_dΣ_kernel_corr1(Λ, v, wc, r, m, a, ap)
                @timeit to "-> Σ corr 2"   compute_dΣ_kernel_corr2(Λ, v, wc, r, m, a, ap)

                # time Katanin part in all channels
                @timeit to "-> Katanin s channel" compute_s_kat!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> Katanin t channel" compute_t_kat!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> Katanin u channel" compute_u_kat!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)

                # time left part in all channels
                @timeit to "-> left s channel" compute_s_left!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> left t channel" compute_t_left!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> left u channel" compute_u_left!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)

                # time central part in all channels
                @timeit to "-> central s channel" compute_s_central!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> central t channel" compute_t_central!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> central u channel" compute_u_central!(Λ, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
            end
        end
    end

    show(to)

    return nothing
end
