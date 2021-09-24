# time integration kernels
function time_rhs!(
    to      :: TimerOutput,
    label   :: String,
    initial :: Float64, 
    buff    :: Vector{Float64},
    r       :: Reduced_lattice,
    m       :: Mesh,
    a       :: Action,
    ap      :: Action,
    temp    :: Array{Float64, 3}
    )       :: Nothing 

    v   = rand() * initial
    wc  = rand() * initial
    vc  = rand() * initial
    vcp = rand() * initial
    dv  = rand()

    # fill self energy with random values and ensure antisymmetry
    a.Σ    .= rand(Float64, m.num_σ)
    a.Σ[1]  = 0.0

    # fill action with random values
    for comp in eachindex(a.Γ)
        for site in eachindex(r.sites)
            for iw in 1 : m.num_Ω
                for iv in 1 : m.num_ν
                    for ivp in 1 : m.num_ν
                        a.Γ[comp].ch_s.q3[site, iw, iv, ivp] = rand()
                        a.Γ[comp].ch_t.q3[site, iw, iv, ivp] = rand()
                        a.Γ[comp].ch_u.q3[site, iw, iv, ivp] = rand()
                    end
                end
            end
        end
    end

    @timeit to "=> $(label)" begin 
        @timeit to "=> parquet" begin 
            for rep in 1 : 100
                @timeit to "-> SDE" compute_Σ_kernel(initial, v, wc, r, m, a, (1e-5, 1e-3))

                @timeit to "-> BSE s channel" compute_s_BSE!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, temp)
                @timeit to "-> BSE t channel" compute_t_BSE!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, temp)
                @timeit to "-> BSE u channel" compute_u_BSE!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, temp)
            end 
        end 

        @timeit to "=> FRG" begin 
            for rep in 1 : 100
                @timeit to "-> Σ one loop" compute_dΣ_kernel(initial, v, wc, r, m, a)
                @timeit to "-> Σ corr 1"   compute_dΣ_kernel_corr1(initial, v, wc, r, m, a, ap)
                @timeit to "-> Σ corr 2"   compute_dΣ_kernel_corr2(initial, v, wc, r, m, a, ap)

                @timeit to "-> Katanin s channel" compute_s_kat!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> Katanin t channel" compute_t_kat!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> Katanin u channel" compute_u_kat!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)

                @timeit to "-> left s channel" compute_s_left!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> left t channel" compute_t_left!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> left u channel" compute_u_left!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)

                @timeit to "-> central s channel" compute_s_central!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> central t channel" compute_t_central!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
                @timeit to "-> central u channel" compute_u_central!(initial, 1, buff, v, dv, wc, vc, vcp, r, m, a, ap, temp)
            end 
        end 
    end

    return nothing 
end

"""
    get_flow_timers() :: Nothing

Time current implementation of integration kernels.
"""
function get_flow_timers() :: Nothing

    # fix some dummy parameters
    initial = 50.0
    num_σ   = 50 
    num_Ω   = 15
    num_ν   = 10
    p_σ     = 0.3
    p_Ω     = 0.3
    p_ν     = 0.5

    # init dummys for Action_su2
    l    = get_lattice("square", 6, verbose = false)
    r    = get_reduced_lattice("heisenberg", [[1.0]], l, verbose = false)
    m    = get_mesh("su2", initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    a    = get_action_empty("su2", r, m); init_action!(l, r, a)
    ap   = get_action_empty("su2", r, m)
    buff = zeros(Float64, length(r.sites))
    temp = zeros(Float64, length(r.sites), 2, 2)

    # time Action_su2 kernels
    to = TimerOutput()
    time_rhs!(to, "su2", initial, buff, r, m, a, ap, temp)
    show(to)
    println()

    # init dummys for Action_u1_dm
    l    = get_lattice("triangular", 6, verbose = false)
    r    = get_reduced_lattice("triangular-dm-c3", [[1.0, 1.0, 1.0]], l, verbose = false)
    m    = get_mesh("u1-dm", initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    a    = get_action_empty("u1-dm", r, m); init_action!(l, r, a)
    ap   = get_action_empty("u1-dm", r, m)
    buff = zeros(Float64, length(r.sites))
    temp = zeros(Float64, length(r.sites), 6, 2)

    # time Action_u1_dm kernels
    to = TimerOutput()
    time_rhs!(to, "u1-dm", initial, buff, r, m, a, ap, temp)
    show(to)

    return nothing
end