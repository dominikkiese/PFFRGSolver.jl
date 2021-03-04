function launch_ml!(
    obs_file :: String,
    cp_file  :: String,
    symmetry :: String,
    l        :: lattice,
    r        :: reduced_lattice,
    m        :: mesh,
    a        :: action,
    loops    :: Int64,
    Λi       :: Float64,
    Λf       :: Float64,
    dΛi      :: Float64,
    bmin     :: Float64,
    bmax     :: Float64,
    eval     :: Int64,
    wt       :: Float64,
    ct       :: Float64
    ;
    S        :: Float64 = 0.5,
    N        :: Float64 = 2.0
    )        :: Nothing

    # init timers for checkpointing
    t0 = Dates.now()
    t  = Dates.now()

    # init ODE solver buffers
    da      = get_action_empty(symmetry, r, m, S = S, N = N)
    a_stage = get_action_empty(symmetry, r, m, S = S, N = N)
    a_inter = get_action_empty(symmetry, r, m, S = S, N = N)
    a_err   = get_action_empty(symmetry, r, m, S = S, N = N)
    init_action!(l, r, a_inter)

    # init left (right part by symmetry) and central part, full loop contribution and self energy corrections
    da_l    = get_action_empty(symmetry, r, m, S = S, N = N)
    da_c    = get_action_empty(symmetry, r, m, S = S, N = N)
    da_temp = get_action_empty(symmetry, r, m, S = S, N = N)
    da_Σ    = get_action_empty(symmetry, r, m, S = S, N = N)

    # init buffers for evaluation of rhs
    num_comps = length(a.Γ)
    num_sites = length(r.sites)
    tbuffs    = NTuple{3, Matrix{Float64}}[(zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites)) for i in 1 : Threads.nthreads()]
    temps     = Array{Float64, 3}[zeros(Float64, num_sites, num_comps, 4) for i in 1 : Threads.nthreads()]

    # init cutoff, step size and energy scale
    Λ  = Λi
    dΛ = dΛi
    Z  = maximum(Float64[norm(a.Γ[i].bare) for i in 1 : num_comps])

    # compute renormalization group flow
    while Λ > Λf
        println("Current cutoff Λ = $(Λ)")

        # prepare da and a_err 
        replace_with!(da, a)
        replace_with!(a_err, a)

        # compute k1 and parse to da and a_err
        compute_dΣ!(Λ, r, m, a, a_stage)
        compute_dΓ_ml!(Λ, r, m, loops, a, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval)
        compute_dΣ_corr!(Λ, r, m, a, a_stage, da_Σ)
        mult_with_add_to!(a_stage, -2.0 * dΛ / 9.0, da)
        mult_with_add_to!(a_stage, -7.0 * dΛ / 24.0, a_err)

        # compute k2 and parse to da and a_err
        replace_with!(a_inter, a)
        mult_with_add_to!(a_stage, -0.5 * dΛ, a_inter)
        compute_dΣ!(Λ - 0.5 * dΛ, r, m, a_inter, a_stage)
        compute_dΓ_ml!(Λ - 0.5 * dΛ, r, m, loops, a_inter, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval) 
        compute_dΣ_corr!(Λ - 0.5 * dΛ, r, m, a_inter, a_stage, da_Σ)
        mult_with_add_to!(a_stage, -1.0 * dΛ / 3.0, da)
        mult_with_add_to!(a_stage, -1.0 * dΛ / 4.0, a_err)

        # compute k3 and parse to da and a_err
        replace_with!(a_inter, a)
        mult_with_add_to!(a_stage, -0.75 * dΛ, a_inter)
        compute_dΣ!(Λ - 0.75 * dΛ, r, m, a_inter, a_stage)
        compute_dΓ_ml!(Λ - 0.75 * dΛ, r, m, loops, a_inter, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval)
        compute_dΣ_corr!(Λ - 0.75 * dΛ, r, m, a_inter, a_stage, da_Σ)
        mult_with_add_to!(a_stage, -4.0 * dΛ / 9.0, da)
        mult_with_add_to!(a_stage, -1.0 * dΛ / 3.0, a_err)

        # compute k4 and parse to a_err
        replace_with!(a_inter, da)
        compute_dΣ!(Λ - dΛ, r, m, a_inter, a_stage)
        compute_dΓ_ml!(Λ - dΛ, r, m, loops, a_inter, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval)
        compute_dΣ_corr!(Λ - dΛ, r, m, a_inter, a_stage, da_Σ)
        mult_with_add_to!(a_stage, -1.0 * dΛ / 8.0, a_err)

        # estimate integration error 
        subtract_from!(a_inter, a_err)
        Δ     = get_abs_max(a_err)
        scale = 1e-10 + max(get_abs_max(a_inter), get_abs_max(a)) * 1e-2
        err   = Δ / scale

        if err <= 1.0 || dΛ == bmin * Λ
            # update cutoff and step size
            Λ  -= dΛ
            dΛ  = max(bmin * Λ, min(bmax * Λ, 0.9 * sqrt(1.0 / err) * dΛ))
            dΛ  = min(dΛ, Λ - Λf)

            # check for divergence
            if get_abs_max(a_inter) > 100.0 * Z
                println("Vertex has diverged, terminating solver ...")
                break 
            end

            # update frequency mesh
            m = resample_from_to(Λ, Z, m, a_inter, a)

            # do measurements and checkpointing 
            t = measure(symmetry, obs_file, cp_file, Λ, dΛ, t, t0, r, m, a, wt, ct)
        else
            # update step size
            dΛ = max(bmin * Λ, min(bmax * Λ, 0.9 * sqrt(1.0 / err) * dΛ))
        end
    end

    # save final result
    m = resample_from_to(Λ, m, a_inter, a)
    t = measure(symmetry, obs_file, cp_file, Λ, dΛ, t, t0, r, m, a, Inf, 0.0)

    # open files 
    obs = h5open(obs_file, "cw")
    cp  = h5open(cp_file, "cw")

    # set finished flags
    obs["finished"] = true
    cp["finished"]  = true

    # close files 
    close(obs)
    close(cp)

    println("Renormalization group flow finished.")

    return nothing 
end