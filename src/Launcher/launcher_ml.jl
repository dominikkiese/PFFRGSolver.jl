function launch_ml!(
    obs_file :: String,
    cp_file  :: String,
    symmetry :: String,
    l        :: Lattice,
    r        :: Reduced_lattice,
    m        :: Mesh,
    a        :: Action,
    p_σ      :: NTuple{4, Float64},
    p_Ω      :: NTuple{8, Float64},
    p_ν      :: NTuple{8, Float64},
    loops    :: Int64,
    Σ_corr   :: Bool,
    Λi       :: Float64,
    Λf       :: Float64,
    dΛi      :: Float64,
    bmin     :: Float64,
    bmax     :: Float64,
    eval     :: Int64,
    Σ_tol    :: NTuple{2, Float64},
    Γ_tol    :: NTuple{2, Float64},
    χ_tol    :: NTuple{2, Float64},
    ODE_tol  :: NTuple{2, Float64},
    t        :: DateTime,
    t0       :: DateTime,
    wt       :: Float64,
    ct       :: Float64
    ;
    S        :: Float64 = 0.5
    )        :: Nothing

    # init ODE solver buffers
    da      = get_action_empty(symmetry, r, m, S = S)
    a_stage = get_action_empty(symmetry, r, m, S = S)
    a_inter = get_action_empty(symmetry, r, m, S = S)
    a_err   = get_action_empty(symmetry, r, m, S = S)
    init_action!(l, r, a_inter)

    # init left (right part by symmetry) and central part, full loop contribution and self energy corrections
    da_l    = get_action_empty(symmetry, r, m, S = S)
    da_c    = get_action_empty(symmetry, r, m, S = S)
    da_temp = get_action_empty(symmetry, r, m, S = S)
    da_Σ    = get_action_empty(symmetry, r, m, S = S)

    # init buffers for evaluation of rhs
    num_comps = length(a.Γ)
    num_sites = length(r.sites)
    tbuffs    = NTuple{3, Vector{Float64}}[(zeros(Float64, num_sites), zeros(Float64, num_sites), zeros(Float64, num_sites)) for i in 1 : Threads.nthreads()]
    temps     = Array{Float64, 3}[zeros(Float64, num_sites, num_comps, 4) for i in 1 : Threads.nthreads()]

    # init cutoff and step size
    Λ  = Λi
    dΛ = dΛi

    # compute renormalization group flow
    while Λ > Λf
        println()
        println("ODE step at cutoff Λ / |J| = $(Λ) ...")
        flush(stdout)

        # prepare da and a_err
        replace_with!(da, a)
        replace_with!(a_err, a)

        # compute k1 and parse to da and a_err
        compute_dΣ!(Λ, r, m, a, a_stage, Σ_tol)
        compute_dΓ_ml!(Λ, r, m, loops, a, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval, Γ_tol)
        if Σ_corr compute_dΣ_corr!(Λ, r, m, a, a_stage, da_Σ, Σ_tol) end
        mult_with_add_to!(a_stage, -2.0 * dΛ / 9.0, da)
        mult_with_add_to!(a_stage, -7.0 * dΛ / 24.0, a_err)

        # compute k2 and parse to da and a_err
        replace_with!(a_inter, a)
        mult_with_add_to!(a_stage, -0.5 * dΛ, a_inter)
        compute_dΣ!(Λ - 0.5 * dΛ, r, m, a_inter, a_stage, Σ_tol)
        compute_dΓ_ml!(Λ - 0.5 * dΛ, r, m, loops, a_inter, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval, Γ_tol)
        if Σ_corr compute_dΣ_corr!(Λ - 0.5 * dΛ, r, m, a_inter, a_stage, da_Σ, Σ_tol) end
        mult_with_add_to!(a_stage, -1.0 * dΛ / 3.0, da)
        mult_with_add_to!(a_stage, -1.0 * dΛ / 4.0, a_err)

        # compute k3 and parse to da and a_err
        replace_with!(a_inter, a)
        mult_with_add_to!(a_stage, -0.75 * dΛ, a_inter)
        compute_dΣ!(Λ - 0.75 * dΛ, r, m, a_inter, a_stage, Σ_tol)
        compute_dΓ_ml!(Λ - 0.75 * dΛ, r, m, loops, a_inter, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval, Γ_tol)
        if Σ_corr compute_dΣ_corr!(Λ - 0.75 * dΛ, r, m, a_inter, a_stage, da_Σ, Σ_tol) end
        mult_with_add_to!(a_stage, -4.0 * dΛ / 9.0, da)
        mult_with_add_to!(a_stage, -1.0 * dΛ / 3.0, a_err)

        # compute k4 and parse to a_err
        replace_with!(a_inter, da)
        compute_dΣ!(Λ - dΛ, r, m, a_inter, a_stage, Σ_tol)
        compute_dΓ_ml!(Λ - dΛ, r, m, loops, a_inter, a_stage, da_l, da_c, da_temp, da_Σ, tbuffs, temps, eval, Γ_tol)
        if Σ_corr compute_dΣ_corr!(Λ - dΛ, r, m, a_inter, a_stage, da_Σ, Σ_tol) end
        mult_with_add_to!(a_stage, -1.0 * dΛ / 8.0, a_err)

        # estimate integration error
        subtract_from!(a_inter, a_err)
        Δ     = get_abs_max(a_err)
        scale = ODE_tol[1] + max(get_abs_max(a_inter), get_abs_max(a)) * ODE_tol[2]
        err   = Δ / scale

        println("Done. Relative integration error err = $(err).")
        println("Performing sanity checks and measurements ...")

        # terminate if integration becomes unfeasible
        if err >= 10.0
            println("Relative integration error has become too large, terminating solver ...")
            break
        end

        if err <= 1.0 || dΛ <= bmin
            # update cutoff
            Λ -= dΛ

            # update step size
            dΛp = dΛ
            dΛ  = max(bmin, min(bmax * Λ, 0.85 * (1.0 / err)^(1.0 / 3.0) * dΛ))
            dΛ  = min(dΛ, 1.1 * dΛp, Λ - Λf)

            # terminate if vertex diverges
            if get_abs_max(a_inter) > max(min(50.0 / Λ, 1000), 10.0)
                println("Vertex has diverged, terminating solver ...")
                break
            end

            # update frequency mesh
            m = resample_from_to(Λ, p_σ, p_Ω, p_ν, m, a_inter, a)

            # do measurements and checkpointing
            t, monotone = measure(symmetry, obs_file, cp_file, Λ, dΛ, χ_tol, t, t0, r, m, a, wt, ct)

            # terminate if correlations show non-monotonicity
            if monotone == false
                println("Flowing correlations show non-monotonicity, terminating solver ...")
                break
            end

            if Λ > Λf
                println("Done. Proceeding to next ODE step.")
            end
        else
            # update step size
            dΛ = max(bmin, min(bmax * Λ, 0.85 * (1.0 / err)^(1.0 / 3.0) * dΛ))
            dΛ = min(dΛ, Λ - Λf)

            println("Done. Repeating ODE step with smaller dΛ.")
        end
    end

    # save final result
    m = resample_from_to(Λ, p_σ, p_Ω, p_ν, m, a_inter, a)
    t = measure(symmetry, obs_file, cp_file, Λ, dΛ, χ_tol, t, t0, r, m, a, Inf, 0.0)

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
    flush(stdout)

    return nothing
end