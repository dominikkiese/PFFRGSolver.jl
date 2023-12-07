function launch_parquet!(
    obs_file    :: String,
    cp_file     :: String,
    symmetry    :: String,
    r           :: Reduced_lattice,
    m           :: Mesh,
    a           :: Action,
    Λ           :: Float64,
    dΛ          :: Float64,
    max_iter    :: Int64,
    min_eval    :: Int64,
    max_eval    :: Int64,
    Σ_tol       :: NTuple{2, Float64},
    Γ_tol       :: NTuple{2, Float64},
    χ_tol       :: NTuple{2, Float64},
    parquet_tol :: NTuple{2, Float64}
    ;
    S           :: Float64 = 0.5
    )           :: Nothing

    # init output and error buffers
    ap    = get_action_empty(symmetry, r, m, S = S)
    a_err = get_action_empty(symmetry, r, m, S = S)

    # init buffers for evaluation of rhs
    num_comps = length(a.Γ)
    num_sites = length(r.sites)
    tbuffs    = NTuple{3, Matrix{Float64}}[(zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites)) for i in 1 : Threads.nthreads()]
    temps     = Array{Float64, 3}[zeros(Float64, num_sites, num_comps, 2) for i in 1 : Threads.nthreads()]
    corrs     = zeros(Float64, 2, 3, m.num_Ω)

    # init errors and iteration count for BSEs
    abs_err = Inf
    rel_err = Inf
    count   = 1

    # init damping parameter
    β = min(Λ, 1.0)

    # set eval for integration
    eval = min(max(ceil(Int64, min_eval / sqrt(Λ)), min_eval), max_eval)

    while abs_err >= parquet_tol[1] && rel_err >= parquet_tol[2] && count <= max_iter
        println()
        println("Parquet iteration $count ...")
        println("   Converging SDE ...")

        # init errors and iteration count for SDE
        Σ_abs_err = Inf
        Σ_rel_err = Inf
        Σ_count   = 1

        while Σ_abs_err >= parquet_tol[1] && Σ_rel_err >= parquet_tol[2] && Σ_count <= max_iter
            compute_Σ!(Λ, r, m, a, ap, tbuffs, temps, corrs, eval, Γ_tol, Σ_tol)

            # compute errors
            Σ_abs_err = maximum(abs.(ap.Σ .- a.Σ))
            Σ_rel_err = Σ_abs_err / max(maximum(abs.(a.Σ)), maximum(abs.(ap.Σ)))

            # update solution
            @turbo a.Σ .= (1.0 - β) .* a.Σ .+ β .* ap.Σ

            # increment iteration count
            Σ_count += 1
        end

        println("   Evaluating BSEs ...")
        compute_Γ!(Λ, r, m, a, ap, tbuffs, temps, corrs, eval, Γ_tol)

        # compute errors
        replace_with_Γ!(a_err, ap)
        mult_with_add_to_Γ!(a, -1.0, a_err)
        abs_err = get_abs_max(a_err)
        rel_err = abs_err / max(get_abs_max(a), get_abs_max(ap))
        println("Done. Relative error err = $(rel_err).")
        flush(stdout)

        # update solution
        mult_with_Γ!(a, 1.0 - β)
        mult_with_add_to_Γ!(ap, β, a)

        # increment iteration count
        count += 1
    end

    if count <= max_iter
        println()
        println("BSEs converged to fixed point, terminating ...")
    else
        println()
        println("Maximum number of iterations reached, terminating ...")
    end
    
    flush(stdout)

    # save final result
    χ = get_χ_empty(symmetry, r, m)
    measure(symmetry, obs_file, cp_file, Λ, dΛ, χ, χ_tol, Dates.now(), Dates.now(), r, m, a, Inf, 0.0)

    return nothing
end