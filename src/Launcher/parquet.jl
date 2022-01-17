function iterate_parquet!(
    r           :: Reduced_lattice,
    m           :: Mesh,
    a           :: Action,
    ap          :: Action,
    app         :: Action,
    a_err       :: Action,
    tbuffs      :: Vector{NTuple{3, Matrix{Float64}}},
    temps       :: Vector{Array{Float64, 3}},
    corrs       :: Array{Float64, 3},
    Λ           :: Float64,
    max_iter    :: Int64,
    eval        :: Int64,
    Σ_tol       :: NTuple{2, Float64},
    Γ_tol       :: NTuple{2, Float64},
    parquet_tol :: NTuple{2, Float64}
    )           :: Nothing

    # init global errors and global iteration count
    abs_err = Inf
    rel_err = Inf
    count   = 1

    # init control parameters for damping
    K_Σ = 1.0
    K_Γ = 1.0

    # iterate parquet equations until convergence or maximum number of iterations is reached
    while abs_err >= parquet_tol[1] && rel_err >= parquet_tol[2] && count <= max_iter
        println()
        println("Parquet iteration $count ...")

        # buffer current solution for global error calculation
        replace_with!(app, a)



        # converge SDE
        println("   Iterating SDE ...")

        # init errors and iteration count
        Σ_abs_err = Inf
        Σ_rel_err = Inf
        Σ_count   = 1

        # iterate until convergence or maximum number of iterations is reached
        while Σ_abs_err >= parquet_tol[1] && Σ_rel_err >= parquet_tol[2] && Σ_count <= max_iter
            println("       SDE iteration $Σ_count ...")

            # evaluate BSEs
            compute_Σ!(Λ, r, m, a, ap, tbuffs, temps, corrs, eval, Γ_tol, Σ_tol)

            # compute errors
            Σp_abs_err = maximum(abs.(ap.Σ .- a.Σ))
            Σp_rel_err = Σp_abs_err / max(maximum(abs.(a.Σ)), maximum(abs.(ap.Σ)))

            # check if errors have decreased, otherwise terminate
            if Σp_abs_err > Σ_abs_err || Σp_rel_err > Σ_rel_err
                println("       Errors did not decrease, proceeding ...")
                K_Σ *= 10.0
                @goto start_BSE
            end

            # update errors 
            Σ_abs_err = Σp_abs_err
            Σ_rel_err = Σp_rel_err

            println("       Done. Relative error err = $(Σ_rel_err).")
            flush(stdout)

            # update current solution using damping factor β
            β    = 1.0 / (1.0 + K_Σ * Σ_rel_err)
            K_Σ *= 0.8
            @turbo a.Σ .= (1.0 - β) .* a.Σ .+ β .* ap.Σ

            # increment iteration count
            Σ_count += 1
        end

        # print result
        if Σ_count <= max_iter
            println("   Converged to fixed point, proceeding ...")
        else
            println("   Maximum number of iterations reached, proceeding ...")
        end



        # converge BSEs
        @label start_BSE
        println()
        println("   Iterating BSEs ...")

        # init errors and iteration count
        Γ_abs_err = Inf
        Γ_rel_err = Inf
        Γ_count   = 1
        
        # iterate BSEs until convergence or maximum number of iterations is reached
        while Γ_abs_err >= parquet_tol[1] && Γ_rel_err >= parquet_tol[2] && Γ_count <= max_iter
            println("       BSEs iteration $Γ_count ...")

            # evaluate BSEs
            compute_Γ!(Λ, r, m, a, ap, tbuffs, temps, corrs, eval, Γ_tol)

            # compute errors
            replace_with_Γ!(a_err, ap)
            mult_with_add_to_Γ!(a, -1.0, a_err)
            Γp_abs_err = get_abs_max(a_err)
            Γp_rel_err = Γp_abs_err / max(get_abs_max(a), get_abs_max(ap))

            # check if errors have decreased, otherwise terminate
            if Γp_abs_err > Γ_abs_err || Γp_rel_err > Γ_rel_err
                println("       Errors did not decrease, proceeding ...")
                K_Γ *= 10.0
                @goto compute_global_errors
            end

            # update errors 
            Γ_abs_err = Γp_abs_err
            Γ_rel_err = Γp_rel_err

            println("       Done. Relative error err = $(Γ_rel_err).")
            flush(stdout)

            # update current solution using damping factor β
            β    = 1.0 / (1.0 + K_Γ * Γ_rel_err)
            K_Γ *= 0.8
            mult_with_Γ!(a, 1.0 - β)
            mult_with_add_to_Γ!(ap, β, a)

            # increment iteration count
            Γ_count += 1
        end

        # print result
        if Γ_count <= max_iter
            println("   Converged to fixed point, proceeding ...")
        else
            println("   Maximum number of iterations reached, proceeding ...")
        end



        # compute global errors
        @label compute_global_errors
        replace_with!(a_err, a)
        mult_with_add_to!(app, -1.0, a_err)
        Σ_diff  = maximum(abs.(a_err.Σ))
        Γ_diff  = get_abs_max(a_err)
        abs_err = max(Σ_diff, Γ_diff)
        rel_err = max(Σ_diff / max(maximum(abs.(a.Σ)), maximum(abs.(app.Σ))), Γ_diff / max(get_abs_max(a), get_abs_max(app)))

        println("Done. Relative error err = $(rel_err).")
        flush(stdout)

        # increment global iteration count
        count += 1
    end

    if count <= max_iter
        println()
        println("Converged to fixed point, terminating parquet iterations ...")
    else
        println()
        println("Maximum number of iterations reached, terminating parquet iterations ...")
    end
    
    flush(stdout)

    return nothing
end

function launch_parquet!(
    obs_file    :: String,
    cp_file     :: String,
    symmetry    :: String,
    l           :: Lattice,
    r           :: Reduced_lattice,
    m           :: Mesh,
    a           :: Action,
    Λ           :: Float64,
    dΛ          :: Float64,
    max_iter    :: Int64,
    eval        :: Int64,
    Σ_tol       :: NTuple{2, Float64},
    Γ_tol       :: NTuple{2, Float64},
    χ_tol       :: NTuple{2, Float64},
    parquet_tol :: NTuple{2, Float64}
    ;
    S           :: Float64 = 0.5
    )           :: Nothing

    # init output and error buffers
    ap    = get_action_empty(symmetry, r, m, S = S)
    app   = get_action_empty(symmetry, r, m, S = S)
    a_err = get_action_empty(symmetry, r, m, S = S)

    # init buffers for evaluation of rhs
    num_comps = length(a.Γ)
    num_sites = length(r.sites)
    tbuffs    = NTuple{3, Matrix{Float64}}[(zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites)) for i in 1 : Threads.nthreads()]
    temps     = Array{Float64, 3}[zeros(Float64, num_sites, num_comps, 2) for i in 1 : Threads.nthreads()]
    corrs     = zeros(Float64, 2, 3, m.num_Ω)

    # run parquet kernel 
    iterate_parquet!(r, m, a, ap, app, a_err, tbuffs, temps, corrs, Λ, max_iter, eval, Σ_tol, Γ_tol, parquet_tol)

    # save final result
    χ = get_χ_empty(symmetry, r, m)
    measure(symmetry, obs_file, cp_file, Λ, dΛ, χ, χ_tol, Dates.now(), Dates.now(), r, m, a, Inf, 0.0)

    return nothing
end