function launch_parquet!(
    obs_file :: String,
    cp_file  :: String,
    symmetry :: String,
    l        :: lattice,
    r        :: reduced_lattice,
    m        :: mesh,
    a        :: action,
    Λ        :: Float64,
    dΛ       :: Float64,
    β        :: Float64, 
    max_iter :: Int64, 
    eval     :: Int64
    ;
    S        :: Float64 = 0.5,
    N        :: Float64 = 2.0
    )        :: Nothing

    # init output and error buffer
    ap   = get_action_empty(symmetry, r, m, S = S, N = N)
    aerr = get_action_empty(symmetry, r, m, S = S, N = N)

    # init buffers for evaluation of rhs
    num_comps = length(a.Γ)
    num_sites = length(r.sites)
    tbuffs    = NTuple{3, Matrix{Float64}}[(zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites), zeros(Float64, num_comps, num_sites)) for i in 1 : Threads.nthreads()]
    temps     = Array{Float64, 3}[zeros(Float64, num_sites, num_comps, 2) for i in 1 : Threads.nthreads()]

    # init errors and iteration count
    abs_err = Inf
    rel_err = Inf
    count   = 0

    # compute fixed point
    while abs_err >= 1e-8 && rel_err >= 1e-5 && count <= max_iter
        println("After iteration $(count), abs_err, rel_err = $(abs_err), $(rel_err).")

        # compute SDE and BSEs
        compute_Σ!(Λ, r, m, a, ap)
        compute_Γ!(Λ, r, m, a, ap, tbuffs, temps, eval)

        # compute the errors 
        replace_with!(aerr, ap)
        mult_with_add_to!(a, -1.0, aerr)
        abs_err = get_abs_max(aerr)
        rel_err = abs_err / max(get_abs_max(a), get_abs_max(ap))

        # update current solution using damping factor β
        mult_with!(a, 1 - β)
        mult_with_add_to!(ap, β, a)

        # increment iteration count
        count += 1
    end

    if count <= max_iter
        println()
        println("Converged to fixed point, final abs_err, rel_err = $(abs_err), $(rel_err).")
    else 
        println()
        println("Maximum number of iterations reached, final abs_err, rel_err = $(abs_err), $(rel_err).")
    end

    # save final result 
    measure(symmetry, obs_file, cp_file, Λ, dΛ, Dates.now(), Dates.now(), r, m, a, Inf, 0.0)

    return nothing
end

