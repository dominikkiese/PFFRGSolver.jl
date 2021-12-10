# compute self energy derivative
function compute_dΣ!(
    Λ     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da    :: Action,
    Σ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # compute self energy derivative for all frequencies
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            integrand = v -> compute_dΣ_kernel(Λ, m.σ[i], v, r, m, a)
            da.Σ[i]   = quadgk(integrand, -Inf, -2.0 * Λ, 0.0, 2.0 * Λ, Inf, atol = Σ_tol[1], rtol = Σ_tol[2])[1]
        end
    end

    return nothing
end

# compute corrections to self energy derivative
function compute_dΣ_corr!(
    Λ     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da    :: Action,
    da_Σ  :: Action,
    Σ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # compute first correction
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            integrand = v -> compute_dΣ_kernel_corr1(Λ, m.σ[i], v, r, m, a, da_Σ)
            da_Σ.Σ[i] = quadgk(integrand, -Inf, -2.0 * Λ, 0.0, 2.0 * Λ, Inf, atol = Σ_tol[1], rtol = Σ_tol[2])[1]
        end
    end

    # compute second correction and parse to da
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            integrand  = v -> compute_dΣ_kernel_corr2(Λ, m.σ[i], v, r, m, a, da_Σ)
            da.Σ[i]   += da_Σ.Σ[i]
            da.Σ[i]   += quadgk(integrand, -Inf, -2.0 * Λ, 0.0, 2.0 * Λ, Inf, atol = Σ_tol[1], rtol = Σ_tol[2])[1]
        end
    end

    return nothing
end