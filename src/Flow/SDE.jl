# compute self energy from SDE
function compute_Σ!(
    Λ     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a1    :: Action,
    a2    :: Action,
    Σ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # compute self energy for all frequencies
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            # compute integral
            integrand = v -> compute_Σ_kernel(Λ, v, m.σ[i], r, m, a1, Σ_tol)
            a2.Σ[i]   = quadgk(integrand, -Inf, -2.0 * Λ, 0.0, 2.0 * Λ, Inf, atol = Σ_tol[1], rtol = Σ_tol[2], order = 10)[1]
        end
    end

    return nothing
end