# generate correlation dummy for su2 symmetry
function get_χ_su2_empty(
    r :: Reduced_lattice,
    m :: Mesh
    ) :: Vector{Matrix{Float64}}

    χs = zeros(Float64, length(r.sites), m.num_χ)
    χ  = Matrix{Float64}[χs]

    return χ 
end

# kernel for double integral
function compute_χ_kernel!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    maps :: NTuple{2, Bool},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v  = vv[1, i]; dv  = 1.0
        vp = vv[2, i]; dvp = 1.0 

        if maps[1] 
            v  = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
            dv = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        end 

        if maps[2] 
            vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
            dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])
        end

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(1, site, bs1, bt1, bu1, r, a, apply_flags_su2) / (2.0 * pi)^2

        if site == 1
            vs, vd   = get_Γ(site, bs2, bt2, bu2, r, a)
            buff[i] -= (2.0 * a.S) * (vs - vd) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# compute isotropic correlation in real and Matsubara space
function compute_χ!(
    Λ     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action_su2,
    χ     :: Vector{Matrix{Float64}},
    χ_tol :: NTuple{2, Float64}
    )     :: Nothing

    @sync for w in 1 : m.num_χ
        for i in eachindex(r.sites)
            Threads.@spawn begin
                # compute vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[1][i, w] = hcubature_v(integrand, Float64[0.0, 0.0], Float64[1.0, 1.0], abstol = χ_tol[1], reltol = χ_tol[2], maxevals = 10^8)[1]

                # compute propagator contribution
                if i == 1
                    integrand   = v -> (2.0 * a.S) * get_G(Λ, v - 0.5 * m.χ[w], m, a) * get_G(Λ, v + 0.5 * m.χ[w], m, a) / (4.0 * pi)
                    ref         = Λ + 0.5 * abs(m.χ[w])
                    χ[1][i, w] += quadgk(integrand, -Inf, -2.0 * ref, 0.0, 2.0 * ref, Inf, atol = χ_tol[1], rtol = χ_tol[2], order = 10, maxevals = 10^8)[1]
                end
            end
        end
    end

    return nothing
end