# Katanin term
function compute_u_kat!( 
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64, 
    vu   :: Float64, 
    vup  :: Float64,  
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_sun,
    da   :: action_sun,
    temp :: Array{Float64, 3}
    )    :: Nothing
    
    # get propagator and prefactors
    p    = get_propagator_kat(Λ, v - 0.5 * u, v + 0.5 * u, m, a, da) + get_propagator_kat(Λ, v + 0.5 * u, v - 0.5 * u, m, a, da)
    pre1 = (a.N^2 - 2.0) / (2.0 * a.N)
    pre2 = (a.N^2 - 1.0) / (4.0 * a.N^2)

    # get buffers for left vertex
    bs1 = get_buffer_sun_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffer_sun_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = get_buffer_sun_u(u, vu, v, m)

    # get buffers for right vertex
    bs2 = get_buffer_sun_s(v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m)
    bt2 = get_buffer_sun_t(-v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m)
    bu2 = get_buffer_sun_u(u, v, vup, m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s = temp[i, 1, 1]; v1d = temp[i, 2, 1]
        v2s = temp[i, 1, 2]; v2d = temp[i, 2, 2]

        # compute contribution at site i 
        Γs = -p * (pre1 * v1s * v2s + v1s * v2d + v1d * v2s)
        Γd = -p * (pre2 * v1s * v2s + v1d * v2d)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end





# left term (right term obtained by symmetries)
function compute_u_left!( 
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64, 
    vu   :: Float64, 
    vup  :: Float64, 
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_sun,
    da   :: action_sun,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors 
    p    = -get_propagator(Λ, v - 0.5 * u, v + 0.5 * u, m, a)
    pre1 = (a.N^2 - 2.0) / (2.0 * a.N)
    pre2 = (a.N^2 - 1.0) / (4.0 * a.N^2)

    # get buffers for left vertex
    bs1 = get_buffer_sun_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffer_sun_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = get_buffer_sun_u(u, vu, v, m)

    # get buffers for right vertex
    bs2 = get_buffer_sun_s(v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m)
    bt2 = get_buffer_sun_t(-v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m)
    bu2 = get_buffer_sun_u(u, v, vup, m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, da, temp, 1, ch_u = false)
    get_Γ_avx!(r, bs2, bt2, bu2,  a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s_st = temp[i, 1, 1]; v1d_st = temp[i, 2, 1]
        v2s    = temp[i, 1, 2]; v2d    = temp[i, 2, 2]

        # compute contribution at site i 
        Γs = -p * (pre1 * v1s_st * v2s + v1s_st * v2d + v1d_st * v2s)
        Γd = -p * (pre2 * v1s_st * v2s + v1d_st * v2d)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end





# central term
function compute_u_central!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64, 
    dv   :: Float64,
    u    :: Float64, 
    vu   :: Float64, 
    vup  :: Float64, 
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_sun,
    da_l :: action_sun,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors 
    p    = -get_propagator(Λ, v - 0.5 * u, v + 0.5 * u, m, a)
    pre1 = (a.N^2 - 2.0) / (2.0 * a.N)
    pre2 = (a.N^2 - 1.0) / (4.0 * a.N^2)

    # get buffers for left vertex
    bs1 = get_buffer_sun_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffer_sun_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = get_buffer_sun_u(u, vu, v, m)

    # get buffers for right vertex
    bs2 = get_buffer_sun_s(v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m)
    bt2 = get_buffer_sun_t(-v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m)
    bu2 = get_buffer_sun_u(u, v, vup, m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1,    a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, da_l, temp, 2, ch_s = false, ch_t = false)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s   = temp[i, 1, 1]; v1d   = temp[i, 2, 1]
        v2s_u = temp[i, 1, 2]; v2d_u = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (pre1 * v1s * v2s_u + v1s * v2d_u + v1d * v2s_u)
        Γd = -p * (pre2 * v1s * v2s_u + v1d * v2d_u)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing 
end
