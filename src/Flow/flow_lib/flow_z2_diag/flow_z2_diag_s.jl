# Katanin kernel
function compute_s_kat!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_z2_diag,
    da   :: Action_z2_diag,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = get_propagator_kat(Λ, v + 0.5 * s, 0.5 * s - v, m, a, da) + get_propagator_kat(Λ, 0.5 * s - v, v + 0.5 * s, m, a, da)

    # get buffers for left vertex
    bs1 = get_buffer_s(s, vs, -v, m)
    bt1 = get_buffer_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffer_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffer_s(s, v, vsp, m)
    bt2 = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffer_u( v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    @turbo unroll = 1 for i in eachindex(r.sites)
        # read cached values for site i
        v1xx = temp[i, 1, 1]
        v1yy = temp[i, 2, 1]
        v1zz = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]

        v2xx = temp[i, 1, 2]
        v2yy = temp[i, 2, 2]
        v2zz = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]

        # compute contribution at site i
        Γxx = -p * (v1dd * v2xx + v1xx * v2dd - v1yy * v2zz - v1zz * v2yy)  
        Γyy = -p * (v1dd * v2yy - v1xx * v2zz + v1yy * v2dd - v1zz * v2xx)  
        Γzz = -p * (v1zz * v2dd + v1dd * v2zz - v1xx * v2yy - v1yy * v2xx)  
        Γdd = -p * (v1xx * v2xx + v1dd * v2dd + v1yy * v2yy + v1zz * v2zz)  

        # parse result to output buffer
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γyy
        buff[3, i] += dv * Γzz
        buff[4, i] += dv * Γdd
    end

    return nothing
end





# left kernel (right part obtained by symmetries)
function compute_s_left!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_z2_diag,
    da   :: Action_z2_diag,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for left vertex
    bs1 = get_buffer_empty()
    bt1 = get_buffer_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffer_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffer_s(s, v, vsp, m)
    bt2 = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffer_u( v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, da, temp, 1, ch_s = false)
    get_Γ_avx!(r, bs2, bt2, bu2,  a, temp, 2)

    # compute contributions for all lattice sites
    @turbo unroll = 1 for i in eachindex(r.sites)
        # read cached values for site i
        v1xx = temp[i, 1, 1]
        v1yy = temp[i, 2, 1]
        v1zz = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]

        v2xx = temp[i, 1, 2]
        v2yy = temp[i, 2, 2]
        v2zz = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]

        # compute contribution at site i
        Γxx = -p * (v1dd * v2xx + v1xx * v2dd - v1yy * v2zz - v1zz * v2yy)  
        Γyy = -p * (v1dd * v2yy - v1xx * v2zz + v1yy * v2dd - v1zz * v2xx)  
        Γzz = -p * (v1zz * v2dd + v1dd * v2zz - v1xx * v2yy - v1yy * v2xx)  
        Γdd = -p * (v1xx * v2xx + v1dd * v2dd + v1yy * v2yy + v1zz * v2zz)  

        # parse result to output buffer
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γyy
        buff[3, i] += dv * Γzz
        buff[4, i] += dv * Γdd
    end

    return nothing
end





# central kernel
function compute_s_central!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_z2_diag,
    da_l :: Action_z2_diag,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for left vertex
    bs1 = get_buffer_s(s, vs, -v, m)
    bt1 = get_buffer_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffer_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffer_s(s, v, vsp, m)
    bt2 = get_buffer_empty()
    bu2 = get_buffer_empty()

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1,    a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, da_l, temp, 2, ch_t = false, ch_u = false)

    # compute contributions for all lattice sites
    @turbo unroll = 1 for i in eachindex(r.sites)
        # read cached values for site i
        v1xx = temp[i, 1, 1]
        v1yy = temp[i, 2, 1]
        v1zz = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]

        v2xx = temp[i, 1, 2]
        v2yy = temp[i, 2, 2]
        v2zz = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]

        # compute contribution at site i
        Γxx = -p * (v1dd * v2xx + v1xx * v2dd - v1yy * v2zz - v1zz * v2yy)  
        Γyy = -p * (v1dd * v2yy - v1xx * v2zz + v1yy * v2dd - v1zz * v2xx)  
        Γzz = -p * (v1zz * v2dd + v1dd * v2zz - v1xx * v2yy - v1yy * v2xx)  
        Γdd = -p * (v1xx * v2xx + v1dd * v2dd + v1yy * v2yy + v1zz * v2zz)  

        # parse result to output buffer
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γyy
        buff[3, i] += dv * Γzz
        buff[4, i] += dv * Γdd
    end

    return nothing
end