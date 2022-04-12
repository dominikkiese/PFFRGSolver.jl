
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
    a    :: Action_su2_hkg,
    da   :: Action_su2_hkg,
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

    # compute contributions on all lattice sites 
    @turbo unroll = 1 for i in eachindex(r.sites)
        # read cached values for site i
        v1xx = temp[i, 1,  1]
        v1yy = temp[i, 2,  1]
        v1zz = temp[i, 3,  1]
        v1xy = temp[i, 4,  1]
        v1xz = temp[i, 5,  1]
        v1yz = temp[i, 6,  1]
        v1yx = temp[i, 7,  1]
        v1zx = temp[i, 8,  1]
        v1zy = temp[i, 9,  1]
        v1dd = temp[i, 10, 1]
        v1xd = temp[i, 11, 1]
        v1yd = temp[i, 12, 1]
        v1zd = temp[i, 13, 1]
        v1dx = temp[i, 14, 1]
        v1dy = temp[i, 15, 1]
        v1dz = temp[i, 16, 1]
        v2xx = temp[i, 1,  2]
        v2yy = temp[i, 2,  2]
        v2zz = temp[i, 3,  2]
        v2xy = temp[i, 4,  2]
        v2xz = temp[i, 5,  2]
        v2yz = temp[i, 6,  2]
        v2yx = temp[i, 7,  2]
        v2zx = temp[i, 8,  2]
        v2zy = temp[i, 9,  2]
        v2dd = temp[i, 10, 2]
        v2xd = temp[i, 11, 2]
        v2yd = temp[i, 12, 2]
        v2zd = temp[i, 13, 2]
        v2dx = temp[i, 14, 2]
        v2dy = temp[i, 15, 2]
        v2dz = temp[i, 16, 2]

        # compute contribution at site i
        Γxx = -p * ( v1dd * v2xx - v1dx * v2xd + v1dy * v2xz - v1dz * v2xy - v1xd * v2dx + v1xx * v2dd + v1xy * v2dz - v1xz * v2dy + v1yd * v2zx + v1yx * v2zd - v1yy * v2zz + v1yz * v2zy - v1zd * v2yx - v1zx * v2yd + v1zy * v2yz - v1zz * v2yy)
        Γyy = -p * ( v1dd * v2yy - v1dx * v2yz - v1dy * v2yd + v1dz * v2yx - v1xd * v2zy - v1xx * v2zz - v1xy * v2zd + v1xz * v2zx - v1yd * v2dy - v1yx * v2dz + v1yy * v2dd + v1yz * v2dx + v1zd * v2xy + v1zx * v2xz + v1zy * v2xd - v1zz * v2xx) 
        Γzz = -p * ( v1dd * v2zz + v1dx * v2zy - v1dy * v2zx - v1dz * v2zd + v1xd * v2yz - v1xx * v2yy + v1xy * v2yx + v1xz * v2yd - v1yd * v2xz + v1yx * v2xy - v1yy * v2xx - v1yz * v2xd - v1zd * v2dz + v1zx * v2dy - v1zy * v2dx + v1zz * v2dd)
        Γxy = -p * ( v1dd * v2xy - v1dx * v2xz - v1dy * v2xd + v1dz * v2xx - v1xd * v2dy - v1xx * v2dz + v1xy * v2dd + v1xz * v2dx + v1yd * v2zy + v1yx * v2zz + v1yy * v2zd - v1yz * v2zx - v1zd * v2yy - v1zx * v2yz - v1zy * v2yd + v1zz * v2yx)
        Γxz = -p * ( v1dd * v2xz + v1dx * v2xy - v1dy * v2xx - v1dz * v2xd - v1xd * v2dz + v1xx * v2dy - v1xy * v2dx + v1xz * v2dd + v1yd * v2zz - v1yx * v2zy + v1yy * v2zx + v1yz * v2zd - v1zd * v2yz + v1zx * v2yy - v1zy * v2yx - v1zz * v2yd)
        Γyz = -p * ( v1dd * v2yz + v1dx * v2yy - v1dy * v2yx - v1dz * v2yd - v1xd * v2zz + v1xx * v2zy - v1xy * v2zx - v1xz * v2zd - v1yd * v2dz + v1yx * v2dy - v1yy * v2dx + v1yz * v2dd + v1zd * v2xz - v1zx * v2xy + v1zy * v2xx + v1zz * v2xd)
        Γyx = -p * ( v1dd * v2yx - v1dx * v2yd + v1dy * v2yz - v1dz * v2yy - v1xd * v2zx - v1xx * v2zd + v1xy * v2zz - v1xz * v2zy - v1yd * v2dx + v1yx * v2dd + v1yy * v2dz - v1yz * v2dy + v1zd * v2xx + v1zx * v2xd - v1zy * v2xz + v1zz * v2xy)
        Γzx = -p * ( v1dd * v2zx - v1dx * v2zd + v1dy * v2zz - v1dz * v2zy + v1xd * v2yx + v1xx * v2yd - v1xy * v2yz + v1xz * v2yy - v1yd * v2xx - v1yx * v2xd + v1yy * v2xz - v1yz * v2xy - v1zd * v2dx + v1zx * v2dd + v1zy * v2dz - v1zz * v2dy)
        Γzy = -p * ( v1dd * v2zy - v1dx * v2zz - v1dy * v2zd + v1dz * v2zx + v1xd * v2yy + v1xx * v2yz + v1xy * v2yd - v1xz * v2yx - v1yd * v2xy - v1yx * v2xz - v1yy * v2xd + v1yz * v2xx - v1zd * v2dy - v1zx * v2dz + v1zy * v2dd + v1zz * v2dx)
        Γdd = -p * ( v1dd * v2dd - v1dx * v2dx - v1dy * v2dy - v1dz * v2dz - v1xd * v2xd + v1xx * v2xx + v1xy * v2xy + v1xz * v2xz - v1yd * v2yd + v1yx * v2yx + v1yy * v2yy + v1yz * v2yz - v1zd * v2zd + v1zx * v2zx + v1zy * v2zy + v1zz * v2zz)
        Γxd = -p * ( v1dd * v2xd + v1dx * v2xx + v1dy * v2xy + v1dz * v2xz + v1xd * v2dd + v1xx * v2dx + v1xy * v2dy + v1xz * v2dz + v1yd * v2zd - v1yx * v2zx - v1yy * v2zy - v1yz * v2zz - v1zd * v2yd + v1zx * v2yx + v1zy * v2yy + v1zz * v2yz)
        Γyd = -p * ( v1dd * v2yd + v1dx * v2yx + v1dy * v2yy + v1dz * v2yz - v1xd * v2zd + v1xx * v2zx + v1xy * v2zy + v1xz * v2zz + v1yd * v2dd + v1yx * v2dx + v1yy * v2dy + v1yz * v2dz + v1zd * v2xd - v1zx * v2xx - v1zy * v2xy - v1zz * v2xz)
        Γzd = -p * ( v1dd * v2zd + v1dx * v2zx + v1dy * v2zy + v1dz * v2zz + v1xd * v2yd - v1xx * v2yx - v1xy * v2yy - v1xz * v2yz - v1yd * v2xd + v1yx * v2xx + v1yy * v2xy + v1yz * v2xz + v1zd * v2dd + v1zx * v2dx + v1zy * v2dy + v1zz * v2dz)
        Γdx = -p * ( v1dd * v2dx + v1dx * v2dd + v1dy * v2dz - v1dz * v2dy + v1xd * v2xx + v1xx * v2xd - v1xy * v2xz + v1xz * v2xy + v1yd * v2yx + v1yx * v2yd - v1yy * v2yz + v1yz * v2yy + v1zd * v2zx + v1zx * v2zd - v1zy * v2zz + v1zz * v2zy)
        Γdy = -p * ( v1dd * v2dy - v1dx * v2dz + v1dy * v2dd + v1dz * v2dx + v1xd * v2xy + v1xx * v2xz + v1xy * v2xd - v1xz * v2xx + v1yd * v2yy + v1yx * v2yz + v1yy * v2yd - v1yz * v2yx + v1zd * v2zy + v1zx * v2zz + v1zy * v2zd - v1zz * v2zx)
        Γdz = -p * ( v1dd * v2dz + v1dx * v2dy - v1dy * v2dx + v1dz * v2dd + v1xd * v2xz - v1xx * v2xy + v1xy * v2xx + v1xz * v2xd + v1yd * v2yz - v1yx * v2yy + v1yy * v2yx + v1yz * v2yd + v1zd * v2zz - v1zx * v2zy + v1zy * v2zx + v1zz * v2zd)

        # parse result to output buffer 
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γyy 
        buff[3, i] += dv * Γzz
        buff[4, i] += dv * Γxy
        buff[5, i] += dv * Γxz
        buff[6, i] += dv * Γyz
        buff[7, i] += dv * Γyx 
        buff[8, i] += dv * Γzx
        buff[9, i] += dv * Γzy
        buff[10, i] += dv * Γdd
        buff[11, i] += dv * Γxd
        buff[12, i] += dv * Γyd 
        buff[13, i] += dv * Γzd
        buff[14, i] += dv * Γdx
        buff[15, i] += dv * Γdy
        buff[16, i] += dv * Γdz
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
    a    :: Action_su2_hkg,
    da   :: Action_su2_hkg,
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

    # compute contributions on all lattice sites 
    @turbo unroll = 1 for i in eachindex(r.sites)
        # read cached values for site i
        v1xx = temp[i, 1,  1]
        v1yy = temp[i, 2,  1]
        v1zz = temp[i, 3,  1]
        v1xy = temp[i, 4,  1]
        v1xz = temp[i, 5,  1]
        v1yz = temp[i, 6,  1]
        v1yx = temp[i, 7,  1]
        v1zx = temp[i, 8,  1]
        v1zy = temp[i, 9,  1]
        v1dd = temp[i, 10, 1]
        v1xd = temp[i, 11, 1]
        v1yd = temp[i, 12, 1]
        v1zd = temp[i, 13, 1]
        v1dx = temp[i, 14, 1]
        v1dy = temp[i, 15, 1]
        v1dz = temp[i, 16, 1]
        v2xx = temp[i, 1,  2]
        v2yy = temp[i, 2,  2]
        v2zz = temp[i, 3,  2]
        v2xy = temp[i, 4,  2]
        v2xz = temp[i, 5,  2]
        v2yz = temp[i, 6,  2]
        v2yx = temp[i, 7,  2]
        v2zx = temp[i, 8,  2]
        v2zy = temp[i, 9,  2]
        v2dd = temp[i, 10, 2]
        v2xd = temp[i, 11, 2]
        v2yd = temp[i, 12, 2]
        v2zd = temp[i, 13, 2]
        v2dx = temp[i, 14, 2]
        v2dy = temp[i, 15, 2]
        v2dz = temp[i, 16, 2]

       # compute contribution at site i
        Γxx = -p * ( v1dd * v2xx - v1dx * v2xd + v1dy * v2xz - v1dz * v2xy - v1xd * v2dx + v1xx * v2dd + v1xy * v2dz - v1xz * v2dy + v1yd * v2zx + v1yx * v2zd - v1yy * v2zz + v1yz * v2zy - v1zd * v2yx - v1zx * v2yd + v1zy * v2yz - v1zz * v2yy)
        Γyy = -p * ( v1dd * v2yy - v1dx * v2yz - v1dy * v2yd + v1dz * v2yx - v1xd * v2zy - v1xx * v2zz - v1xy * v2zd + v1xz * v2zx - v1yd * v2dy - v1yx * v2dz + v1yy * v2dd + v1yz * v2dx + v1zd * v2xy + v1zx * v2xz + v1zy * v2xd - v1zz * v2xx) 
        Γzz = -p * ( v1dd * v2zz + v1dx * v2zy - v1dy * v2zx - v1dz * v2zd + v1xd * v2yz - v1xx * v2yy + v1xy * v2yx + v1xz * v2yd - v1yd * v2xz + v1yx * v2xy - v1yy * v2xx - v1yz * v2xd - v1zd * v2dz + v1zx * v2dy - v1zy * v2dx + v1zz * v2dd)
        Γxy = -p * ( v1dd * v2xy - v1dx * v2xz - v1dy * v2xd + v1dz * v2xx - v1xd * v2dy - v1xx * v2dz + v1xy * v2dd + v1xz * v2dx + v1yd * v2zy + v1yx * v2zz + v1yy * v2zd - v1yz * v2zx - v1zd * v2yy - v1zx * v2yz - v1zy * v2yd + v1zz * v2yx)
        Γxz = -p * ( v1dd * v2xz + v1dx * v2xy - v1dy * v2xx - v1dz * v2xd - v1xd * v2dz + v1xx * v2dy - v1xy * v2dx + v1xz * v2dd + v1yd * v2zz - v1yx * v2zy + v1yy * v2zx + v1yz * v2zd - v1zd * v2yz + v1zx * v2yy - v1zy * v2yx - v1zz * v2yd)
        Γyz = -p * ( v1dd * v2yz + v1dx * v2yy - v1dy * v2yx - v1dz * v2yd - v1xd * v2zz + v1xx * v2zy - v1xy * v2zx - v1xz * v2zd - v1yd * v2dz + v1yx * v2dy - v1yy * v2dx + v1yz * v2dd + v1zd * v2xz - v1zx * v2xy + v1zy * v2xx + v1zz * v2xd)
        Γyx = -p * ( v1dd * v2yx - v1dx * v2yd + v1dy * v2yz - v1dz * v2yy - v1xd * v2zx - v1xx * v2zd + v1xy * v2zz - v1xz * v2zy - v1yd * v2dx + v1yx * v2dd + v1yy * v2dz - v1yz * v2dy + v1zd * v2xx + v1zx * v2xd - v1zy * v2xz + v1zz * v2xy)
        Γzx = -p * ( v1dd * v2zx - v1dx * v2zd + v1dy * v2zz - v1dz * v2zy + v1xd * v2yx + v1xx * v2yd - v1xy * v2yz + v1xz * v2yy - v1yd * v2xx - v1yx * v2xd + v1yy * v2xz - v1yz * v2xy - v1zd * v2dx + v1zx * v2dd + v1zy * v2dz - v1zz * v2dy)
        Γzy = -p * ( v1dd * v2zy - v1dx * v2zz - v1dy * v2zd + v1dz * v2zx + v1xd * v2yy + v1xx * v2yz + v1xy * v2yd - v1xz * v2yx - v1yd * v2xy - v1yx * v2xz - v1yy * v2xd + v1yz * v2xx - v1zd * v2dy - v1zx * v2dz + v1zy * v2dd + v1zz * v2dx)
        Γdd = -p * ( v1dd * v2dd - v1dx * v2dx - v1dy * v2dy - v1dz * v2dz - v1xd * v2xd + v1xx * v2xx + v1xy * v2xy + v1xz * v2xz - v1yd * v2yd + v1yx * v2yx + v1yy * v2yy + v1yz * v2yz - v1zd * v2zd + v1zx * v2zx + v1zy * v2zy + v1zz * v2zz)
        Γxd = -p * ( v1dd * v2xd + v1dx * v2xx + v1dy * v2xy + v1dz * v2xz + v1xd * v2dd + v1xx * v2dx + v1xy * v2dy + v1xz * v2dz + v1yd * v2zd - v1yx * v2zx - v1yy * v2zy - v1yz * v2zz - v1zd * v2yd + v1zx * v2yx + v1zy * v2yy + v1zz * v2yz)
        Γyd = -p * ( v1dd * v2yd + v1dx * v2yx + v1dy * v2yy + v1dz * v2yz - v1xd * v2zd + v1xx * v2zx + v1xy * v2zy + v1xz * v2zz + v1yd * v2dd + v1yx * v2dx + v1yy * v2dy + v1yz * v2dz + v1zd * v2xd - v1zx * v2xx - v1zy * v2xy - v1zz * v2xz)
        Γzd = -p * ( v1dd * v2zd + v1dx * v2zx + v1dy * v2zy + v1dz * v2zz + v1xd * v2yd - v1xx * v2yx - v1xy * v2yy - v1xz * v2yz - v1yd * v2xd + v1yx * v2xx + v1yy * v2xy + v1yz * v2xz + v1zd * v2dd + v1zx * v2dx + v1zy * v2dy + v1zz * v2dz)
        Γdx = -p * ( v1dd * v2dx + v1dx * v2dd + v1dy * v2dz - v1dz * v2dy + v1xd * v2xx + v1xx * v2xd - v1xy * v2xz + v1xz * v2xy + v1yd * v2yx + v1yx * v2yd - v1yy * v2yz + v1yz * v2yy + v1zd * v2zx + v1zx * v2zd - v1zy * v2zz + v1zz * v2zy)
        Γdy = -p * ( v1dd * v2dy - v1dx * v2dz + v1dy * v2dd + v1dz * v2dx + v1xd * v2xy + v1xx * v2xz + v1xy * v2xd - v1xz * v2xx + v1yd * v2yy + v1yx * v2yz + v1yy * v2yd - v1yz * v2yx + v1zd * v2zy + v1zx * v2zz + v1zy * v2zd - v1zz * v2zx)
        Γdz = -p * ( v1dd * v2dz + v1dx * v2dy - v1dy * v2dx + v1dz * v2dd + v1xd * v2xz - v1xx * v2xy + v1xy * v2xx + v1xz * v2xd + v1yd * v2yz - v1yx * v2yy + v1yy * v2yx + v1yz * v2yd + v1zd * v2zz - v1zx * v2zy + v1zy * v2zx + v1zz * v2zd)

        # parse result to output buffer 
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γyy 
        buff[3, i] += dv * Γzz
        buff[4, i] += dv * Γxy
        buff[5, i] += dv * Γxz
        buff[6, i] += dv * Γyz
        buff[7, i] += dv * Γyx 
        buff[8, i] += dv * Γzx
        buff[9, i] += dv * Γzy
        buff[10, i] += dv * Γdd
        buff[11, i] += dv * Γxd
        buff[12, i] += dv * Γyd 
        buff[13, i] += dv * Γzd
        buff[14, i] += dv * Γdx
        buff[15, i] += dv * Γdy
        buff[16, i] += dv * Γdz
    end

    return nothing 
 end
 
 
 
 
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
    a    :: Action_su2_hkg,
    da_l :: Action_su2_hkg,
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
    
    # compute contributions on all lattice sites 
    @turbo unroll = 1 for i in eachindex(r.sites)
       # read cached values for site i
       v1xx = temp[i, 1,  1]
       v1yy = temp[i, 2,  1]
       v1zz = temp[i, 3,  1]
       v1xy = temp[i, 4,  1]
       v1xz = temp[i, 5,  1]
       v1yz = temp[i, 6,  1]
       v1yx = temp[i, 7,  1]
       v1zx = temp[i, 8,  1]
       v1zy = temp[i, 9,  1]
       v1dd = temp[i, 10, 1]
       v1xd = temp[i, 11, 1]
       v1yd = temp[i, 12, 1]
       v1zd = temp[i, 13, 1]
       v1dx = temp[i, 14, 1]
       v1dy = temp[i, 15, 1]
       v1dz = temp[i, 16, 1]
       v2xx = temp[i, 1,  2]
       v2yy = temp[i, 2,  2]
       v2zz = temp[i, 3,  2]
       v2xy = temp[i, 4,  2]
       v2xz = temp[i, 5,  2]
       v2yz = temp[i, 6,  2]
       v2yx = temp[i, 7,  2]
       v2zx = temp[i, 8,  2]
       v2zy = temp[i, 9,  2]
       v2dd = temp[i, 10, 2]
       v2xd = temp[i, 11, 2]
       v2yd = temp[i, 12, 2]
       v2zd = temp[i, 13, 2]
       v2dx = temp[i, 14, 2]
       v2dy = temp[i, 15, 2]
       v2dz = temp[i, 16, 2]

      # compute contribution at site i
       Γxx = -p * ( v1dd * v2xx - v1dx * v2xd + v1dy * v2xz - v1dz * v2xy - v1xd * v2dx + v1xx * v2dd + v1xy * v2dz - v1xz * v2dy + v1yd * v2zx + v1yx * v2zd - v1yy * v2zz + v1yz * v2zy - v1zd * v2yx - v1zx * v2yd + v1zy * v2yz - v1zz * v2yy)
       Γyy = -p * ( v1dd * v2yy - v1dx * v2yz - v1dy * v2yd + v1dz * v2yx - v1xd * v2zy - v1xx * v2zz - v1xy * v2zd + v1xz * v2zx - v1yd * v2dy - v1yx * v2dz + v1yy * v2dd + v1yz * v2dx + v1zd * v2xy + v1zx * v2xz + v1zy * v2xd - v1zz * v2xx) 
       Γzz = -p * ( v1dd * v2zz + v1dx * v2zy - v1dy * v2zx - v1dz * v2zd + v1xd * v2yz - v1xx * v2yy + v1xy * v2yx + v1xz * v2yd - v1yd * v2xz + v1yx * v2xy - v1yy * v2xx - v1yz * v2xd - v1zd * v2dz + v1zx * v2dy - v1zy * v2dx + v1zz * v2dd)
       Γxy = -p * ( v1dd * v2xy - v1dx * v2xz - v1dy * v2xd + v1dz * v2xx - v1xd * v2dy - v1xx * v2dz + v1xy * v2dd + v1xz * v2dx + v1yd * v2zy + v1yx * v2zz + v1yy * v2zd - v1yz * v2zx - v1zd * v2yy - v1zx * v2yz - v1zy * v2yd + v1zz * v2yx)
       Γxz = -p * ( v1dd * v2xz + v1dx * v2xy - v1dy * v2xx - v1dz * v2xd - v1xd * v2dz + v1xx * v2dy - v1xy * v2dx + v1xz * v2dd + v1yd * v2zz - v1yx * v2zy + v1yy * v2zx + v1yz * v2zd - v1zd * v2yz + v1zx * v2yy - v1zy * v2yx - v1zz * v2yd)
       Γyz = -p * ( v1dd * v2yz + v1dx * v2yy - v1dy * v2yx - v1dz * v2yd - v1xd * v2zz + v1xx * v2zy - v1xy * v2zx - v1xz * v2zd - v1yd * v2dz + v1yx * v2dy - v1yy * v2dx + v1yz * v2dd + v1zd * v2xz - v1zx * v2xy + v1zy * v2xx + v1zz * v2xd)
       Γyx = -p * ( v1dd * v2yx - v1dx * v2yd + v1dy * v2yz - v1dz * v2yy - v1xd * v2zx - v1xx * v2zd + v1xy * v2zz - v1xz * v2zy - v1yd * v2dx + v1yx * v2dd + v1yy * v2dz - v1yz * v2dy + v1zd * v2xx + v1zx * v2xd - v1zy * v2xz + v1zz * v2xy)
       Γzx = -p * ( v1dd * v2zx - v1dx * v2zd + v1dy * v2zz - v1dz * v2zy + v1xd * v2yx + v1xx * v2yd - v1xy * v2yz + v1xz * v2yy - v1yd * v2xx - v1yx * v2xd + v1yy * v2xz - v1yz * v2xy - v1zd * v2dx + v1zx * v2dd + v1zy * v2dz - v1zz * v2dy)
       Γzy = -p * ( v1dd * v2zy - v1dx * v2zz - v1dy * v2zd + v1dz * v2zx + v1xd * v2yy + v1xx * v2yz + v1xy * v2yd - v1xz * v2yx - v1yd * v2xy - v1yx * v2xz - v1yy * v2xd + v1yz * v2xx - v1zd * v2dy - v1zx * v2dz + v1zy * v2dd + v1zz * v2dx)
       Γdd = -p * ( v1dd * v2dd - v1dx * v2dx - v1dy * v2dy - v1dz * v2dz - v1xd * v2xd + v1xx * v2xx + v1xy * v2xy + v1xz * v2xz - v1yd * v2yd + v1yx * v2yx + v1yy * v2yy + v1yz * v2yz - v1zd * v2zd + v1zx * v2zx + v1zy * v2zy + v1zz * v2zz)
       Γxd = -p * ( v1dd * v2xd + v1dx * v2xx + v1dy * v2xy + v1dz * v2xz + v1xd * v2dd + v1xx * v2dx + v1xy * v2dy + v1xz * v2dz + v1yd * v2zd - v1yx * v2zx - v1yy * v2zy - v1yz * v2zz - v1zd * v2yd + v1zx * v2yx + v1zy * v2yy + v1zz * v2yz)
       Γyd = -p * ( v1dd * v2yd + v1dx * v2yx + v1dy * v2yy + v1dz * v2yz - v1xd * v2zd + v1xx * v2zx + v1xy * v2zy + v1xz * v2zz + v1yd * v2dd + v1yx * v2dx + v1yy * v2dy + v1yz * v2dz + v1zd * v2xd - v1zx * v2xx - v1zy * v2xy - v1zz * v2xz)
       Γzd = -p * ( v1dd * v2zd + v1dx * v2zx + v1dy * v2zy + v1dz * v2zz + v1xd * v2yd - v1xx * v2yx - v1xy * v2yy - v1xz * v2yz - v1yd * v2xd + v1yx * v2xx + v1yy * v2xy + v1yz * v2xz + v1zd * v2dd + v1zx * v2dx + v1zy * v2dy + v1zz * v2dz)
       Γdx = -p * ( v1dd * v2dx + v1dx * v2dd + v1dy * v2dz - v1dz * v2dy + v1xd * v2xx + v1xx * v2xd - v1xy * v2xz + v1xz * v2xy + v1yd * v2yx + v1yx * v2yd - v1yy * v2yz + v1yz * v2yy + v1zd * v2zx + v1zx * v2zd - v1zy * v2zz + v1zz * v2zy)
       Γdy = -p * ( v1dd * v2dy - v1dx * v2dz + v1dy * v2dd + v1dz * v2dx + v1xd * v2xy + v1xx * v2xz + v1xy * v2xd - v1xz * v2xx + v1yd * v2yy + v1yx * v2yz + v1yy * v2yd - v1yz * v2yx + v1zd * v2zy + v1zx * v2zz + v1zy * v2zd - v1zz * v2zx)
       Γdz = -p * ( v1dd * v2dz + v1dx * v2dy - v1dy * v2dx + v1dz * v2dd + v1xd * v2xz - v1xx * v2xy + v1xy * v2xx + v1xz * v2xd + v1yd * v2yz - v1yx * v2yy + v1yy * v2yx + v1yz * v2yd + v1zd * v2zz - v1zx * v2zy + v1zy * v2zx + v1zz * v2zd)

       # parse result to output buffer 
       buff[1, i] += dv * Γxx
       buff[2, i] += dv * Γyy 
       buff[3, i] += dv * Γzz
       buff[4, i] += dv * Γxy
       buff[5, i] += dv * Γxz
       buff[6, i] += dv * Γyz
       buff[7, i] += dv * Γyx 
       buff[8, i] += dv * Γzx
       buff[9, i] += dv * Γzy
       buff[10, i] += dv * Γdd
       buff[11, i] += dv * Γxd
       buff[12, i] += dv * Γyd 
       buff[13, i] += dv * Γzd
       buff[14, i] += dv * Γdx
       buff[15, i] += dv * Γdy
       buff[16, i] += dv * Γdz
    end

    return nothing 
 end