# Katanin kernel
function compute_u_kat!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64,
    vu   :: Float64,
    vup  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg,
    da   :: Action_su2_hkg,
    temp :: Array{Float64, 3}
    )    :: Nothing
    # get propagator
    p = get_propagator_kat(Λ, v - 0.5 * u, v + 0.5 * u, m, a, da) + get_propagator_kat(Λ, v + 0.5 * u, v - 0.5 * u, m, a, da)
    # get buffers for left vertex
    bs1 = get_buffer_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffer_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = get_buffer_u(u, vu, v, m)
    # get buffers for right vertex
    bs2 = get_buffer_s( v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m)
    bt2 = get_buffer_t(-v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m)
    bu2 = get_buffer_u(u, v, vup, m)
    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)
    # compute contributions for all lattice sites
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

        # compute contributions at site i 
        Γdd = -p * (+ 1.0 * v1dd * v2dd
        - 1.0 * v1dx * v2dx
        - 1.0 * v1dy * v2dy
        - 1.0 * v1dz * v2dz
        - 1.0 * v1xd * v2xd
        + 1.0 * v1xx * v2xx
        + 1.0 * v1xy * v2xy
        + 1.0 * v1xz * v2xz
        - 1.0 * v1yd * v2yd
        + 1.0 * v1yx * v2yx
        + 1.0 * v1yy * v2yy
        + 1.0 * v1yz * v2yz
        - 1.0 * v1zd * v2zd
        + 1.0 * v1zx * v2zx
        + 1.0 * v1zy * v2zy
        + 1.0 * v1zz * v2zz
        )

        Γdx = -p * (+ 1.0 * v1dd * v2dx
        + 1.0 * v1dx * v2dd
        - 1.0 * v1dy * v2dz
        + 1.0 * v1dz * v2dy
        + 1.0 * v1xd * v2xx
        + 1.0 * v1xx * v2xd
        + 1.0 * v1xy * v2xz
        - 1.0 * v1xz * v2xy
        + 1.0 * v1yd * v2yx
        + 1.0 * v1yx * v2yd
        + 1.0 * v1yy * v2yz
        - 1.0 * v1yz * v2yy
        + 1.0 * v1zd * v2zx
        + 1.0 * v1zx * v2zd
        + 1.0 * v1zy * v2zz
        - 1.0 * v1zz * v2zy
        )

        Γdy = -p * (+ 1.0 * v1dd * v2dy
        + 1.0 * v1dx * v2dz
        + 1.0 * v1dy * v2dd
        - 1.0 * v1dz * v2dx
        + 1.0 * v1xd * v2xy
        - 1.0 * v1xx * v2xz
        + 1.0 * v1xy * v2xd
        + 1.0 * v1xz * v2xx
        + 1.0 * v1yd * v2yy
        - 1.0 * v1yx * v2yz
        + 1.0 * v1yy * v2yd
        + 1.0 * v1yz * v2yx
        + 1.0 * v1zd * v2zy
        - 1.0 * v1zx * v2zz
        + 1.0 * v1zy * v2zd
        + 1.0 * v1zz * v2zx
        )

        Γdz = -p * (+ 1.0 * v1dd * v2dz
        - 1.0 * v1dx * v2dy
        + 1.0 * v1dy * v2dx
        + 1.0 * v1dz * v2dd
        + 1.0 * v1xd * v2xz
        + 1.0 * v1xx * v2xy
        - 1.0 * v1xy * v2xx
        + 1.0 * v1xz * v2xd
        + 1.0 * v1yd * v2yz
        + 1.0 * v1yx * v2yy
        - 1.0 * v1yy * v2yx
        + 1.0 * v1yz * v2yd
        + 1.0 * v1zd * v2zz
        + 1.0 * v1zx * v2zy
        - 1.0 * v1zy * v2zx
        + 1.0 * v1zz * v2zd
        )

        Γxd = -p * (+ 1.0 * v1dd * v2xd
        + 1.0 * v1dx * v2xx
        + 1.0 * v1dy * v2xy
        + 1.0 * v1dz * v2xz
        + 1.0 * v1xd * v2dd
        + 1.0 * v1xx * v2dx
        + 1.0 * v1xy * v2dy
        + 1.0 * v1xz * v2dz
        + 1.0 * v1yd * v2zd
        - 1.0 * v1yx * v2zx
        - 1.0 * v1yy * v2zy
        - 1.0 * v1yz * v2zz
        - 1.0 * v1zd * v2yd
        + 1.0 * v1zx * v2yx
        + 1.0 * v1zy * v2yy
        + 1.0 * v1zz * v2yz
        )

        Γxx = -p * (+ 1.0 * v1dd * v2xx
        - 1.0 * v1dx * v2xd
        - 1.0 * v1dy * v2xz
        + 1.0 * v1dz * v2xy
        - 1.0 * v1xd * v2dx
        + 1.0 * v1xx * v2dd
        - 1.0 * v1xy * v2dz
        + 1.0 * v1xz * v2dy
        + 1.0 * v1yd * v2zx
        + 1.0 * v1yx * v2zd
        + 1.0 * v1yy * v2zz
        - 1.0 * v1yz * v2zy
        - 1.0 * v1zd * v2yx
        - 1.0 * v1zx * v2yd
        - 1.0 * v1zy * v2yz
        + 1.0 * v1zz * v2yy
        )

        Γxy = -p * (+ 1.0 * v1dd * v2xy
        + 1.0 * v1dx * v2xz
        - 1.0 * v1dy * v2xd
        - 1.0 * v1dz * v2xx
        - 1.0 * v1xd * v2dy
        + 1.0 * v1xx * v2dz
        + 1.0 * v1xy * v2dd
        - 1.0 * v1xz * v2dx
        + 1.0 * v1yd * v2zy
        - 1.0 * v1yx * v2zz
        + 1.0 * v1yy * v2zd
        + 1.0 * v1yz * v2zx
        - 1.0 * v1zd * v2yy
        + 1.0 * v1zx * v2yz
        - 1.0 * v1zy * v2yd
        - 1.0 * v1zz * v2yx
        )

        Γxz = -p * (+ 1.0 * v1dd * v2xz
        - 1.0 * v1dx * v2xy
        + 1.0 * v1dy * v2xx
        - 1.0 * v1dz * v2xd
        - 1.0 * v1xd * v2dz
        - 1.0 * v1xx * v2dy
        + 1.0 * v1xy * v2dx
        + 1.0 * v1xz * v2dd
        + 1.0 * v1yd * v2zz
        + 1.0 * v1yx * v2zy
        - 1.0 * v1yy * v2zx
        + 1.0 * v1yz * v2zd
        - 1.0 * v1zd * v2yz
        - 1.0 * v1zx * v2yy
        + 1.0 * v1zy * v2yx
        - 1.0 * v1zz * v2yd

        )

        Γyd = -p * (+ 1.0 * v1dd * v2yd
        + 1.0 * v1dx * v2yx
        + 1.0 * v1dy * v2yy
        + 1.0 * v1dz * v2yz
        - 1.0 * v1xd * v2zd
        + 1.0 * v1xx * v2zx
        + 1.0 * v1xy * v2zy
        + 1.0 * v1xz * v2zz
        + 1.0 * v1yd * v2dd
        + 1.0 * v1yx * v2dx
        + 1.0 * v1yy * v2dy
        + 1.0 * v1yz * v2dz
        + 1.0 * v1zd * v2xd
        - 1.0 * v1zx * v2xx
        - 1.0 * v1zy * v2xy
        - 1.0 * v1zz * v2xz
        )

        Γyx = -p * (+ 1.0 * v1dd * v2yx
        - 1.0 * v1dx * v2yd
        - 1.0 * v1dy * v2yz
        + 1.0 * v1dz * v2yy
        - 1.0 * v1xd * v2zx
        - 1.0 * v1xx * v2zd
        - 1.0 * v1xy * v2zz
        + 1.0 * v1xz * v2zy
        - 1.0 * v1yd * v2dx
        + 1.0 * v1yx * v2dd
        - 1.0 * v1yy * v2dz
        + 1.0 * v1yz * v2dy
        + 1.0 * v1zd * v2xx
        + 1.0 * v1zx * v2xd
        + 1.0 * v1zy * v2xz
        - 1.0 * v1zz * v2xy
        )

        Γyy = -p * (+ 1.0 * v1dd * v2yy
        + 1.0 * v1dx * v2yz
        - 1.0 * v1dy * v2yd
        - 1.0 * v1dz * v2yx
        - 1.0 * v1xd * v2zy
        + 1.0 * v1xx * v2zz
        - 1.0 * v1xy * v2zd
        - 1.0 * v1xz * v2zx
        - 1.0 * v1yd * v2dy
        + 1.0 * v1yx * v2dz
        + 1.0 * v1yy * v2dd
        - 1.0 * v1yz * v2dx
        + 1.0 * v1zd * v2xy
        - 1.0 * v1zx * v2xz
        + 1.0 * v1zy * v2xd
        + 1.0 * v1zz * v2xx
        )

        Γyz = -p * (+ 1.0 * v1dd * v2yz
        - 1.0 * v1dx * v2yy
        + 1.0 * v1dy * v2yx
        - 1.0 * v1dz * v2yd
        - 1.0 * v1xd * v2zz
        - 1.0 * v1xx * v2zy
        + 1.0 * v1xy * v2zx
        - 1.0 * v1xz * v2zd
        - 1.0 * v1yd * v2dz
        - 1.0 * v1yx * v2dy
        + 1.0 * v1yy * v2dx
        + 1.0 * v1yz * v2dd
        + 1.0 * v1zd * v2xz
        + 1.0 * v1zx * v2xy
        - 1.0 * v1zy * v2xx
        + 1.0 * v1zz * v2xd
        )

        Γzd = -p * (+ 1.0 * v1dd * v2zd
        + 1.0 * v1dx * v2zx
        + 1.0 * v1dy * v2zy
        + 1.0 * v1dz * v2zz
        + 1.0 * v1xd * v2yd
        - 1.0 * v1xx * v2yx
        - 1.0 * v1xy * v2yy
        - 1.0 * v1xz * v2yz
        - 1.0 * v1yd * v2xd
        + 1.0 * v1yx * v2xx
        + 1.0 * v1yy * v2xy
        + 1.0 * v1yz * v2xz
        + 1.0 * v1zd * v2dd
        + 1.0 * v1zx * v2dx
        + 1.0 * v1zy * v2dy
        + 1.0 * v1zz * v2dz
        )

        Γzx = -p * (+ 1.0 * v1dd * v2zx
        - 1.0 * v1dx * v2zd
        - 1.0 * v1dy * v2zz
        + 1.0 * v1dz * v2zy
        + 1.0 * v1xd * v2yx
        + 1.0 * v1xx * v2yd
        + 1.0 * v1xy * v2yz
        - 1.0 * v1xz * v2yy
        - 1.0 * v1yd * v2xx
        - 1.0 * v1yx * v2xd
        - 1.0 * v1yy * v2xz
        + 1.0 * v1yz * v2xy
        - 1.0 * v1zd * v2dx
        + 1.0 * v1zx * v2dd
        - 1.0 * v1zy * v2dz
        + 1.0 * v1zz * v2dy
        )

        Γzy = -p * (+ 1.0 * v1dd * v2zy
        + 1.0 * v1dx * v2zz
        - 1.0 * v1dy * v2zd
        - 1.0 * v1dz * v2zx
        + 1.0 * v1xd * v2yy
        - 1.0 * v1xx * v2yz
        + 1.0 * v1xy * v2yd
        + 1.0 * v1xz * v2yx
        - 1.0 * v1yd * v2xy
        + 1.0 * v1yx * v2xz
        - 1.0 * v1yy * v2xd
        - 1.0 * v1yz * v2xx
        - 1.0 * v1zd * v2dy
        + 1.0 * v1zx * v2dz
        + 1.0 * v1zy * v2dd
        - 1.0 * v1zz * v2dx
        )

        Γzz = -p * (+ 1.0 * v1dd * v2zz
        - 1.0 * v1dx * v2zy
        + 1.0 * v1dy * v2zx
        - 1.0 * v1dz * v2zd
        + 1.0 * v1xd * v2yz
        + 1.0 * v1xx * v2yy
        - 1.0 * v1xy * v2yx
        + 1.0 * v1xz * v2yd
        - 1.0 * v1yd * v2xz
        - 1.0 * v1yx * v2xy
        + 1.0 * v1yy * v2xx
        - 1.0 * v1yz * v2xd
        - 1.0 * v1zd * v2dz
        - 1.0 * v1zx * v2dy
        + 1.0 * v1zy * v2dx
        + 1.0 * v1zz * v2dd
        )

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
 function compute_u_left!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64,
    vu   :: Float64,
    vup  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg,
    da   :: Action_su2_hkg,
    temp :: Array{Float64, 3}
    )    :: Nothing
    # get propagator
    p = -get_propagator(Λ, v - 0.5 * u, v + 0.5 * u, m, a)
    # get buffers for left vertex
    bs1 = get_buffer_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffer_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = get_buffer_empty()
    # get buffers for right vertex
    bs2 = get_buffer_s( v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m)
    bt2 = get_buffer_t(-v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m)
    bu2 = get_buffer_u(u, v, vup, m)
    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, da, temp, 1, ch_u = false)
    get_Γ_avx!(r, bs2, bt2, bu2,  a, temp, 2)
    # compute contributions for all lattice sites
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

        # compute contributions at site i 
        Γdd = -p * (+ 1.0 * v1dd * v2dd
        - 1.0 * v1dx * v2dx
        - 1.0 * v1dy * v2dy
        - 1.0 * v1dz * v2dz
        - 1.0 * v1xd * v2xd
        + 1.0 * v1xx * v2xx
        + 1.0 * v1xy * v2xy
        + 1.0 * v1xz * v2xz
        - 1.0 * v1yd * v2yd
        + 1.0 * v1yx * v2yx
        + 1.0 * v1yy * v2yy
        + 1.0 * v1yz * v2yz
        - 1.0 * v1zd * v2zd
        + 1.0 * v1zx * v2zx
        + 1.0 * v1zy * v2zy
        + 1.0 * v1zz * v2zz
        )

        Γdx = -p * (+ 1.0 * v1dd * v2dx
        + 1.0 * v1dx * v2dd
        - 1.0 * v1dy * v2dz
        + 1.0 * v1dz * v2dy
        + 1.0 * v1xd * v2xx
        + 1.0 * v1xx * v2xd
        + 1.0 * v1xy * v2xz
        - 1.0 * v1xz * v2xy
        + 1.0 * v1yd * v2yx
        + 1.0 * v1yx * v2yd
        + 1.0 * v1yy * v2yz
        - 1.0 * v1yz * v2yy
        + 1.0 * v1zd * v2zx
        + 1.0 * v1zx * v2zd
        + 1.0 * v1zy * v2zz
        - 1.0 * v1zz * v2zy
        )

        Γdy = -p * (+ 1.0 * v1dd * v2dy
        + 1.0 * v1dx * v2dz
        + 1.0 * v1dy * v2dd
        - 1.0 * v1dz * v2dx
        + 1.0 * v1xd * v2xy
        - 1.0 * v1xx * v2xz
        + 1.0 * v1xy * v2xd
        + 1.0 * v1xz * v2xx
        + 1.0 * v1yd * v2yy
        - 1.0 * v1yx * v2yz
        + 1.0 * v1yy * v2yd
        + 1.0 * v1yz * v2yx
        + 1.0 * v1zd * v2zy
        - 1.0 * v1zx * v2zz
        + 1.0 * v1zy * v2zd
        + 1.0 * v1zz * v2zx
        )

        Γdz = -p * (+ 1.0 * v1dd * v2dz
        - 1.0 * v1dx * v2dy
        + 1.0 * v1dy * v2dx
        + 1.0 * v1dz * v2dd
        + 1.0 * v1xd * v2xz
        + 1.0 * v1xx * v2xy
        - 1.0 * v1xy * v2xx
        + 1.0 * v1xz * v2xd
        + 1.0 * v1yd * v2yz
        + 1.0 * v1yx * v2yy
        - 1.0 * v1yy * v2yx
        + 1.0 * v1yz * v2yd
        + 1.0 * v1zd * v2zz
        + 1.0 * v1zx * v2zy
        - 1.0 * v1zy * v2zx
        + 1.0 * v1zz * v2zd
        )

        Γxd = -p * (+ 1.0 * v1dd * v2xd
        + 1.0 * v1dx * v2xx
        + 1.0 * v1dy * v2xy
        + 1.0 * v1dz * v2xz
        + 1.0 * v1xd * v2dd
        + 1.0 * v1xx * v2dx
        + 1.0 * v1xy * v2dy
        + 1.0 * v1xz * v2dz
        + 1.0 * v1yd * v2zd
        - 1.0 * v1yx * v2zx
        - 1.0 * v1yy * v2zy
        - 1.0 * v1yz * v2zz
        - 1.0 * v1zd * v2yd
        + 1.0 * v1zx * v2yx
        + 1.0 * v1zy * v2yy
        + 1.0 * v1zz * v2yz
        )

        Γxx = -p * (+ 1.0 * v1dd * v2xx
        - 1.0 * v1dx * v2xd
        - 1.0 * v1dy * v2xz
        + 1.0 * v1dz * v2xy
        - 1.0 * v1xd * v2dx
        + 1.0 * v1xx * v2dd
        - 1.0 * v1xy * v2dz
        + 1.0 * v1xz * v2dy
        + 1.0 * v1yd * v2zx
        + 1.0 * v1yx * v2zd
        + 1.0 * v1yy * v2zz
        - 1.0 * v1yz * v2zy
        - 1.0 * v1zd * v2yx
        - 1.0 * v1zx * v2yd
        - 1.0 * v1zy * v2yz
        + 1.0 * v1zz * v2yy
        )

        Γxy = -p * (+ 1.0 * v1dd * v2xy
        + 1.0 * v1dx * v2xz
        - 1.0 * v1dy * v2xd
        - 1.0 * v1dz * v2xx
        - 1.0 * v1xd * v2dy
        + 1.0 * v1xx * v2dz
        + 1.0 * v1xy * v2dd
        - 1.0 * v1xz * v2dx
        + 1.0 * v1yd * v2zy
        - 1.0 * v1yx * v2zz
        + 1.0 * v1yy * v2zd
        + 1.0 * v1yz * v2zx
        - 1.0 * v1zd * v2yy
        + 1.0 * v1zx * v2yz
        - 1.0 * v1zy * v2yd
        - 1.0 * v1zz * v2yx
        )

        Γxz = -p * (+ 1.0 * v1dd * v2xz
        - 1.0 * v1dx * v2xy
        + 1.0 * v1dy * v2xx
        - 1.0 * v1dz * v2xd
        - 1.0 * v1xd * v2dz
        - 1.0 * v1xx * v2dy
        + 1.0 * v1xy * v2dx
        + 1.0 * v1xz * v2dd
        + 1.0 * v1yd * v2zz
        + 1.0 * v1yx * v2zy
        - 1.0 * v1yy * v2zx
        + 1.0 * v1yz * v2zd
        - 1.0 * v1zd * v2yz
        - 1.0 * v1zx * v2yy
        + 1.0 * v1zy * v2yx
        - 1.0 * v1zz * v2yd

        )

        Γyd = -p * (+ 1.0 * v1dd * v2yd
        + 1.0 * v1dx * v2yx
        + 1.0 * v1dy * v2yy
        + 1.0 * v1dz * v2yz
        - 1.0 * v1xd * v2zd
        + 1.0 * v1xx * v2zx
        + 1.0 * v1xy * v2zy
        + 1.0 * v1xz * v2zz
        + 1.0 * v1yd * v2dd
        + 1.0 * v1yx * v2dx
        + 1.0 * v1yy * v2dy
        + 1.0 * v1yz * v2dz
        + 1.0 * v1zd * v2xd
        - 1.0 * v1zx * v2xx
        - 1.0 * v1zy * v2xy
        - 1.0 * v1zz * v2xz
        )

        Γyx = -p * (+ 1.0 * v1dd * v2yx
        - 1.0 * v1dx * v2yd
        - 1.0 * v1dy * v2yz
        + 1.0 * v1dz * v2yy
        - 1.0 * v1xd * v2zx
        - 1.0 * v1xx * v2zd
        - 1.0 * v1xy * v2zz
        + 1.0 * v1xz * v2zy
        - 1.0 * v1yd * v2dx
        + 1.0 * v1yx * v2dd
        - 1.0 * v1yy * v2dz
        + 1.0 * v1yz * v2dy
        + 1.0 * v1zd * v2xx
        + 1.0 * v1zx * v2xd
        + 1.0 * v1zy * v2xz
        - 1.0 * v1zz * v2xy
        )

        Γyy = -p * (+ 1.0 * v1dd * v2yy
        + 1.0 * v1dx * v2yz
        - 1.0 * v1dy * v2yd
        - 1.0 * v1dz * v2yx
        - 1.0 * v1xd * v2zy
        + 1.0 * v1xx * v2zz
        - 1.0 * v1xy * v2zd
        - 1.0 * v1xz * v2zx
        - 1.0 * v1yd * v2dy
        + 1.0 * v1yx * v2dz
        + 1.0 * v1yy * v2dd
        - 1.0 * v1yz * v2dx
        + 1.0 * v1zd * v2xy
        - 1.0 * v1zx * v2xz
        + 1.0 * v1zy * v2xd
        + 1.0 * v1zz * v2xx
        )

        Γyz = -p * (+ 1.0 * v1dd * v2yz
        - 1.0 * v1dx * v2yy
        + 1.0 * v1dy * v2yx
        - 1.0 * v1dz * v2yd
        - 1.0 * v1xd * v2zz
        - 1.0 * v1xx * v2zy
        + 1.0 * v1xy * v2zx
        - 1.0 * v1xz * v2zd
        - 1.0 * v1yd * v2dz
        - 1.0 * v1yx * v2dy
        + 1.0 * v1yy * v2dx
        + 1.0 * v1yz * v2dd
        + 1.0 * v1zd * v2xz
        + 1.0 * v1zx * v2xy
        - 1.0 * v1zy * v2xx
        + 1.0 * v1zz * v2xd
        )

        Γzd = -p * (+ 1.0 * v1dd * v2zd
        + 1.0 * v1dx * v2zx
        + 1.0 * v1dy * v2zy
        + 1.0 * v1dz * v2zz
        + 1.0 * v1xd * v2yd
        - 1.0 * v1xx * v2yx
        - 1.0 * v1xy * v2yy
        - 1.0 * v1xz * v2yz
        - 1.0 * v1yd * v2xd
        + 1.0 * v1yx * v2xx
        + 1.0 * v1yy * v2xy
        + 1.0 * v1yz * v2xz
        + 1.0 * v1zd * v2dd
        + 1.0 * v1zx * v2dx
        + 1.0 * v1zy * v2dy
        + 1.0 * v1zz * v2dz
        )

        Γzx = -p * (+ 1.0 * v1dd * v2zx
        - 1.0 * v1dx * v2zd
        - 1.0 * v1dy * v2zz
        + 1.0 * v1dz * v2zy
        + 1.0 * v1xd * v2yx
        + 1.0 * v1xx * v2yd
        + 1.0 * v1xy * v2yz
        - 1.0 * v1xz * v2yy
        - 1.0 * v1yd * v2xx
        - 1.0 * v1yx * v2xd
        - 1.0 * v1yy * v2xz
        + 1.0 * v1yz * v2xy
        - 1.0 * v1zd * v2dx
        + 1.0 * v1zx * v2dd
        - 1.0 * v1zy * v2dz
        + 1.0 * v1zz * v2dy
        )

        Γzy = -p * (+ 1.0 * v1dd * v2zy
        + 1.0 * v1dx * v2zz
        - 1.0 * v1dy * v2zd
        - 1.0 * v1dz * v2zx
        + 1.0 * v1xd * v2yy
        - 1.0 * v1xx * v2yz
        + 1.0 * v1xy * v2yd
        + 1.0 * v1xz * v2yx
        - 1.0 * v1yd * v2xy
        + 1.0 * v1yx * v2xz
        - 1.0 * v1yy * v2xd
        - 1.0 * v1yz * v2xx
        - 1.0 * v1zd * v2dy
        + 1.0 * v1zx * v2dz
        + 1.0 * v1zy * v2dd
        - 1.0 * v1zz * v2dx
        )

        Γzz = -p * (+ 1.0 * v1dd * v2zz
        - 1.0 * v1dx * v2zy
        + 1.0 * v1dy * v2zx
        - 1.0 * v1dz * v2zd
        + 1.0 * v1xd * v2yz
        + 1.0 * v1xx * v2yy
        - 1.0 * v1xy * v2yx
        + 1.0 * v1xz * v2yd
        - 1.0 * v1yd * v2xz
        - 1.0 * v1yx * v2xy
        + 1.0 * v1yy * v2xx
        - 1.0 * v1yz * v2xd
        - 1.0 * v1zd * v2dz
        - 1.0 * v1zx * v2dy
        + 1.0 * v1zy * v2dx
        + 1.0 * v1zz * v2dd
        )
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
 
 
 
 # central kernel
 function compute_u_central!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64,
    vu   :: Float64,
    vup  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg,
    da_l :: Action_su2_hkg,
    temp :: Array{Float64, 3}
    )    :: Nothing
    # get propagator
    p = -get_propagator(Λ, v - 0.5 * u, v + 0.5 * u, m, a)
    # get buffers for left vertex
    bs1 = get_buffer_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffer_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = get_buffer_u(u, vu, v, m)
    # get buffers for right vertex
    bs2 = get_buffer_empty()
    bt2 = get_buffer_empty()
    bu2 = get_buffer_u(u, v, vup, m)
    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1,    a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, da_l, temp, 2, ch_s = false, ch_t = false)
    # compute contributions for all lattice sites
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

        # compute contributions at site i 
        Γdd = -p * (+ 1.0 * v1dd * v2dd
        - 1.0 * v1dx * v2dx
        - 1.0 * v1dy * v2dy
        - 1.0 * v1dz * v2dz
        - 1.0 * v1xd * v2xd
        + 1.0 * v1xx * v2xx
        + 1.0 * v1xy * v2xy
        + 1.0 * v1xz * v2xz
        - 1.0 * v1yd * v2yd
        + 1.0 * v1yx * v2yx
        + 1.0 * v1yy * v2yy
        + 1.0 * v1yz * v2yz
        - 1.0 * v1zd * v2zd
        + 1.0 * v1zx * v2zx
        + 1.0 * v1zy * v2zy
        + 1.0 * v1zz * v2zz
        )

        Γdx = -p * (+ 1.0 * v1dd * v2dx
        + 1.0 * v1dx * v2dd
        - 1.0 * v1dy * v2dz
        + 1.0 * v1dz * v2dy
        + 1.0 * v1xd * v2xx
        + 1.0 * v1xx * v2xd
        + 1.0 * v1xy * v2xz
        - 1.0 * v1xz * v2xy
        + 1.0 * v1yd * v2yx
        + 1.0 * v1yx * v2yd
        + 1.0 * v1yy * v2yz
        - 1.0 * v1yz * v2yy
        + 1.0 * v1zd * v2zx
        + 1.0 * v1zx * v2zd
        + 1.0 * v1zy * v2zz
        - 1.0 * v1zz * v2zy
        )

        Γdy = -p * (+ 1.0 * v1dd * v2dy
        + 1.0 * v1dx * v2dz
        + 1.0 * v1dy * v2dd
        - 1.0 * v1dz * v2dx
        + 1.0 * v1xd * v2xy
        - 1.0 * v1xx * v2xz
        + 1.0 * v1xy * v2xd
        + 1.0 * v1xz * v2xx
        + 1.0 * v1yd * v2yy
        - 1.0 * v1yx * v2yz
        + 1.0 * v1yy * v2yd
        + 1.0 * v1yz * v2yx
        + 1.0 * v1zd * v2zy
        - 1.0 * v1zx * v2zz
        + 1.0 * v1zy * v2zd
        + 1.0 * v1zz * v2zx
        )

        Γdz = -p * (+ 1.0 * v1dd * v2dz
        - 1.0 * v1dx * v2dy
        + 1.0 * v1dy * v2dx
        + 1.0 * v1dz * v2dd
        + 1.0 * v1xd * v2xz
        + 1.0 * v1xx * v2xy
        - 1.0 * v1xy * v2xx
        + 1.0 * v1xz * v2xd
        + 1.0 * v1yd * v2yz
        + 1.0 * v1yx * v2yy
        - 1.0 * v1yy * v2yx
        + 1.0 * v1yz * v2yd
        + 1.0 * v1zd * v2zz
        + 1.0 * v1zx * v2zy
        - 1.0 * v1zy * v2zx
        + 1.0 * v1zz * v2zd
        )

        Γxd = -p * (+ 1.0 * v1dd * v2xd
        + 1.0 * v1dx * v2xx
        + 1.0 * v1dy * v2xy
        + 1.0 * v1dz * v2xz
        + 1.0 * v1xd * v2dd
        + 1.0 * v1xx * v2dx
        + 1.0 * v1xy * v2dy
        + 1.0 * v1xz * v2dz
        + 1.0 * v1yd * v2zd
        - 1.0 * v1yx * v2zx
        - 1.0 * v1yy * v2zy
        - 1.0 * v1yz * v2zz
        - 1.0 * v1zd * v2yd
        + 1.0 * v1zx * v2yx
        + 1.0 * v1zy * v2yy
        + 1.0 * v1zz * v2yz
        )

        Γxx = -p * (+ 1.0 * v1dd * v2xx
        - 1.0 * v1dx * v2xd
        - 1.0 * v1dy * v2xz
        + 1.0 * v1dz * v2xy
        - 1.0 * v1xd * v2dx
        + 1.0 * v1xx * v2dd
        - 1.0 * v1xy * v2dz
        + 1.0 * v1xz * v2dy
        + 1.0 * v1yd * v2zx
        + 1.0 * v1yx * v2zd
        + 1.0 * v1yy * v2zz
        - 1.0 * v1yz * v2zy
        - 1.0 * v1zd * v2yx
        - 1.0 * v1zx * v2yd
        - 1.0 * v1zy * v2yz
        + 1.0 * v1zz * v2yy
        )

        Γxy = -p * (+ 1.0 * v1dd * v2xy
        + 1.0 * v1dx * v2xz
        - 1.0 * v1dy * v2xd
        - 1.0 * v1dz * v2xx
        - 1.0 * v1xd * v2dy
        + 1.0 * v1xx * v2dz
        + 1.0 * v1xy * v2dd
        - 1.0 * v1xz * v2dx
        + 1.0 * v1yd * v2zy
        - 1.0 * v1yx * v2zz
        + 1.0 * v1yy * v2zd
        + 1.0 * v1yz * v2zx
        - 1.0 * v1zd * v2yy
        + 1.0 * v1zx * v2yz
        - 1.0 * v1zy * v2yd
        - 1.0 * v1zz * v2yx
        )

        Γxz = -p * (+ 1.0 * v1dd * v2xz
        - 1.0 * v1dx * v2xy
        + 1.0 * v1dy * v2xx
        - 1.0 * v1dz * v2xd
        - 1.0 * v1xd * v2dz
        - 1.0 * v1xx * v2dy
        + 1.0 * v1xy * v2dx
        + 1.0 * v1xz * v2dd
        + 1.0 * v1yd * v2zz
        + 1.0 * v1yx * v2zy
        - 1.0 * v1yy * v2zx
        + 1.0 * v1yz * v2zd
        - 1.0 * v1zd * v2yz
        - 1.0 * v1zx * v2yy
        + 1.0 * v1zy * v2yx
        - 1.0 * v1zz * v2yd

        )

        Γyd = -p * (+ 1.0 * v1dd * v2yd
        + 1.0 * v1dx * v2yx
        + 1.0 * v1dy * v2yy
        + 1.0 * v1dz * v2yz
        - 1.0 * v1xd * v2zd
        + 1.0 * v1xx * v2zx
        + 1.0 * v1xy * v2zy
        + 1.0 * v1xz * v2zz
        + 1.0 * v1yd * v2dd
        + 1.0 * v1yx * v2dx
        + 1.0 * v1yy * v2dy
        + 1.0 * v1yz * v2dz
        + 1.0 * v1zd * v2xd
        - 1.0 * v1zx * v2xx
        - 1.0 * v1zy * v2xy
        - 1.0 * v1zz * v2xz
        )

        Γyx = -p * (+ 1.0 * v1dd * v2yx
        - 1.0 * v1dx * v2yd
        - 1.0 * v1dy * v2yz
        + 1.0 * v1dz * v2yy
        - 1.0 * v1xd * v2zx
        - 1.0 * v1xx * v2zd
        - 1.0 * v1xy * v2zz
        + 1.0 * v1xz * v2zy
        - 1.0 * v1yd * v2dx
        + 1.0 * v1yx * v2dd
        - 1.0 * v1yy * v2dz
        + 1.0 * v1yz * v2dy
        + 1.0 * v1zd * v2xx
        + 1.0 * v1zx * v2xd
        + 1.0 * v1zy * v2xz
        - 1.0 * v1zz * v2xy
        )

        Γyy = -p * (+ 1.0 * v1dd * v2yy
        + 1.0 * v1dx * v2yz
        - 1.0 * v1dy * v2yd
        - 1.0 * v1dz * v2yx
        - 1.0 * v1xd * v2zy
        + 1.0 * v1xx * v2zz
        - 1.0 * v1xy * v2zd
        - 1.0 * v1xz * v2zx
        - 1.0 * v1yd * v2dy
        + 1.0 * v1yx * v2dz
        + 1.0 * v1yy * v2dd
        - 1.0 * v1yz * v2dx
        + 1.0 * v1zd * v2xy
        - 1.0 * v1zx * v2xz
        + 1.0 * v1zy * v2xd
        + 1.0 * v1zz * v2xx
        )

        Γyz = -p * (+ 1.0 * v1dd * v2yz
        - 1.0 * v1dx * v2yy
        + 1.0 * v1dy * v2yx
        - 1.0 * v1dz * v2yd
        - 1.0 * v1xd * v2zz
        - 1.0 * v1xx * v2zy
        + 1.0 * v1xy * v2zx
        - 1.0 * v1xz * v2zd
        - 1.0 * v1yd * v2dz
        - 1.0 * v1yx * v2dy
        + 1.0 * v1yy * v2dx
        + 1.0 * v1yz * v2dd
        + 1.0 * v1zd * v2xz
        + 1.0 * v1zx * v2xy
        - 1.0 * v1zy * v2xx
        + 1.0 * v1zz * v2xd
        )

        Γzd = -p * (+ 1.0 * v1dd * v2zd
        + 1.0 * v1dx * v2zx
        + 1.0 * v1dy * v2zy
        + 1.0 * v1dz * v2zz
        + 1.0 * v1xd * v2yd
        - 1.0 * v1xx * v2yx
        - 1.0 * v1xy * v2yy
        - 1.0 * v1xz * v2yz
        - 1.0 * v1yd * v2xd
        + 1.0 * v1yx * v2xx
        + 1.0 * v1yy * v2xy
        + 1.0 * v1yz * v2xz
        + 1.0 * v1zd * v2dd
        + 1.0 * v1zx * v2dx
        + 1.0 * v1zy * v2dy
        + 1.0 * v1zz * v2dz
        )

        Γzx = -p * (+ 1.0 * v1dd * v2zx
        - 1.0 * v1dx * v2zd
        - 1.0 * v1dy * v2zz
        + 1.0 * v1dz * v2zy
        + 1.0 * v1xd * v2yx
        + 1.0 * v1xx * v2yd
        + 1.0 * v1xy * v2yz
        - 1.0 * v1xz * v2yy
        - 1.0 * v1yd * v2xx
        - 1.0 * v1yx * v2xd
        - 1.0 * v1yy * v2xz
        + 1.0 * v1yz * v2xy
        - 1.0 * v1zd * v2dx
        + 1.0 * v1zx * v2dd
        - 1.0 * v1zy * v2dz
        + 1.0 * v1zz * v2dy
        )

        Γzy = -p * (+ 1.0 * v1dd * v2zy
        + 1.0 * v1dx * v2zz
        - 1.0 * v1dy * v2zd
        - 1.0 * v1dz * v2zx
        + 1.0 * v1xd * v2yy
        - 1.0 * v1xx * v2yz
        + 1.0 * v1xy * v2yd
        + 1.0 * v1xz * v2yx
        - 1.0 * v1yd * v2xy
        + 1.0 * v1yx * v2xz
        - 1.0 * v1yy * v2xd
        - 1.0 * v1yz * v2xx
        - 1.0 * v1zd * v2dy
        + 1.0 * v1zx * v2dz
        + 1.0 * v1zy * v2dd
        - 1.0 * v1zz * v2dx
        )

        Γzz = -p * (+ 1.0 * v1dd * v2zz
        - 1.0 * v1dx * v2zy
        + 1.0 * v1dy * v2zx
        - 1.0 * v1dz * v2zd
        + 1.0 * v1xd * v2yz
        + 1.0 * v1xx * v2yy
        - 1.0 * v1xy * v2yx
        + 1.0 * v1xz * v2yd
        - 1.0 * v1yd * v2xz
        - 1.0 * v1yx * v2xy
        + 1.0 * v1yy * v2xx
        - 1.0 * v1yz * v2xd
        - 1.0 * v1zd * v2dz
        - 1.0 * v1zx * v2dy
        + 1.0 * v1zy * v2dx
        + 1.0 * v1zz * v2dd
        )
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
