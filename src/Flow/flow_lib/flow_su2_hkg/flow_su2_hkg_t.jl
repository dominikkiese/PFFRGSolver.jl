# Katanin kernel
function compute_t_kat!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg,
    da   :: Action_su2_hkg,
    temp :: Array{Float64, 3}
    )    :: Nothing
    # get propagator and overlap
    p       = get_propagator_kat(Λ, v + 0.5 * t, v - 0.5 * t, m, a, da) + get_propagator_kat(Λ, v - 0.5 * t, v + 0.5 * t, m, a, da)
    overlap = r.overlap
    # get buffers for left non-local vertex
    bs1 = get_buffer_s( v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_t(t, vt, v, m)
    bu1 = get_buffer_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * ( t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffer_t(t, v, vtp, m)
    bu2 = get_buffer_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * ( t + v + vtp), m)

    # get buffers for local left vertex
    bs3 = get_buffer_s( v + vt, 0.5 * (t + v - vt), 0.5 * (-t + v - vt), m)
    bt3 = get_buffer_t(-v + vt, 0.5 * (t + v + vt), 0.5 * (-t + v + vt), m)
    bu3 = get_buffer_u(t, v, vt, m)

    # get buffers for local right vertex
    bs4 = get_buffer_s(v + vtp, 0.5 * (t - v + vtp), 0.5 * (-t - v + vtp), m)
    bt4 = get_buffer_t(v - vtp, 0.5 * (t + v + vtp), 0.5 * (-t + v + vtp), m)
    bu4 = get_buffer_u(t, vtp, v, m)

    # cache local vertex values
    v3xx, v3yy, v3zz, v3xy, v3xz, v3yz, v3yx, v3zx, v3zy, v3dd, v3xd, v3yd, v3zd, v3dx, v3dy, v3dz = get_Γ(1, bs3, bt3, bu3, r, a)
    v4xx, v4yy, v4zz, v4xy, v4xz, v4yz, v4yx, v4zx, v4zy, v4dd, v4xd, v4yd, v4zd, v4dx, v4dy, v4dz = get_Γ(1, bs4, bt4, bu4, r, a)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)
    
    # compute contributions for all lattice sites 
    for i in eachindex(r.sites)
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
        Γxx = -p * (-v1xd * v4dx - v1xd * v4xd + v1xd * v4yz - v1xd * v4zy + v1xx * v4dd + v1xx * v4xx - v1xx * v4yy - v1xx * v4zz + v1xy * v4dz + v1xy * v4xy + v1xy * v4yx - v1xy * v4zd - v1xz * v4dy + v1xz * v4xz + v1xz * v4yd + v1xz * v4zx +
                   v3dd * v2xx - v3dx * v2dx + v3dy * v2zx - v3dz * v2yx - v3xd * v2dx + v3xx * v2xx + v3xy * v2yx + v3xz * v2zx - v3yd * v2zx + v3yx * v2yx - v3yy * v2xx - v3yz * v2dx + v3zd * v2yx + v3zx * v2zx + v3zy * v2dx - v3zz * v2xx)

        Γyy = -p * (-v1yd * v4dy - v1yd * v4xz - v1yd * v4yd + v1yd * v4zx - v1yx * v4dz + v1yx * v4xy + v1yx * v4yx + v1yx * v4zd + v1yy * v4dd - v1yy * v4xx + v1yy * v4yy - v1yy * v4zz + v1yz * v4dx - v1yz * v4xd + v1yz * v4yz + v1yz * v4zy +
                   v3dd * v2yy - v3dx * v2zy - v3dy * v2dy + v3dz * v2xy + v3xd * v2zy - v3xx * v2yy + v3xy * v2xy + v3xz * v2dy - v3yd * v2dy + v3yx * v2xy + v3yy * v2yy + v3yz * v2zy - v3zd * v2xy - v3zx * v2dy + v3zy * v2zy - v3zz * v2yy)

        Γzz = -p * (-v1zd * v4dz + v1zd * v4xy - v1zd * v4yx - v1zd * v4zd + v1zx * v4dy + v1zx * v4xz - v1zx * v4yd + v1zx * v4zx - v1zy * v4dx + v1zy * v4xd + v1zy * v4yz + v1zy * v4zy + v1zz * v4dd - v1zz * v4xx - v1zz * v4yy + v1zz * v4zz +
                   v3dd * v2zz + v3dx * v2yz - v3dy * v2xz - v3dz * v2dz - v3xd * v2yz - v3xx * v2zz - v3xy * v2dz + v3xz * v2xz + v3yd * v2xz + v3yx * v2dz - v3yy * v2zz + v3yz * v2yz - v3zd * v2dz + v3zx * v2xz + v3zy * v2yz + v3zz * v2zz)

        Γxy = -p * (-v1xd * v4dy - v1xd * v4xz - v1xd * v4yd + v1xd * v4zx - v1xx * v4dz + v1xx * v4xy + v1xx * v4yx + v1xx * v4zd + v1xy * v4dd - v1xy * v4xx + v1xy * v4yy - v1xy * v4zz + v1xz * v4dx - v1xz * v4xd + v1xz * v4yz + v1xz * v4zy +
                   v3dd * v2xy - v3dx * v2dy + v3dy * v2zy - v3dz * v2yy - v3xd * v2dy + v3xx * v2xy + v3xy * v2yy + v3xz * v2zy - v3yd * v2zy + v3yx * v2yy - v3yy * v2xy - v3yz * v2dy + v3zd * v2yy + v3zx * v2zy + v3zy * v2dy - v3zz * v2xy)

        Γxz = -p * (-v1xd * v4dz + v1xd * v4xy - v1xd * v4yx - v1xd * v4zd + v1xx * v4dy + v1xx * v4xz - v1xx * v4yd + v1xx * v4zx - v1xy * v4dx + v1xy * v4xd + v1xy * v4yz + v1xy * v4zy + v1xz * v4dd - v1xz * v4xx - v1xz * v4yy + v1xz * v4zz +
                   v3dd * v2xz - v3dx * v2dz + v3dy * v2zz - v3dz * v2yz - v3xd * v2dz + v3xx * v2xz + v3xy * v2yz + v3xz * v2zz - v3yd * v2zz + v3yx * v2yz - v3yy * v2xz - v3yz * v2dz + v3zd * v2yz + v3zx * v2zz + v3zy * v2dz - v3zz * v2xz) 

        Γyz = -p * (-v1yd * v4dz + v1yd * v4xy - v1yd * v4yx - v1yd * v4zd + v1yx * v4dy + v1yx * v4xz - v1yx * v4yd + v1yx * v4zx - v1yy * v4dx + v1yy * v4xd + v1yy * v4yz + v1yy * v4zy + v1yz * v4dd - v1yz * v4xx - v1yz * v4yy + v1yz * v4zz +
                   v3dd * v2yz - v3dx * v2zz - v3dy * v2dz + v3dz * v2xz + v3xd * v2zz - v3xx * v2yz + v3xy * v2xz + v3xz * v2dz - v3yd * v2dz + v3yx * v2xz + v3yy * v2yz + v3yz * v2zz - v3zd * v2xz - v3zx * v2dz + v3zy * v2zz - v3zz * v2yz) 

        Γyx = -p * (-v1yd * v4dx - v1yd * v4xd + v1yd * v4yz - v1yd * v4zy + v1yx * v4dd + v1yx * v4xx - v1yx * v4yy - v1yx * v4zz + v1yy * v4dz + v1yy * v4xy + v1yy * v4yx - v1yy * v4zd - v1yz * v4dy + v1yz * v4xz + v1yz * v4yd + v1yz * v4zx +
                   v3dd * v2yx - v3dx * v2zx - v3dy * v2dx + v3dz * v2xx + v3xd * v2zx - v3xx * v2yx + v3xy * v2xx + v3xz * v2dx - v3yd * v2dx + v3yx * v2xx + v3yy * v2yx + v3yz * v2zx - v3zd * v2xx - v3zx * v2dx + v3zy * v2zx - v3zz * v2yx)

        Γzx = -p * (-v1zd * v4dx - v1zd * v4xd + v1zd * v4yz - v1zd * v4zy + v1zx * v4dd + v1zx * v4xx - v1zx * v4yy - v1zx * v4zz + v1zy * v4dz + v1zy * v4xy + v1zy * v4yx - v1zy * v4zd - v1zz * v4dy + v1zz * v4xz + v1zz * v4yd + v1zz * v4zx + 
                   v3dd * v2zx +  v3dx * v2yx - v3dy * v2xx - v3dz * v2dx - v3xd * v2yx - v3xx * v2zx - v3xy * v2dx + v3xz * v2xx + v3yd * v2xx + v3yx * v2dx - v3yy * v2zx + v3yz * v2yx - v3zd * v2dx + v3zx * v2xx + v3zy * v2yx + v3zz * v2zx)

        Γzy = -p * (-v1zd * v4dy - v1zd * v4xz - v1zd * v4yd + v1zd * v4zx - v1zx * v4dz + v1zx * v4xy + v1zx * v4yx + v1zx * v4zd + v1zy * v4dd - v1zy * v4xx + v1zy * v4yy - v1zy * v4zz + v1zz * v4dx - v1zz * v4xd + v1zz * v4yz + v1zz * v4zy +
                   v3dd * v2zy + v3dx * v2yy - v3dy * v2xy - v3dz * v2dy - v3xd * v2yy - v3xx * v2zy - v3xy * v2dy + v3xz * v2xy + v3yd * v2xy + v3yx * v2dy - v3yy * v2zy + v3yz * v2yy - v3zd * v2dy + v3zx * v2xy + v3zy * v2yy + v3zz * v2zy)

        Γdd = -p * ( v1dd * v4dd + v1dd * v4xx + v1dd * v4yy + v1dd * v4zz - v1dx * v4dx - v1dx * v4xd - v1dx * v4yz + v1dx * v4zy - v1dy * v4dy + v1dy * v4xz - v1dy * v4yd - v1dy * v4zx - v1dz * v4dz - v1dz * v4xy + v1dz * v4yx - v1dz * v4zd +
                   v3dd * v2dd - v3dx * v2xd - v3dy * v2yd - v3dz * v2zd - v3xd * v2xd + v3xx * v2dd + v3xy * v2zd - v3xz * v2yd - v3yd * v2yd - v3yx * v2zd + v3yy * v2dd + v3yz * v2xd - v3zd * v2zd + v3zx * v2yd - v3zy * v2xd + v3zz * v2dd)

        Γxd = -p * ( v1xd * v4dd + v1xd * v4xx + v1xd * v4yy + v1xd * v4zz + v1xx * v4dx + v1xx * v4xd + v1xx * v4yz - v1xx * v4zy + v1xy * v4dy - v1xy * v4xz + v1xy * v4yd + v1xy * v4zx + v1xz * v4dz + v1xz * v4xy - v1xz * v4yx + v1xz * v4zd +
                   v3dd * v2xd + v3dx * v2dd + v3dy * v2zd - v3dz * v2yd + v3xd * v2dd + v3xx * v2xd + v3xy * v2yd + v3xz * v2zd - v3yd * v2zd + v3yx * v2yd - v3yy * v2xd + v3yz * v2dd + v3zd * v2yd + v3zx * v2zd - v3zy * v2dd - v3zz * v2xd)

        Γyd = -p * ( v1yd * v4dd + v1yd * v4xx + v1yd * v4yy + v1yd * v4zz + v1yx * v4dx + v1yx * v4xd + v1yx * v4yz - v1yx * v4zy + v1yy * v4dy - v1yy * v4xz + v1yy * v4yd + v1yy * v4zx + v1yz * v4dz + v1yz * v4xy - v1yz * v4yx + v1yz * v4zd +
                   v3dd * v2yd - v3dx * v2zd + v3dy * v2dd + v3dz * v2xd + v3xd * v2zd - v3xx * v2yd + v3xy * v2xd - v3xz * v2dd + v3yd * v2dd + v3yx * v2xd + v3yy * v2yd + v3yz * v2zd - v3zd * v2xd + v3zx * v2dd + v3zy * v2zd - v3zz * v2yd)

        Γzd = -p * ( v1zd * v4dd + v1zd * v4xx + v1zd * v4yy + v1zd * v4zz + v1zx * v4dx + v1zx * v4xd + v1zx * v4yz - v1zx * v4zy + v1zy * v4dy - v1zy * v4xz + v1zy * v4yd + v1zy * v4zx + v1zz * v4dz + v1zz * v4xy - v1zz * v4yx + v1zz * v4zd +
                   v3dd * v2zd + v3dx * v2yd - v3dy * v2xd + v3dz * v2dd - v3xd * v2yd - v3xx * v2zd + v3xy * v2dd + v3xz * v2xd + v3yd * v2xd - v3yx * v2dd - v3yy * v2zd + v3yz * v2yd + v3zd * v2dd + v3zx * v2xd + v3zy * v2yd + v3zz * v2zd)

        Γdx = -p * ( v1dd * v4dx + v1dd * v4xd - v1dd * v4yz + v1dd * v4zy + v1dx * v4dd + v1dx * v4xx - v1dx * v4yy - v1dx * v4zz + v1dy * v4dz + v1dy * v4xy + v1dy * v4yx - v1dy * v4zd - v1dz * v4dy + v1dz * v4xz + v1dz * v4yd + v1dz * v4zx +
                   v3dd * v2dx + v3dx * v2xx + v3dy * v2yx + v3dz * v2zx + v3xd * v2xx + v3xx * v2dx - v3xy * v2zx + v3xz * v2yx + v3yd * v2yx + v3yx * v2zx + v3yy * v2dx - v3yz * v2xx + v3zd * v2zx - v3zx * v2yx + v3zy * v2xx + v3zz * v2dx)

        Γdy = -p * ( v1dd * v4dy + v1dd * v4xz + v1dd * v4yd - v1dd * v4zx - v1dx * v4dz + v1dx * v4xy + v1dx * v4yx + v1dx * v4zd + v1dy * v4dd - v1dy * v4xx + v1dy * v4yy - v1dy * v4zz + v1dz * v4dx - v1dz * v4xd + v1dz * v4yz + v1dz * v4zy +
                   v3dd * v2dy + v3dx * v2xy + v3dy * v2yy + v3dz * v2zy + v3xd * v2xy + v3xx * v2dy - v3xy * v2zy + v3xz * v2yy + v3yd * v2yy + v3yx * v2zy + v3yy * v2dy - v3yz * v2xy + v3zd * v2zy - v3zx * v2yy + v3zy * v2xy + v3zz * v2dy)

        Γdz = -p * ( v1dd * v4dz - v1dd * v4xy + v1dd * v4yx + v1dd * v4zd + v1dx * v4dy + v1dx * v4xz - v1dx * v4yd + v1dx * v4zx - v1dy * v4dx + v1dy * v4xd + v1dy * v4yz + v1dy * v4zy + v1dz * v4dd - v1dz * v4xx - v1dz * v4yy + v1dz * v4zz +
                   v3dd * v2dz + v3dx * v2xz + v3dy * v2yz + v3dz * v2zz + v3xd * v2xz + v3xx * v2dz - v3xy * v2zz + v3xz * v2yz + v3yd * v2yz + v3yx * v2zz + v3yy * v2dz - v3yz * v2xz + v3zd * v2zz - v3zx * v2yz + v3zy * v2xz + v3zz * v2dz)
                   

        # determine overlap for site i
        overlap_i = overlap[i]
        # determine range for inner sum
        Range = size(overlap_i, 1)
        #compute inner sum 
        @turbo unroll = 1 for j in 1 : Range
            # read cached values for inner site
            v1xx = temp[overlap_i[j, 1],  1, 1]
            v1yy = temp[overlap_i[j, 1],  2, 1]
            v1zz = temp[overlap_i[j, 1],  3, 1]
            v1xy = temp[overlap_i[j, 1],  4, 1]
            v1xz = temp[overlap_i[j, 1],  5, 1]
            v1yz = temp[overlap_i[j, 1],  6, 1]
            v1yx = temp[overlap_i[j, 1],  7, 1]
            v1zx = temp[overlap_i[j, 1],  8, 1]
            v1zy = temp[overlap_i[j, 1],  9, 1]
            v1dd = temp[overlap_i[j, 1], 10, 1]
            v1xd = temp[overlap_i[j, 1], 11, 1]
            v1yd = temp[overlap_i[j, 1], 12, 1]
            v1zd = temp[overlap_i[j, 1], 13, 1]
            v1dx = temp[overlap_i[j, 1], 14, 1]
            v1dy = temp[overlap_i[j, 1], 15, 1]
            v1dz = temp[overlap_i[j, 1], 16, 1]

            v2xx = temp[overlap_i[j, 2],  1, 2]
            v2yy = temp[overlap_i[j, 2],  2, 2]
            v2zz = temp[overlap_i[j, 2],  3, 2]
            v2xy = temp[overlap_i[j, 2],  4, 2]
            v2xz = temp[overlap_i[j, 2],  5, 2]
            v2yz = temp[overlap_i[j, 2],  6, 2]
            v2yx = temp[overlap_i[j, 2],  7, 2]
            v2zx = temp[overlap_i[j, 2],  8, 2]
            v2zy = temp[overlap_i[j, 2],  9, 2]
            v2dd = temp[overlap_i[j, 2], 10, 2]
            v2xd = temp[overlap_i[j, 2], 11, 2]
            v2yd = temp[overlap_i[j, 2], 12, 2]
            v2zd = temp[overlap_i[j, 2], 13, 2]
            v2dx = temp[overlap_i[j, 2], 14, 2]
            v2dy = temp[overlap_i[j, 2], 15, 2]
            v2dz = temp[overlap_i[j, 2], 16, 2]

            # compute contribution at inner site
            Γxx += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dx - 2.0 * v1xx * v2xx - 2.0 * v1xy * v2yx - 2.0 * v1xz * v2zx)
            Γyy += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dy - 2.0 * v1yx * v2xy - 2.0 * v1yy * v2yy - 2.0 * v1yz * v2zy)
            Γzz += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dz - 2.0 * v1zx * v2xz - 2.0 * v1zy * v2yz - 2.0 * v1zz * v2zz)
            Γxy += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dy - 2.0 * v1xx * v2xy - 2.0 * v1xy * v2yy - 2.0 * v1xz * v2zy)
            Γxz += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dz - 2.0 * v1xx * v2xz - 2.0 * v1xy * v2yz - 2.0 * v1xz * v2zz)
            Γyz += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dz - 2.0 * v1yx * v2xz - 2.0 * v1yy * v2yz - 2.0 * v1yz * v2zz)
            Γyx += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dx - 2.0 * v1yx * v2xx - 2.0 * v1yy * v2yx - 2.0 * v1yz * v2zx)
            Γzx += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dx - 2.0 * v1zx * v2xx - 2.0 * v1zy * v2yx - 2.0 * v1zz * v2zx)
            Γzy += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dy - 2.0 * v1zx * v2xy - 2.0 * v1zy * v2yy - 2.0 * v1zz * v2zy)
            Γdd += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dd + 2.0 * v1dx * v2xd + 2.0 * v1dy * v2yd + 2.0 * v1dz * v2zd)
            Γxd += -p * overlap_i[j, 3] * (- 2.0 * v1xd * v2dd - 2.0 * v1xx * v2xd - 2.0 * v1xy * v2yd - 2.0 * v1xz * v2zd)
            Γyd += -p * overlap_i[j, 3] * (- 2.0 * v1yd * v2dd - 2.0 * v1yx * v2xd - 2.0 * v1yy * v2yd - 2.0 * v1yz * v2zd)
            Γzd += -p * overlap_i[j, 3] * (- 2.0 * v1zd * v2dd - 2.0 * v1zx * v2xd - 2.0 * v1zy * v2yd - 2.0 * v1zz * v2zd)
            Γdx += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dx - 2.0 * v1dx * v2xx - 2.0 * v1dy * v2yx - 2.0 * v1dz * v2zx)
            Γdy += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dy - 2.0 * v1dx * v2xy - 2.0 * v1dy * v2yy - 2.0 * v1dz * v2zy)
            Γdz += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dz - 2.0 * v1dx * v2xz - 2.0 * v1dy * v2yz - 2.0 * v1dz * v2zz)
        end 
        # parse result to output buffer 
        buff[1 , i] += dv * Γxx
        buff[2 , i] += dv * Γyy
        buff[3 , i] += dv * Γzz
        buff[4 , i] += dv * Γxy
        buff[5 , i] += dv * Γxz
        buff[6 , i] += dv * Γyz
        buff[7 , i] += dv * Γyx
        buff[8 , i] += dv * Γzx
        buff[9 , i] += dv * Γzy
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
 function compute_t_left!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg,
    da   :: Action_su2_hkg,
    temp :: Array{Float64, 3}
    )    :: Nothing
    # get propagator and overlap
    p       = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = get_buffer_s( v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_empty()
    bu1 = get_buffer_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * ( t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffer_t(t, v, vtp, m)
    bu2 = get_buffer_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * ( t + v + vtp), m)

    # get buffers for local left vertex
    bs3 = get_buffer_s( v + vt, 0.5 * (t + v - vt), 0.5 * (-t + v - vt), m)
    bt3 = get_buffer_t(-v + vt, 0.5 * (t + v + vt), 0.5 * (-t + v + vt), m)
    bu3 = get_buffer_empty()
    
    # get buffers for local right vertex
    bs4 = get_buffer_s(v + vtp, 0.5 * (t - v + vtp), 0.5 * (-t - v + vtp), m)
    bt4 = get_buffer_t(v - vtp, 0.5 * (t + v + vtp), 0.5 * (-t + v + vtp), m)
    bu4 = get_buffer_u(t, vtp, v, m)

    # cache local vertex values
    v3xx, v3yy, v3zz, v3xy, v3xz, v3yz, v3yx, v3zx, v3zy, v3dd, v3xd, v3yd, v3zd, v3dx, v3dy, v3dz = get_Γ(1, bs3, bt3, bu3, r, da, ch_u = false)
    v4xx, v4yy, v4zz, v4xy, v4xz, v4yz, v4yx, v4zx, v4zy, v4dd, v4xd, v4yd, v4zd, v4dx, v4dy, v4dz = get_Γ(1, bs4, bt4, bu4, r,  a)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, da, temp, 1, ch_t = false)
    get_Γ_avx!(r, bs2, bt2, bu2,  a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
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
        Γxx = -p * (-v1xd * v4dx - v1xd * v4xd + v1xd * v4yz - v1xd * v4zy + v1xx * v4dd + v1xx * v4xx - v1xx * v4yy - v1xx * v4zz + v1xy * v4dz + v1xy * v4xy + v1xy * v4yx - v1xy * v4zd - v1xz * v4dy + v1xz * v4xz + v1xz * v4yd + v1xz * v4zx +
                   v3dd * v2xx - v3dx * v2dx + v3dy * v2zx - v3dz * v2yx - v3xd * v2dx + v3xx * v2xx + v3xy * v2yx + v3xz * v2zx - v3yd * v2zx + v3yx * v2yx - v3yy * v2xx - v3yz * v2dx + v3zd * v2yx + v3zx * v2zx + v3zy * v2dx - v3zz * v2xx)

        Γyy = -p * (-v1yd * v4dy - v1yd * v4xz - v1yd * v4yd + v1yd * v4zx - v1yx * v4dz + v1yx * v4xy + v1yx * v4yx + v1yx * v4zd + v1yy * v4dd - v1yy * v4xx + v1yy * v4yy - v1yy * v4zz + v1yz * v4dx - v1yz * v4xd + v1yz * v4yz + v1yz * v4zy +
                   v3dd * v2yy - v3dx * v2zy - v3dy * v2dy + v3dz * v2xy + v3xd * v2zy - v3xx * v2yy + v3xy * v2xy + v3xz * v2dy - v3yd * v2dy + v3yx * v2xy + v3yy * v2yy + v3yz * v2zy - v3zd * v2xy - v3zx * v2dy + v3zy * v2zy - v3zz * v2yy)

        Γzz = -p * (-v1zd * v4dz + v1zd * v4xy - v1zd * v4yx - v1zd * v4zd + v1zx * v4dy + v1zx * v4xz - v1zx * v4yd + v1zx * v4zx - v1zy * v4dx + v1zy * v4xd + v1zy * v4yz + v1zy * v4zy + v1zz * v4dd - v1zz * v4xx - v1zz * v4yy + v1zz * v4zz +
                   v3dd * v2zz + v3dx * v2yz - v3dy * v2xz - v3dz * v2dz - v3xd * v2yz - v3xx * v2zz - v3xy * v2dz + v3xz * v2xz + v3yd * v2xz + v3yx * v2dz - v3yy * v2zz + v3yz * v2yz - v3zd * v2dz + v3zx * v2xz + v3zy * v2yz + v3zz * v2zz)

        Γxy = -p * (-v1xd * v4dy - v1xd * v4xz - v1xd * v4yd + v1xd * v4zx - v1xx * v4dz + v1xx * v4xy + v1xx * v4yx + v1xx * v4zd + v1xy * v4dd - v1xy * v4xx + v1xy * v4yy - v1xy * v4zz + v1xz * v4dx - v1xz * v4xd + v1xz * v4yz + v1xz * v4zy +
                   v3dd * v2xy - v3dx * v2dy + v3dy * v2zy - v3dz * v2yy - v3xd * v2dy + v3xx * v2xy + v3xy * v2yy + v3xz * v2zy - v3yd * v2zy + v3yx * v2yy - v3yy * v2xy - v3yz * v2dy + v3zd * v2yy + v3zx * v2zy + v3zy * v2dy - v3zz * v2xy)

        Γxz = -p * (-v1xd * v4dz + v1xd * v4xy - v1xd * v4yx - v1xd * v4zd + v1xx * v4dy + v1xx * v4xz - v1xx * v4yd + v1xx * v4zx - v1xy * v4dx + v1xy * v4xd + v1xy * v4yz + v1xy * v4zy + v1xz * v4dd - v1xz * v4xx - v1xz * v4yy + v1xz * v4zz +
                   v3dd * v2xz - v3dx * v2dz + v3dy * v2zz - v3dz * v2yz - v3xd * v2dz + v3xx * v2xz + v3xy * v2yz + v3xz * v2zz - v3yd * v2zz + v3yx * v2yz - v3yy * v2xz - v3yz * v2dz + v3zd * v2yz + v3zx * v2zz + v3zy * v2dz - v3zz * v2xz) 

        Γyz = -p * (-v1yd * v4dz + v1yd * v4xy - v1yd * v4yx - v1yd * v4zd + v1yx * v4dy + v1yx * v4xz - v1yx * v4yd + v1yx * v4zx - v1yy * v4dx + v1yy * v4xd + v1yy * v4yz + v1yy * v4zy + v1yz * v4dd - v1yz * v4xx - v1yz * v4yy + v1yz * v4zz +
                   v3dd * v2yz - v3dx * v2zz - v3dy * v2dz + v3dz * v2xz + v3xd * v2zz - v3xx * v2yz + v3xy * v2xz + v3xz * v2dz - v3yd * v2dz + v3yx * v2xz + v3yy * v2yz + v3yz * v2zz - v3zd * v2xz - v3zx * v2dz + v3zy * v2zz - v3zz * v2yz) 

        Γyx = -p * (-v1yd * v4dx - v1yd * v4xd + v1yd * v4yz - v1yd * v4zy + v1yx * v4dd + v1yx * v4xx - v1yx * v4yy - v1yx * v4zz + v1yy * v4dz + v1yy * v4xy + v1yy * v4yx - v1yy * v4zd - v1yz * v4dy + v1yz * v4xz + v1yz * v4yd + v1yz * v4zx +
                   v3dd * v2yx - v3dx * v2zx - v3dy * v2dx + v3dz * v2xx + v3xd * v2zx - v3xx * v2yx + v3xy * v2xx + v3xz * v2dx - v3yd * v2dx + v3yx * v2xx + v3yy * v2yx + v3yz * v2zx - v3zd * v2xx - v3zx * v2dx + v3zy * v2zx - v3zz * v2yx)

        Γzx = -p * (-v1zd * v4dx - v1zd * v4xd + v1zd * v4yz - v1zd * v4zy + v1zx * v4dd + v1zx * v4xx - v1zx * v4yy - v1zx * v4zz + v1zy * v4dz + v1zy * v4xy + v1zy * v4yx - v1zy * v4zd - v1zz * v4dy + v1zz * v4xz + v1zz * v4yd + v1zz * v4zx + 
                   v3dd * v2zx +  v3dx * v2yx - v3dy * v2xx - v3dz * v2dx - v3xd * v2yx - v3xx * v2zx - v3xy * v2dx + v3xz * v2xx + v3yd * v2xx + v3yx * v2dx - v3yy * v2zx + v3yz * v2yx - v3zd * v2dx + v3zx * v2xx + v3zy * v2yx + v3zz * v2zx)

        Γzy = -p * (-v1zd * v4dy - v1zd * v4xz - v1zd * v4yd + v1zd * v4zx - v1zx * v4dz + v1zx * v4xy + v1zx * v4yx + v1zx * v4zd + v1zy * v4dd - v1zy * v4xx + v1zy * v4yy - v1zy * v4zz + v1zz * v4dx - v1zz * v4xd + v1zz * v4yz + v1zz * v4zy +
                   v3dd * v2zy + v3dx * v2yy - v3dy * v2xy - v3dz * v2dy - v3xd * v2yy - v3xx * v2zy - v3xy * v2dy + v3xz * v2xy + v3yd * v2xy + v3yx * v2dy - v3yy * v2zy + v3yz * v2yy - v3zd * v2dy + v3zx * v2xy + v3zy * v2yy + v3zz * v2zy)

        Γdd = -p * ( v1dd * v4dd + v1dd * v4xx + v1dd * v4yy + v1dd * v4zz - v1dx * v4dx - v1dx * v4xd - v1dx * v4yz + v1dx * v4zy - v1dy * v4dy + v1dy * v4xz - v1dy * v4yd - v1dy * v4zx - v1dz * v4dz - v1dz * v4xy + v1dz * v4yx - v1dz * v4zd +
                   v3dd * v2dd - v3dx * v2xd - v3dy * v2yd - v3dz * v2zd - v3xd * v2xd + v3xx * v2dd + v3xy * v2zd - v3xz * v2yd - v3yd * v2yd - v3yx * v2zd + v3yy * v2dd + v3yz * v2xd - v3zd * v2zd + v3zx * v2yd - v3zy * v2xd + v3zz * v2dd)

        Γxd = -p * ( v1xd * v4dd + v1xd * v4xx + v1xd * v4yy + v1xd * v4zz + v1xx * v4dx + v1xx * v4xd + v1xx * v4yz - v1xx * v4zy + v1xy * v4dy - v1xy * v4xz + v1xy * v4yd + v1xy * v4zx + v1xz * v4dz + v1xz * v4xy - v1xz * v4yx + v1xz * v4zd +
                   v3dd * v2xd + v3dx * v2dd + v3dy * v2zd - v3dz * v2yd + v3xd * v2dd + v3xx * v2xd + v3xy * v2yd + v3xz * v2zd - v3yd * v2zd + v3yx * v2yd - v3yy * v2xd + v3yz * v2dd + v3zd * v2yd + v3zx * v2zd - v3zy * v2dd - v3zz * v2xd)

        Γyd = -p * ( v1yd * v4dd + v1yd * v4xx + v1yd * v4yy + v1yd * v4zz + v1yx * v4dx + v1yx * v4xd + v1yx * v4yz - v1yx * v4zy + v1yy * v4dy - v1yy * v4xz + v1yy * v4yd + v1yy * v4zx + v1yz * v4dz + v1yz * v4xy - v1yz * v4yx + v1yz * v4zd +
                   v3dd * v2yd - v3dx * v2zd + v3dy * v2dd + v3dz * v2xd + v3xd * v2zd - v3xx * v2yd + v3xy * v2xd - v3xz * v2dd + v3yd * v2dd + v3yx * v2xd + v3yy * v2yd + v3yz * v2zd - v3zd * v2xd + v3zx * v2dd + v3zy * v2zd - v3zz * v2yd)

        Γzd = -p * ( v1zd * v4dd + v1zd * v4xx + v1zd * v4yy + v1zd * v4zz + v1zx * v4dx + v1zx * v4xd + v1zx * v4yz - v1zx * v4zy + v1zy * v4dy - v1zy * v4xz + v1zy * v4yd + v1zy * v4zx + v1zz * v4dz + v1zz * v4xy - v1zz * v4yx + v1zz * v4zd +
                   v3dd * v2zd + v3dx * v2yd - v3dy * v2xd + v3dz * v2dd - v3xd * v2yd - v3xx * v2zd + v3xy * v2dd + v3xz * v2xd + v3yd * v2xd - v3yx * v2dd - v3yy * v2zd + v3yz * v2yd + v3zd * v2dd + v3zx * v2xd + v3zy * v2yd + v3zz * v2zd)

        Γdx = -p * ( v1dd * v4dx + v1dd * v4xd - v1dd * v4yz + v1dd * v4zy + v1dx * v4dd + v1dx * v4xx - v1dx * v4yy - v1dx * v4zz + v1dy * v4dz + v1dy * v4xy + v1dy * v4yx - v1dy * v4zd - v1dz * v4dy + v1dz * v4xz + v1dz * v4yd + v1dz * v4zx +
                   v3dd * v2dx + v3dx * v2xx + v3dy * v2yx + v3dz * v2zx + v3xd * v2xx + v3xx * v2dx - v3xy * v2zx + v3xz * v2yx + v3yd * v2yx + v3yx * v2zx + v3yy * v2dx - v3yz * v2xx + v3zd * v2zx - v3zx * v2yx + v3zy * v2xx + v3zz * v2dx)

        Γdy = -p * ( v1dd * v4dy + v1dd * v4xz + v1dd * v4yd - v1dd * v4zx - v1dx * v4dz + v1dx * v4xy + v1dx * v4yx + v1dx * v4zd + v1dy * v4dd - v1dy * v4xx + v1dy * v4yy - v1dy * v4zz + v1dz * v4dx - v1dz * v4xd + v1dz * v4yz + v1dz * v4zy +
                   v3dd * v2dy + v3dx * v2xy + v3dy * v2yy + v3dz * v2zy + v3xd * v2xy + v3xx * v2dy - v3xy * v2zy + v3xz * v2yy + v3yd * v2yy + v3yx * v2zy + v3yy * v2dy - v3yz * v2xy + v3zd * v2zy - v3zx * v2yy + v3zy * v2xy + v3zz * v2dy)

        Γdz = -p * ( v1dd * v4dz - v1dd * v4xy + v1dd * v4yx + v1dd * v4zd + v1dx * v4dy + v1dx * v4xz - v1dx * v4yd + v1dx * v4zx - v1dy * v4dx + v1dy * v4xd + v1dy * v4yz + v1dy * v4zy + v1dz * v4dd - v1dz * v4xx - v1dz * v4yy + v1dz * v4zz +
                   v3dd * v2dz + v3dx * v2xz + v3dy * v2yz + v3dz * v2zz + v3xd * v2xz + v3xx * v2dz - v3xy * v2zz + v3xz * v2yz + v3yd * v2yz + v3yx * v2zz + v3yy * v2dz - v3yz * v2xz + v3zd * v2zz - v3zx * v2yz + v3zy * v2xz + v3zz * v2dz)
                   
        # determine overlap for site i
        overlap_i = overlap[i]
        # determine range for inner sum
        Range = size(overlap_i, 1)
        #compute inner sum 
        @turbo unroll = 1 for j in 1 : Range
            # read cached values for inner site
            v1xx = temp[overlap_i[j, 1],  1, 1]
            v1yy = temp[overlap_i[j, 1],  2, 1]
            v1zz = temp[overlap_i[j, 1],  3, 1]
            v1xy = temp[overlap_i[j, 1],  4, 1]
            v1xz = temp[overlap_i[j, 1],  5, 1]
            v1yz = temp[overlap_i[j, 1],  6, 1]
            v1yx = temp[overlap_i[j, 1],  7, 1]
            v1zx = temp[overlap_i[j, 1],  8, 1]
            v1zy = temp[overlap_i[j, 1],  9, 1]
            v1dd = temp[overlap_i[j, 1], 10, 1]
            v1xd = temp[overlap_i[j, 1], 11, 1]
            v1yd = temp[overlap_i[j, 1], 12, 1]
            v1zd = temp[overlap_i[j, 1], 13, 1]
            v1dx = temp[overlap_i[j, 1], 14, 1]
            v1dy = temp[overlap_i[j, 1], 15, 1]
            v1dz = temp[overlap_i[j, 1], 16, 1]

            v2xx = temp[overlap_i[j, 2],  1, 2]
            v2yy = temp[overlap_i[j, 2],  2, 2]
            v2zz = temp[overlap_i[j, 2],  3, 2]
            v2xy = temp[overlap_i[j, 2],  4, 2]
            v2xz = temp[overlap_i[j, 2],  5, 2]
            v2yz = temp[overlap_i[j, 2],  6, 2]
            v2yx = temp[overlap_i[j, 2],  7, 2]
            v2zx = temp[overlap_i[j, 2],  8, 2]
            v2zy = temp[overlap_i[j, 2],  9, 2]
            v2dd = temp[overlap_i[j, 2], 10, 2]
            v2xd = temp[overlap_i[j, 2], 11, 2]
            v2yd = temp[overlap_i[j, 2], 12, 2]
            v2zd = temp[overlap_i[j, 2], 13, 2]
            v2dx = temp[overlap_i[j, 2], 14, 2]
            v2dy = temp[overlap_i[j, 2], 15, 2]
            v2dz = temp[overlap_i[j, 2], 16, 2]

            # compute contribution at inner site
            Γxx += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dx - 2.0 * v1xx * v2xx - 2.0 * v1xy * v2yx - 2.0 * v1xz * v2zx)
            Γyy += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dy - 2.0 * v1yx * v2xy - 2.0 * v1yy * v2yy - 2.0 * v1yz * v2zy)
            Γzz += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dz - 2.0 * v1zx * v2xz - 2.0 * v1zy * v2yz - 2.0 * v1zz * v2zz)
            Γxy += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dy - 2.0 * v1xx * v2xy - 2.0 * v1xy * v2yy - 2.0 * v1xz * v2zy)
            Γxz += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dz - 2.0 * v1xx * v2xz - 2.0 * v1xy * v2yz - 2.0 * v1xz * v2zz)
            Γyz += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dz - 2.0 * v1yx * v2xz - 2.0 * v1yy * v2yz - 2.0 * v1yz * v2zz)
            Γyx += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dx - 2.0 * v1yx * v2xx - 2.0 * v1yy * v2yx - 2.0 * v1yz * v2zx)
            Γzx += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dx - 2.0 * v1zx * v2xx - 2.0 * v1zy * v2yx - 2.0 * v1zz * v2zx)
            Γzy += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dy - 2.0 * v1zx * v2xy - 2.0 * v1zy * v2yy - 2.0 * v1zz * v2zy)
            Γdd += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dd + 2.0 * v1dx * v2xd + 2.0 * v1dy * v2yd + 2.0 * v1dz * v2zd)
            Γxd += -p * overlap_i[j, 3] * (- 2.0 * v1xd * v2dd - 2.0 * v1xx * v2xd - 2.0 * v1xy * v2yd - 2.0 * v1xz * v2zd)
            Γyd += -p * overlap_i[j, 3] * (- 2.0 * v1yd * v2dd - 2.0 * v1yx * v2xd - 2.0 * v1yy * v2yd - 2.0 * v1yz * v2zd)
            Γzd += -p * overlap_i[j, 3] * (- 2.0 * v1zd * v2dd - 2.0 * v1zx * v2xd - 2.0 * v1zy * v2yd - 2.0 * v1zz * v2zd)
            Γdx += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dx - 2.0 * v1dx * v2xx - 2.0 * v1dy * v2yx - 2.0 * v1dz * v2zx)
            Γdy += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dy - 2.0 * v1dx * v2xy - 2.0 * v1dy * v2yy - 2.0 * v1dz * v2zy)
            Γdz += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dz - 2.0 * v1dx * v2xz - 2.0 * v1dy * v2yz - 2.0 * v1dz * v2zz)
        end 
        # parse result to output buffer 
        buff[1 , i] += dv * Γxx
        buff[2 , i] += dv * Γyy
        buff[3 , i] += dv * Γzz
        buff[4 , i] += dv * Γxy
        buff[5 , i] += dv * Γxz
        buff[6 , i] += dv * Γyz
        buff[7 , i] += dv * Γyx
        buff[8 , i] += dv * Γzx
        buff[9 , i] += dv * Γzy
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
 function compute_t_central!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg,
    da_l :: Action_su2_hkg,
    temp :: Array{Float64, 3}
    )    :: Nothing
    # get propagator and overlap
    p       = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = get_buffer_s( v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_t(t, vt, v, m)
    bu1 = get_buffer_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * ( t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_empty()
    bt2 = get_buffer_t(t, v, vtp, m)
    bu2 = get_buffer_empty()

    # get buffers for local left vertex
    bs3 = get_buffer_s( v + vt, 0.5 * (t + v - vt), 0.5 * (-t + v - vt), m)
    bt3 = get_buffer_t(-v + vt, 0.5 * (t + v + vt), 0.5 * (-t + v + vt), m)
    bu3 = get_buffer_u(t, v, vt, m)

    # get buffers for local right vertex
    bs4 = get_buffer_empty()
    bt4 = get_buffer_empty()
    bu4 = get_buffer_u(t, vtp, v, m)

    # cache local vertex values
    v3xx, v3yy, v3zz, v3xy, v3xz, v3yz, v3yx, v3zx, v3zy, v3dd, v3xd, v3yd, v3zd, v3dx, v3dy, v3dz = get_Γ(1, bs3, bt3, bu3, r, a)
    v4xx, v4yy, v4zz, v4xy, v4xz, v4yz, v4yx, v4zx, v4zy, v4dd, v4xd, v4yd, v4zd, v4dx, v4dy, v4dz = get_Γ(1, bs4, bt4, bu4, r, da_l, ch_s = false, ch_t = false)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1,    a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, da_l, temp, 2, ch_s = false, ch_u = false)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
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
        Γxx = -p * (-v1xd * v4dx - v1xd * v4xd + v1xd * v4yz - v1xd * v4zy + v1xx * v4dd + v1xx * v4xx - v1xx * v4yy - v1xx * v4zz + v1xy * v4dz + v1xy * v4xy + v1xy * v4yx - v1xy * v4zd - v1xz * v4dy + v1xz * v4xz + v1xz * v4yd + v1xz * v4zx +
                   v3dd * v2xx - v3dx * v2dx + v3dy * v2zx - v3dz * v2yx - v3xd * v2dx + v3xx * v2xx + v3xy * v2yx + v3xz * v2zx - v3yd * v2zx + v3yx * v2yx - v3yy * v2xx - v3yz * v2dx + v3zd * v2yx + v3zx * v2zx + v3zy * v2dx - v3zz * v2xx)

        Γyy = -p * (-v1yd * v4dy - v1yd * v4xz - v1yd * v4yd + v1yd * v4zx - v1yx * v4dz + v1yx * v4xy + v1yx * v4yx + v1yx * v4zd + v1yy * v4dd - v1yy * v4xx + v1yy * v4yy - v1yy * v4zz + v1yz * v4dx - v1yz * v4xd + v1yz * v4yz + v1yz * v4zy +
                   v3dd * v2yy - v3dx * v2zy - v3dy * v2dy + v3dz * v2xy + v3xd * v2zy - v3xx * v2yy + v3xy * v2xy + v3xz * v2dy - v3yd * v2dy + v3yx * v2xy + v3yy * v2yy + v3yz * v2zy - v3zd * v2xy - v3zx * v2dy + v3zy * v2zy - v3zz * v2yy)

        Γzz = -p * (-v1zd * v4dz + v1zd * v4xy - v1zd * v4yx - v1zd * v4zd + v1zx * v4dy + v1zx * v4xz - v1zx * v4yd + v1zx * v4zx - v1zy * v4dx + v1zy * v4xd + v1zy * v4yz + v1zy * v4zy + v1zz * v4dd - v1zz * v4xx - v1zz * v4yy + v1zz * v4zz +
                   v3dd * v2zz + v3dx * v2yz - v3dy * v2xz - v3dz * v2dz - v3xd * v2yz - v3xx * v2zz - v3xy * v2dz + v3xz * v2xz + v3yd * v2xz + v3yx * v2dz - v3yy * v2zz + v3yz * v2yz - v3zd * v2dz + v3zx * v2xz + v3zy * v2yz + v3zz * v2zz)

        Γxy = -p * (-v1xd * v4dy - v1xd * v4xz - v1xd * v4yd + v1xd * v4zx - v1xx * v4dz + v1xx * v4xy + v1xx * v4yx + v1xx * v4zd + v1xy * v4dd - v1xy * v4xx + v1xy * v4yy - v1xy * v4zz + v1xz * v4dx - v1xz * v4xd + v1xz * v4yz + v1xz * v4zy +
                   v3dd * v2xy - v3dx * v2dy + v3dy * v2zy - v3dz * v2yy - v3xd * v2dy + v3xx * v2xy + v3xy * v2yy + v3xz * v2zy - v3yd * v2zy + v3yx * v2yy - v3yy * v2xy - v3yz * v2dy + v3zd * v2yy + v3zx * v2zy + v3zy * v2dy - v3zz * v2xy)

        Γxz = -p * (-v1xd * v4dz + v1xd * v4xy - v1xd * v4yx - v1xd * v4zd + v1xx * v4dy + v1xx * v4xz - v1xx * v4yd + v1xx * v4zx - v1xy * v4dx + v1xy * v4xd + v1xy * v4yz + v1xy * v4zy + v1xz * v4dd - v1xz * v4xx - v1xz * v4yy + v1xz * v4zz +
                   v3dd * v2xz - v3dx * v2dz + v3dy * v2zz - v3dz * v2yz - v3xd * v2dz + v3xx * v2xz + v3xy * v2yz + v3xz * v2zz - v3yd * v2zz + v3yx * v2yz - v3yy * v2xz - v3yz * v2dz + v3zd * v2yz + v3zx * v2zz + v3zy * v2dz - v3zz * v2xz) 

        Γyz = -p * (-v1yd * v4dz + v1yd * v4xy - v1yd * v4yx - v1yd * v4zd + v1yx * v4dy + v1yx * v4xz - v1yx * v4yd + v1yx * v4zx - v1yy * v4dx + v1yy * v4xd + v1yy * v4yz + v1yy * v4zy + v1yz * v4dd - v1yz * v4xx - v1yz * v4yy + v1yz * v4zz +
                   v3dd * v2yz - v3dx * v2zz - v3dy * v2dz + v3dz * v2xz + v3xd * v2zz - v3xx * v2yz + v3xy * v2xz + v3xz * v2dz - v3yd * v2dz + v3yx * v2xz + v3yy * v2yz + v3yz * v2zz - v3zd * v2xz - v3zx * v2dz + v3zy * v2zz - v3zz * v2yz) 

        Γyx = -p * (-v1yd * v4dx - v1yd * v4xd + v1yd * v4yz - v1yd * v4zy + v1yx * v4dd + v1yx * v4xx - v1yx * v4yy - v1yx * v4zz + v1yy * v4dz + v1yy * v4xy + v1yy * v4yx - v1yy * v4zd - v1yz * v4dy + v1yz * v4xz + v1yz * v4yd + v1yz * v4zx +
                   v3dd * v2yx - v3dx * v2zx - v3dy * v2dx + v3dz * v2xx + v3xd * v2zx - v3xx * v2yx + v3xy * v2xx + v3xz * v2dx - v3yd * v2dx + v3yx * v2xx + v3yy * v2yx + v3yz * v2zx - v3zd * v2xx - v3zx * v2dx + v3zy * v2zx - v3zz * v2yx)

        Γzx = -p * (-v1zd * v4dx - v1zd * v4xd + v1zd * v4yz - v1zd * v4zy + v1zx * v4dd + v1zx * v4xx - v1zx * v4yy - v1zx * v4zz + v1zy * v4dz + v1zy * v4xy + v1zy * v4yx - v1zy * v4zd - v1zz * v4dy + v1zz * v4xz + v1zz * v4yd + v1zz * v4zx + 
                   v3dd * v2zx +  v3dx * v2yx - v3dy * v2xx - v3dz * v2dx - v3xd * v2yx - v3xx * v2zx - v3xy * v2dx + v3xz * v2xx + v3yd * v2xx + v3yx * v2dx - v3yy * v2zx + v3yz * v2yx - v3zd * v2dx + v3zx * v2xx + v3zy * v2yx + v3zz * v2zx)

        Γzy = -p * (-v1zd * v4dy - v1zd * v4xz - v1zd * v4yd + v1zd * v4zx - v1zx * v4dz + v1zx * v4xy + v1zx * v4yx + v1zx * v4zd + v1zy * v4dd - v1zy * v4xx + v1zy * v4yy - v1zy * v4zz + v1zz * v4dx - v1zz * v4xd + v1zz * v4yz + v1zz * v4zy +
                   v3dd * v2zy + v3dx * v2yy - v3dy * v2xy - v3dz * v2dy - v3xd * v2yy - v3xx * v2zy - v3xy * v2dy + v3xz * v2xy + v3yd * v2xy + v3yx * v2dy - v3yy * v2zy + v3yz * v2yy - v3zd * v2dy + v3zx * v2xy + v3zy * v2yy + v3zz * v2zy)

        Γdd = -p * ( v1dd * v4dd + v1dd * v4xx + v1dd * v4yy + v1dd * v4zz - v1dx * v4dx - v1dx * v4xd - v1dx * v4yz + v1dx * v4zy - v1dy * v4dy + v1dy * v4xz - v1dy * v4yd - v1dy * v4zx - v1dz * v4dz - v1dz * v4xy + v1dz * v4yx - v1dz * v4zd +
                   v3dd * v2dd - v3dx * v2xd - v3dy * v2yd - v3dz * v2zd - v3xd * v2xd + v3xx * v2dd + v3xy * v2zd - v3xz * v2yd - v3yd * v2yd - v3yx * v2zd + v3yy * v2dd + v3yz * v2xd - v3zd * v2zd + v3zx * v2yd - v3zy * v2xd + v3zz * v2dd)

        Γxd = -p * ( v1xd * v4dd + v1xd * v4xx + v1xd * v4yy + v1xd * v4zz + v1xx * v4dx + v1xx * v4xd + v1xx * v4yz - v1xx * v4zy + v1xy * v4dy - v1xy * v4xz + v1xy * v4yd + v1xy * v4zx + v1xz * v4dz + v1xz * v4xy - v1xz * v4yx + v1xz * v4zd +
                   v3dd * v2xd + v3dx * v2dd + v3dy * v2zd - v3dz * v2yd + v3xd * v2dd + v3xx * v2xd + v3xy * v2yd + v3xz * v2zd - v3yd * v2zd + v3yx * v2yd - v3yy * v2xd + v3yz * v2dd + v3zd * v2yd + v3zx * v2zd - v3zy * v2dd - v3zz * v2xd)

        Γyd = -p * ( v1yd * v4dd + v1yd * v4xx + v1yd * v4yy + v1yd * v4zz + v1yx * v4dx + v1yx * v4xd + v1yx * v4yz - v1yx * v4zy + v1yy * v4dy - v1yy * v4xz + v1yy * v4yd + v1yy * v4zx + v1yz * v4dz + v1yz * v4xy - v1yz * v4yx + v1yz * v4zd +
                   v3dd * v2yd - v3dx * v2zd + v3dy * v2dd + v3dz * v2xd + v3xd * v2zd - v3xx * v2yd + v3xy * v2xd - v3xz * v2dd + v3yd * v2dd + v3yx * v2xd + v3yy * v2yd + v3yz * v2zd - v3zd * v2xd + v3zx * v2dd + v3zy * v2zd - v3zz * v2yd)

        Γzd = -p * ( v1zd * v4dd + v1zd * v4xx + v1zd * v4yy + v1zd * v4zz + v1zx * v4dx + v1zx * v4xd + v1zx * v4yz - v1zx * v4zy + v1zy * v4dy - v1zy * v4xz + v1zy * v4yd + v1zy * v4zx + v1zz * v4dz + v1zz * v4xy - v1zz * v4yx + v1zz * v4zd +
                   v3dd * v2zd + v3dx * v2yd - v3dy * v2xd + v3dz * v2dd - v3xd * v2yd - v3xx * v2zd + v3xy * v2dd + v3xz * v2xd + v3yd * v2xd - v3yx * v2dd - v3yy * v2zd + v3yz * v2yd + v3zd * v2dd + v3zx * v2xd + v3zy * v2yd + v3zz * v2zd)

        Γdx = -p * ( v1dd * v4dx + v1dd * v4xd - v1dd * v4yz + v1dd * v4zy + v1dx * v4dd + v1dx * v4xx - v1dx * v4yy - v1dx * v4zz + v1dy * v4dz + v1dy * v4xy + v1dy * v4yx - v1dy * v4zd - v1dz * v4dy + v1dz * v4xz + v1dz * v4yd + v1dz * v4zx +
                   v3dd * v2dx + v3dx * v2xx + v3dy * v2yx + v3dz * v2zx + v3xd * v2xx + v3xx * v2dx - v3xy * v2zx + v3xz * v2yx + v3yd * v2yx + v3yx * v2zx + v3yy * v2dx - v3yz * v2xx + v3zd * v2zx - v3zx * v2yx + v3zy * v2xx + v3zz * v2dx)

        Γdy = -p * ( v1dd * v4dy + v1dd * v4xz + v1dd * v4yd - v1dd * v4zx - v1dx * v4dz + v1dx * v4xy + v1dx * v4yx + v1dx * v4zd + v1dy * v4dd - v1dy * v4xx + v1dy * v4yy - v1dy * v4zz + v1dz * v4dx - v1dz * v4xd + v1dz * v4yz + v1dz * v4zy +
                   v3dd * v2dy + v3dx * v2xy + v3dy * v2yy + v3dz * v2zy + v3xd * v2xy + v3xx * v2dy - v3xy * v2zy + v3xz * v2yy + v3yd * v2yy + v3yx * v2zy + v3yy * v2dy - v3yz * v2xy + v3zd * v2zy - v3zx * v2yy + v3zy * v2xy + v3zz * v2dy)

        Γdz = -p * ( v1dd * v4dz - v1dd * v4xy + v1dd * v4yx + v1dd * v4zd + v1dx * v4dy + v1dx * v4xz - v1dx * v4yd + v1dx * v4zx - v1dy * v4dx + v1dy * v4xd + v1dy * v4yz + v1dy * v4zy + v1dz * v4dd - v1dz * v4xx - v1dz * v4yy + v1dz * v4zz +
                   v3dd * v2dz + v3dx * v2xz + v3dy * v2yz + v3dz * v2zz + v3xd * v2xz + v3xx * v2dz - v3xy * v2zz + v3xz * v2yz + v3yd * v2yz + v3yx * v2zz + v3yy * v2dz - v3yz * v2xz + v3zd * v2zz - v3zx * v2yz + v3zy * v2xz + v3zz * v2dz)
    
        # determine overlap for site i
        overlap_i = overlap[i]
    
        # determine range for inner sum
        Range = size(overlap_i, 1)
    
        #compute inner sum 
        @turbo unroll = 1 for j in 1 : Range
            # read cached values for inner site
            v1xx = temp[overlap_i[j, 1],  1, 1]
            v1yy = temp[overlap_i[j, 1],  2, 1]
            v1zz = temp[overlap_i[j, 1],  3, 1]
            v1xy = temp[overlap_i[j, 1],  4, 1]
            v1xz = temp[overlap_i[j, 1],  5, 1]
            v1yz = temp[overlap_i[j, 1],  6, 1]
            v1yx = temp[overlap_i[j, 1],  7, 1]
            v1zx = temp[overlap_i[j, 1],  8, 1]
            v1zy = temp[overlap_i[j, 1],  9, 1]
            v1dd = temp[overlap_i[j, 1], 10, 1]
            v1xd = temp[overlap_i[j, 1], 11, 1]
            v1yd = temp[overlap_i[j, 1], 12, 1]
            v1zd = temp[overlap_i[j, 1], 13, 1]
            v1dx = temp[overlap_i[j, 1], 14, 1]
            v1dy = temp[overlap_i[j, 1], 15, 1]
            v1dz = temp[overlap_i[j, 1], 16, 1]

            v2xx = temp[overlap_i[j, 2],  1, 2]
            v2yy = temp[overlap_i[j, 2],  2, 2]
            v2zz = temp[overlap_i[j, 2],  3, 2]
            v2xy = temp[overlap_i[j, 2],  4, 2]
            v2xz = temp[overlap_i[j, 2],  5, 2]
            v2yz = temp[overlap_i[j, 2],  6, 2]
            v2yx = temp[overlap_i[j, 2],  7, 2]
            v2zx = temp[overlap_i[j, 2],  8, 2]
            v2zy = temp[overlap_i[j, 2],  9, 2]
            v2dd = temp[overlap_i[j, 2], 10, 2]
            v2xd = temp[overlap_i[j, 2], 11, 2]
            v2yd = temp[overlap_i[j, 2], 12, 2]
            v2zd = temp[overlap_i[j, 2], 13, 2]
            v2dx = temp[overlap_i[j, 2], 14, 2]
            v2dy = temp[overlap_i[j, 2], 15, 2]
            v2dz = temp[overlap_i[j, 2], 16, 2]

            # compute contribution at inner site
            Γxx += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dx - 2.0 * v1xx * v2xx - 2.0 * v1xy * v2yx - 2.0 * v1xz * v2zx)
            Γyy += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dy - 2.0 * v1yx * v2xy - 2.0 * v1yy * v2yy - 2.0 * v1yz * v2zy)
            Γzz += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dz - 2.0 * v1zx * v2xz - 2.0 * v1zy * v2yz - 2.0 * v1zz * v2zz)
            Γxy += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dy - 2.0 * v1xx * v2xy - 2.0 * v1xy * v2yy - 2.0 * v1xz * v2zy)
            Γxz += -p * overlap_i[j, 3] * (+ 2.0 * v1xd * v2dz - 2.0 * v1xx * v2xz - 2.0 * v1xy * v2yz - 2.0 * v1xz * v2zz)
            Γyz += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dz - 2.0 * v1yx * v2xz - 2.0 * v1yy * v2yz - 2.0 * v1yz * v2zz)
            Γyx += -p * overlap_i[j, 3] * (+ 2.0 * v1yd * v2dx - 2.0 * v1yx * v2xx - 2.0 * v1yy * v2yx - 2.0 * v1yz * v2zx)
            Γzx += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dx - 2.0 * v1zx * v2xx - 2.0 * v1zy * v2yx - 2.0 * v1zz * v2zx)
            Γzy += -p * overlap_i[j, 3] * (+ 2.0 * v1zd * v2dy - 2.0 * v1zx * v2xy - 2.0 * v1zy * v2yy - 2.0 * v1zz * v2zy)
            Γdd += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dd + 2.0 * v1dx * v2xd + 2.0 * v1dy * v2yd + 2.0 * v1dz * v2zd)
            Γxd += -p * overlap_i[j, 3] * (- 2.0 * v1xd * v2dd - 2.0 * v1xx * v2xd - 2.0 * v1xy * v2yd - 2.0 * v1xz * v2zd)
            Γyd += -p * overlap_i[j, 3] * (- 2.0 * v1yd * v2dd - 2.0 * v1yx * v2xd - 2.0 * v1yy * v2yd - 2.0 * v1yz * v2zd)
            Γzd += -p * overlap_i[j, 3] * (- 2.0 * v1zd * v2dd - 2.0 * v1zx * v2xd - 2.0 * v1zy * v2yd - 2.0 * v1zz * v2zd)
            Γdx += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dx - 2.0 * v1dx * v2xx - 2.0 * v1dy * v2yx - 2.0 * v1dz * v2zx)
            Γdy += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dy - 2.0 * v1dx * v2xy - 2.0 * v1dy * v2yy - 2.0 * v1dz * v2zy)
            Γdz += -p * overlap_i[j, 3] * (- 2.0 * v1dd * v2dz - 2.0 * v1dx * v2xz - 2.0 * v1dy * v2yz - 2.0 * v1dz * v2zz)
        end     
        # parse result to output buffer 
        buff[1 , i] += dv * Γxx
        buff[2 , i] += dv * Γyy
        buff[3 , i] += dv * Γzz
        buff[4 , i] += dv * Γxy
        buff[5 , i] += dv * Γxz
        buff[6 , i] += dv * Γyz
        buff[7 , i] += dv * Γyx
        buff[8 , i] += dv * Γzx
        buff[9 , i] += dv * Γzy
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