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

    # cache left local vertex values
    #left vertex (Γ_ii)
    v3xx, v3yy, v3zz, v3xy, v3xz, v3yz, v3yx, v3zx, v3zy, v3dd, v3xd, v3yd, v3zd, v3dx, v3dy, v3dz = get_Γ(1, bs3, bt3, bu3, r, a)
    # Right local vertex (Γ_jj)
    v4 = collect(get_Γ(1, bs4, bt4, bu4, r, a)) 

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

        #Map right local vertex (v4) from reference site to Γ_jj
        map = r.localmap[i]
        v4xx = map.signs[1] * v4[map.components[1]]
        v4yy = map.signs[2] * v4[map.components[2]]
        v4zz = map.signs[3] * v4[map.components[3]]
        v4xy = map.signs[4] * v4[map.components[4]]
        v4xz = map.signs[5] * v4[map.components[5]]
        v4yz = map.signs[6] * v4[map.components[6]]
        v4yx = map.signs[7] * v4[map.components[7]]
        v4zx = map.signs[8] * v4[map.components[8]]
        v4zy = map.signs[9] * v4[map.components[9]]
        v4dd = map.signs[10] * v4[map.components[10]]
        v4xd = map.signs[11] * v4[map.components[11]]
        v4yd = map.signs[12] * v4[map.components[12]]
        v4zd = map.signs[13] * v4[map.components[13]]
        v4dx = map.signs[14] * v4[map.components[14]]
        v4dy = map.signs[15] * v4[map.components[15]]
        v4dz = map.signs[16] * v4[map.components[16]]
        
        # compute contribution at site i
        Γdd = -p * (+ 1.0 * v1dd * v4dd
        + 1.0 * v1dd * v4xx
        + 1.0 * v1dd * v4yy
        + 1.0 * v1dd * v4zz
        - 1.0 * v1dx * v4dx
        - 1.0 * v1dx * v4xd        
        - 1.0 * v1dx * v4yz
        + 1.0 * v1dx * v4zy
        - 1.0 * v1dy * v4dy
        + 1.0 * v1dy * v4xz
        - 1.0 * v1dy * v4yd
        - 1.0 * v1dy * v4zx
        - 1.0 * v1dz * v4dz
        - 1.0 * v1dz * v4xy
        + 1.0 * v1dz * v4yx
        - 1.0 * v1dz * v4zd
        + 1.0 * v3dd * v2dd
        - 1.0 * v3dx * v2xd
        - 1.0 * v3dy * v2yd
        - 1.0 * v3dz * v2zd
        - 1.0 * v3xd * v2xd
        + 1.0 * v3xx * v2dd
        + 1.0 * v3xy * v2zd
        - 1.0 * v3xz * v2yd
        - 1.0 * v3yd * v2yd
        - 1.0 * v3yx * v2zd
        + 1.0 * v3yy * v2dd
        + 1.0 * v3yz * v2xd
        - 1.0 * v3zd * v2zd
        + 1.0 * v3zx * v2yd
        - 1.0 * v3zy * v2xd
        + 1.0 * v3zz * v2dd
        )

        Γdx = -p * (+ 1.0 * v1dd * v4dx
        + 1.0 * v1dd * v4xd
        - 1.0 * v1dd * v4yz
        + 1.0 * v1dd * v4zy
        + 1.0 * v1dx * v4dd
        + 1.0 * v1dx * v4xx
        - 1.0 * v1dx * v4yy
        - 1.0 * v1dx * v4zz
        + 1.0 * v1dy * v4dz
        + 1.0 * v1dy * v4xy
        + 1.0 * v1dy * v4yx
        - 1.0 * v1dy * v4zd
        - 1.0 * v1dz * v4dy
        + 1.0 * v1dz * v4xz
        + 1.0 * v1dz * v4yd
        + 1.0 * v1dz * v4zx
        + 1.0 * v3dd * v2dx
        + 1.0 * v3dx * v2xx
        + 1.0 * v3dy * v2yx
        + 1.0 * v3dz * v2zx
        + 1.0 * v3xd * v2xx
        + 1.0 * v3xx * v2dx
        - 1.0 * v3xy * v2zx
        + 1.0 * v3xz * v2yx
        + 1.0 * v3yd * v2yx
        + 1.0 * v3yx * v2zx
        + 1.0 * v3yy * v2dx
        - 1.0 * v3yz * v2xx
        + 1.0 * v3zd * v2zx
        - 1.0 * v3zx * v2yx
        + 1.0 * v3zy * v2xx
        + 1.0 * v3zz * v2dx
        )

        Γdy = -p * (+ 1.0 * v1dd * v4dy
        + 1.0 * v1dd * v4xz
        + 1.0 * v1dd * v4yd
        - 1.0 * v1dd * v4zx
        - 1.0 * v1dx * v4dz
        + 1.0 * v1dx * v4xy
        + 1.0 * v1dx * v4yx
        + 1.0 * v1dx * v4zd
        + 1.0 * v1dy * v4dd
        - 1.0 * v1dy * v4xx
        + 1.0 * v1dy * v4yy
        - 1.0 * v1dy * v4zz
        + 1.0 * v1dz * v4dx
        - 1.0 * v1dz * v4xd
        + 1.0 * v1dz * v4yz
        + 1.0 * v1dz * v4zy
        + 1.0 * v3dd * v2dy
        + 1.0 * v3dx * v2xy
        + 1.0 * v3dy * v2yy
        + 1.0 * v3dz * v2zy
        + 1.0 * v3xd * v2xy
        + 1.0 * v3xx * v2dy
        - 1.0 * v3xy * v2zy
        + 1.0 * v3xz * v2yy
        + 1.0 * v3yd * v2yy
        + 1.0 * v3yx * v2zy
        + 1.0 * v3yy * v2dy
        - 1.0 * v3yz * v2xy
        + 1.0 * v3zd * v2zy
        - 1.0 * v3zx * v2yy
        + 1.0 * v3zy * v2xy
        + 1.0 * v3zz * v2dy
        )

        Γdz = -p * (+ 1.0 * v1dd * v4dz
        - 1.0 * v1dd * v4xy
        + 1.0 * v1dd * v4yx
        + 1.0 * v1dd * v4zd
        + 1.0 * v1dx * v4dy
        + 1.0 * v1dx * v4xz
        - 1.0 * v1dx * v4yd
        + 1.0 * v1dx * v4zx
        - 1.0 * v1dy * v4dx
        + 1.0 * v1dy * v4xd
        + 1.0 * v1dy * v4yz
        + 1.0 * v1dy * v4zy
        + 1.0 * v1dz * v4dd
        - 1.0 * v1dz * v4xx
        - 1.0 * v1dz * v4yy
        + 1.0 * v1dz * v4zz
        + 1.0 * v3dd * v2dz
        + 1.0 * v3dx * v2xz
        + 1.0 * v3dy * v2yz
        + 1.0 * v3dz * v2zz
        + 1.0 * v3xd * v2xz
        + 1.0 * v3xx * v2dz
        - 1.0 * v3xy * v2zz
        + 1.0 * v3xz * v2yz
        + 1.0 * v3yd * v2yz
        + 1.0 * v3yx * v2zz
        + 1.0 * v3yy * v2dz
        - 1.0 * v3yz * v2xz
        + 1.0 * v3zd * v2zz
        - 1.0 * v3zx * v2yz
        + 1.0 * v3zy * v2xz
        + 1.0 * v3zz * v2dz
        )

        Γxd = -p * (+ 1.0 * v1xd * v4dd
        + 1.0 * v1xd * v4xx
        + 1.0 * v1xd * v4yy
        + 1.0 * v1xd * v4zz
        + 1.0 * v1xx * v4dx
        + 1.0 * v1xx * v4xd
        + 1.0 * v1xx * v4yz
        - 1.0 * v1xx * v4zy
        + 1.0 * v1xy * v4dy
        - 1.0 * v1xy * v4xz
        + 1.0 * v1xy * v4yd
        + 1.0 * v1xy * v4zx
        + 1.0 * v1xz * v4dz
        + 1.0 * v1xz * v4xy
        - 1.0 * v1xz * v4yx
        + 1.0 * v1xz * v4zd
        + 1.0 * v3dd * v2xd
        + 1.0 * v3dx * v2dd
        + 1.0 * v3dy * v2zd
        - 1.0 * v3dz * v2yd
        + 1.0 * v3xd * v2dd
        + 1.0 * v3xx * v2xd
        + 1.0 * v3xy * v2yd
        + 1.0 * v3xz * v2zd
        - 1.0 * v3yd * v2zd
        + 1.0 * v3yx * v2yd
        - 1.0 * v3yy * v2xd
        + 1.0 * v3yz * v2dd
        + 1.0 * v3zd * v2yd
        + 1.0 * v3zx * v2zd
        - 1.0 * v3zy * v2dd
        - 1.0 * v3zz * v2xd
        )

        Γxx = -p * (- 1.0 * v1xd * v4dx
        - 1.0 * v1xd * v4xd
        + 1.0 * v1xd * v4yz
        - 1.0 * v1xd * v4zy
        + 1.0 * v1xx * v4dd
        + 1.0 * v1xx * v4xx
        - 1.0 * v1xx * v4yy
        - 1.0 * v1xx * v4zz
        + 1.0 * v1xy * v4dz
        + 1.0 * v1xy * v4xy
        + 1.0 * v1xy * v4yx
        - 1.0 * v1xy * v4zd
        - 1.0 * v1xz * v4dy
        + 1.0 * v1xz * v4xz
        + 1.0 * v1xz * v4yd
        + 1.0 * v1xz * v4zx
        + 1.0 * v3dd * v2xx
        - 1.0 * v3dx * v2dx
        + 1.0 * v3dy * v2zx
        - 1.0 * v3dz * v2yx
        - 1.0 * v3xd * v2dx
        + 1.0 * v3xx * v2xx
        + 1.0 * v3xy * v2yx
        + 1.0 * v3xz * v2zx
        - 1.0 * v3yd * v2zx
        + 1.0 * v3yx * v2yx
        - 1.0 * v3yy * v2xx
        - 1.0 * v3yz * v2dx
        + 1.0 * v3zd * v2yx
        + 1.0 * v3zx * v2zx
        + 1.0 * v3zy * v2dx
        - 1.0 * v3zz * v2xx
        )

        Γxy = -p * (- 1.0 * v1xd * v4dy
        - 1.0 * v1xd * v4xz
        - 1.0 * v1xd * v4yd
        + 1.0 * v1xd * v4zx
        - 1.0 * v1xx * v4dz
        + 1.0 * v1xx * v4xy
        + 1.0 * v1xx * v4yx
        + 1.0 * v1xx * v4zd
        + 1.0 * v1xy * v4dd
        - 1.0 * v1xy * v4xx
        + 1.0 * v1xy * v4yy
        - 1.0 * v1xy * v4zz
        + 1.0 * v1xz * v4dx
        - 1.0 * v1xz * v4xd
        + 1.0 * v1xz * v4yz
        + 1.0 * v1xz * v4zy
        + 1.0 * v3dd * v2xy
        - 1.0 * v3dx * v2dy
        + 1.0 * v3dy * v2zy
        - 1.0 * v3dz * v2yy
        - 1.0 * v3xd * v2dy
        + 1.0 * v3xx * v2xy
        + 1.0 * v3xy * v2yy
        + 1.0 * v3xz * v2zy
        - 1.0 * v3yd * v2zy
        + 1.0 * v3yx * v2yy
        - 1.0 * v3yy * v2xy
        - 1.0 * v3yz * v2dy
        + 1.0 * v3zd * v2yy
        + 1.0 * v3zx * v2zy
        + 1.0 * v3zy * v2dy
        - 1.0 * v3zz * v2xy
        )

        Γxz = -p * (- 1.0 * v1xd * v4dz
        + 1.0 * v1xd * v4xy
        - 1.0 * v1xd * v4yx
        - 1.0 * v1xd * v4zd
        + 1.0 * v1xx * v4dy
        + 1.0 * v1xx * v4xz
        - 1.0 * v1xx * v4yd
        + 1.0 * v1xx * v4zx
        - 1.0 * v1xy * v4dx
        + 1.0 * v1xy * v4xd
        + 1.0 * v1xy * v4yz
        + 1.0 * v1xy * v4zy
        + 1.0 * v1xz * v4dd
        - 1.0 * v1xz * v4xx
        - 1.0 * v1xz * v4yy
        + 1.0 * v1xz * v4zz
        + 1.0 * v3dd * v2xz
        - 1.0 * v3dx * v2dz
        + 1.0 * v3dy * v2zz
        - 1.0 * v3dz * v2yz
        - 1.0 * v3xd * v2dz
        + 1.0 * v3xx * v2xz
        + 1.0 * v3xy * v2yz
        + 1.0 * v3xz * v2zz
        - 1.0 * v3yd * v2zz
        + 1.0 * v3yx * v2yz
        - 1.0 * v3yy * v2xz
        - 1.0 * v3yz * v2dz
        + 1.0 * v3zd * v2yz
        + 1.0 * v3zx * v2zz
        + 1.0 * v3zy * v2dz
        - 1.0 * v3zz * v2xz
        )

        Γyd = -p * (+ 1.0 * v1yd * v4dd
        + 1.0 * v1yd * v4xx
        + 1.0 * v1yd * v4yy
        + 1.0 * v1yd * v4zz
        + 1.0 * v1yx * v4dx
        + 1.0 * v1yx * v4xd
        + 1.0 * v1yx * v4yz
        - 1.0 * v1yx * v4zy
        + 1.0 * v1yy * v4dy
        - 1.0 * v1yy * v4xz
        + 1.0 * v1yy * v4yd
        + 1.0 * v1yy * v4zx
        + 1.0 * v1yz * v4dz
        + 1.0 * v1yz * v4xy
        - 1.0 * v1yz * v4yx
        + 1.0 * v1yz * v4zd
        + 1.0 * v3dd * v2yd
        - 1.0 * v3dx * v2zd
        + 1.0 * v3dy * v2dd
        + 1.0 * v3dz * v2xd
        + 1.0 * v3xd * v2zd
        - 1.0 * v3xx * v2yd
        + 1.0 * v3xy * v2xd
        - 1.0 * v3xz * v2dd
        + 1.0 * v3yd * v2dd
        + 1.0 * v3yx * v2xd
        + 1.0 * v3yy * v2yd
        + 1.0 * v3yz * v2zd
        - 1.0 * v3zd * v2xd
        + 1.0 * v3zx * v2dd
        + 1.0 * v3zy * v2zd
        - 1.0 * v3zz * v2yd
        )

        Γyx = -p * (- 1.0 * v1yd * v4dx
        - 1.0 * v1yd * v4xd
        + 1.0 * v1yd * v4yz
        - 1.0 * v1yd * v4zy
        + 1.0 * v1yx * v4dd
        + 1.0 * v1yx * v4xx
        - 1.0 * v1yx * v4yy
        - 1.0 * v1yx * v4zz
        + 1.0 * v1yy * v4dz
        + 1.0 * v1yy * v4xy
        + 1.0 * v1yy * v4yx
        - 1.0 * v1yy * v4zd
        - 1.0 * v1yz * v4dy
        + 1.0 * v1yz * v4xz
        + 1.0 * v1yz * v4yd
        + 1.0 * v1yz * v4zx
        + 1.0 * v3dd * v2yx
        - 1.0 * v3dx * v2zx
        - 1.0 * v3dy * v2dx
        + 1.0 * v3dz * v2xx
        + 1.0 * v3xd * v2zx
        - 1.0 * v3xx * v2yx
        + 1.0 * v3xy * v2xx
        + 1.0 * v3xz * v2dx
        - 1.0 * v3yd * v2dx
        + 1.0 * v3yx * v2xx
        + 1.0 * v3yy * v2yx
        + 1.0 * v3yz * v2zx
        - 1.0 * v3zd * v2xx
        - 1.0 * v3zx * v2dx
        + 1.0 * v3zy * v2zx
        - 1.0 * v3zz * v2yx
        )

        Γyy = -p * (- 1.0 * v1yd * v4dy
        - 1.0 * v1yd * v4xz
        - 1.0 * v1yd * v4yd
        + 1.0 * v1yd * v4zx
        - 1.0 * v1yx * v4dz
        + 1.0 * v1yx * v4xy
        + 1.0 * v1yx * v4yx
        + 1.0 * v1yx * v4zd
        + 1.0 * v1yy * v4dd
        - 1.0 * v1yy * v4xx
        + 1.0 * v1yy * v4yy
        - 1.0 * v1yy * v4zz
        + 1.0 * v1yz * v4dx
        - 1.0 * v1yz * v4xd
        + 1.0 * v1yz * v4yz
        + 1.0 * v1yz * v4zy
        + 1.0 * v3dd * v2yy
        - 1.0 * v3dx * v2zy
        - 1.0 * v3dy * v2dy
        + 1.0 * v3dz * v2xy
        + 1.0 * v3xd * v2zy
        - 1.0 * v3xx * v2yy
        + 1.0 * v3xy * v2xy
        + 1.0 * v3xz * v2dy
        - 1.0 * v3yd * v2dy
        + 1.0 * v3yx * v2xy
        + 1.0 * v3yy * v2yy
        + 1.0 * v3yz * v2zy
        - 1.0 * v3zd * v2xy
        - 1.0 * v3zx * v2dy
        + 1.0 * v3zy * v2zy
        - 1.0 * v3zz * v2yy
        )

        Γyz = -p * (- 1.0 * v1yd * v4dz
        + 1.0 * v1yd * v4xy
        - 1.0 * v1yd * v4yx
        - 1.0 * v1yd * v4zd
        + 1.0 * v1yx * v4dy
        + 1.0 * v1yx * v4xz
        - 1.0 * v1yx * v4yd
        + 1.0 * v1yx * v4zx
        - 1.0 * v1yy * v4dx
        + 1.0 * v1yy * v4xd
        + 1.0 * v1yy * v4yz
        + 1.0 * v1yy * v4zy
        + 1.0 * v1yz * v4dd
        - 1.0 * v1yz * v4xx
        - 1.0 * v1yz * v4yy
        + 1.0 * v1yz * v4zz
        + 1.0 * v3dd * v2yz
        - 1.0 * v3dx * v2zz
        - 1.0 * v3dy * v2dz
        + 1.0 * v3dz * v2xz
        + 1.0 * v3xd * v2zz
        - 1.0 * v3xx * v2yz
        + 1.0 * v3xy * v2xz
        + 1.0 * v3xz * v2dz
        - 1.0 * v3yd * v2dz
        + 1.0 * v3yx * v2xz
        + 1.0 * v3yy * v2yz
        + 1.0 * v3yz * v2zz
        - 1.0 * v3zd * v2xz
        - 1.0 * v3zx * v2dz
        + 1.0 * v3zy * v2zz
        - 1.0 * v3zz * v2yz
        )

        Γzd = -p * (+ 1.0 * v1zd * v4dd
        + 1.0 * v1zd * v4xx
        + 1.0 * v1zd * v4yy
        + 1.0 * v1zd * v4zz
        + 1.0 * v1zx * v4dx
        + 1.0 * v1zx * v4xd
        + 1.0 * v1zx * v4yz
        - 1.0 * v1zx * v4zy
        + 1.0 * v1zy * v4dy
        - 1.0 * v1zy * v4xz
        + 1.0 * v1zy * v4yd
        + 1.0 * v1zy * v4zx
        + 1.0 * v1zz * v4dz
        + 1.0 * v1zz * v4xy
        - 1.0 * v1zz * v4yx
        + 1.0 * v1zz * v4zd
        + 1.0 * v3dd * v2zd
        + 1.0 * v3dx * v2yd
        - 1.0 * v3dy * v2xd
        + 1.0 * v3dz * v2dd
        - 1.0 * v3xd * v2yd
        - 1.0 * v3xx * v2zd
        + 1.0 * v3xy * v2dd
        + 1.0 * v3xz * v2xd
        + 1.0 * v3yd * v2xd
        - 1.0 * v3yx * v2dd
        - 1.0 * v3yy * v2zd
        + 1.0 * v3yz * v2yd
        + 1.0 * v3zd * v2dd
        + 1.0 * v3zx * v2xd
        + 1.0 * v3zy * v2yd
        + 1.0 * v3zz * v2zd
        )

        Γzx = -p * (- 1.0 * v1zd * v4dx
        - 1.0 * v1zd * v4xd
        + 1.0 * v1zd * v4yz
        - 1.0 * v1zd * v4zy
        + 1.0 * v1zx * v4dd
        + 1.0 * v1zx * v4xx
        - 1.0 * v1zx * v4yy
        - 1.0 * v1zx * v4zz
        + 1.0 * v1zy * v4dz
        + 1.0 * v1zy * v4xy
        + 1.0 * v1zy * v4yx
        - 1.0 * v1zy * v4zd
        - 1.0 * v1zz * v4dy
        + 1.0 * v1zz * v4xz
        + 1.0 * v1zz * v4yd
        + 1.0 * v1zz * v4zx
        + 1.0 * v3dd * v2zx
        + 1.0 * v3dx * v2yx
        - 1.0 * v3dy * v2xx
        - 1.0 * v3dz * v2dx
        - 1.0 * v3xd * v2yx
        - 1.0 * v3xx * v2zx
        - 1.0 * v3xy * v2dx
        + 1.0 * v3xz * v2xx
        + 1.0 * v3yd * v2xx
        + 1.0 * v3yx * v2dx
        - 1.0 * v3yy * v2zx
        + 1.0 * v3yz * v2yx
        - 1.0 * v3zd * v2dx
        + 1.0 * v3zx * v2xx
        + 1.0 * v3zy * v2yx
        + 1.0 * v3zz * v2zx
        )

        Γzy = -p * (- 1.0 * v1zd * v4dy
        - 1.0 * v1zd * v4xz
        - 1.0 * v1zd * v4yd
        + 1.0 * v1zd * v4zx
        - 1.0 * v1zx * v4dz
        + 1.0 * v1zx * v4xy
        + 1.0 * v1zx * v4yx
        + 1.0 * v1zx * v4zd
        + 1.0 * v1zy * v4dd
        - 1.0 * v1zy * v4xx
        + 1.0 * v1zy * v4yy
        - 1.0 * v1zy * v4zz
        + 1.0 * v1zz * v4dx
        - 1.0 * v1zz * v4xd
        + 1.0 * v1zz * v4yz
        + 1.0 * v1zz * v4zy
        + 1.0 * v3dd * v2zy
        + 1.0 * v3dx * v2yy
        - 1.0 * v3dy * v2xy
        - 1.0 * v3dz * v2dy
        - 1.0 * v3xd * v2yy
        - 1.0 * v3xx * v2zy
        - 1.0 * v3xy * v2dy
        + 1.0 * v3xz * v2xy
        + 1.0 * v3yd * v2xy
        + 1.0 * v3yx * v2dy
        - 1.0 * v3yy * v2zy
        + 1.0 * v3yz * v2yy
        - 1.0 * v3zd * v2dy
        + 1.0 * v3zx * v2xy
        + 1.0 * v3zy * v2yy
        + 1.0 * v3zz * v2zy
        )

        Γzz = -p * (- 1.0 * v1zd * v4dz
        + 1.0 * v1zd * v4xy
        - 1.0 * v1zd * v4yx
        - 1.0 * v1zd * v4zd
        + 1.0 * v1zx * v4dy
        + 1.0 * v1zx * v4xz
        - 1.0 * v1zx * v4yd
        + 1.0 * v1zx * v4zx
        - 1.0 * v1zy * v4dx
        + 1.0 * v1zy * v4xd
        + 1.0 * v1zy * v4yz
        + 1.0 * v1zy * v4zy
        + 1.0 * v1zz * v4dd
        - 1.0 * v1zz * v4xx
        - 1.0 * v1zz * v4yy
        + 1.0 * v1zz * v4zz
        + 1.0 * v3dd * v2zz
        + 1.0 * v3dx * v2yz
        - 1.0 * v3dy * v2xz
        - 1.0 * v3dz * v2dz
        - 1.0 * v3xd * v2yz
        - 1.0 * v3xx * v2zz
        - 1.0 * v3xy * v2dz
        + 1.0 * v3xz * v2xz
        + 1.0 * v3yd * v2xz
        + 1.0 * v3yx * v2dz
        - 1.0 * v3yy * v2zz
        + 1.0 * v3yz * v2yz
        - 1.0 * v3zd * v2dz
        + 1.0 * v3zx * v2xz
        + 1.0 * v3zy * v2yz
        + 1.0 * v3zz * v2zz
        )

        # determine overlap for site i
        overlap_i = overlap[i]
        # determine range for inner sum
        Range = size(overlap_i, 1)

        #compute inner sum 
        @inbounds @fastmath for j in 1 : Range

            # read cached values for inner site, respecting mappings
            signs_1 = overlap_i[j][1].signs
            components_1 = overlap_i[j][1].components_1
            site_1 = overlap_i[j][1].Site

            v1xx = signs_1[1] * temp[site_1, components_1[1], 1]
            v1yy = signs_1[2] * temp[site_1, components_1[2], 1]
            v1zz = signs_1[3] * temp[site_1, components_1[3], 1]
            v1xy = signs_1[4] * temp[site_1, components_1[4], 1]
            v1xz = signs_1[5] * temp[site_1, components_1[5], 1]
            v1yz = signs_1[6] * temp[site_1, components_1[6], 1]
            v1yx = signs_1[7] * temp[site_1, components_1[7], 1]
            v1zx = signs_1[8] * temp[site_1, components_1[8], 1]
            v1zy = signs_1[9] * temp[site_1, components_1[9], 1]
            v1dd = signs_1[10] * temp[site_1, components_1[10], 1]
            v1xd = signs_1[11] * temp[site_1, components_1[11], 1]
            v1yd = signs_1[12] * temp[site_1, components_1[12], 1]
            v1zd = signs_1[13] * temp[site_1, components_1[13], 1]
            v1dx = signs_1[14] * temp[site_1, components_1[14], 1]
            v1dy = signs_1[15] * temp[site_1, components_1[15], 1]
            v1dz = signs_1[16] * temp[site_1, components_1[16], 1]

            signs_2 = overlap_i[j][2].signs
            components_2 = overlap_i[j][2].components_1
            site_2 = overlap_i[j][2].Site

            v2xx = signs_2[1]  * temp[site_2, components_2[1], 2]
            v2yy = signs_2[2]  * temp[site_2, components_2[2], 2]
            v2zz = signs_2[3]  * temp[site_2, components_2[3], 2]
            v2xy = signs_2[4]  * temp[site_2, components_2[4], 2]
            v2xz = signs_2[5]  * temp[site_2, components_2[5], 2]
            v2yz = signs_2[6]  * temp[site_2, components_2[6], 2]
            v2yx = signs_2[7]  * temp[site_2, components_2[7], 2]
            v2zx = signs_2[8]  * temp[site_2, components_2[8], 2]
            v2zy = signs_2[9]  * temp[site_2, components_2[9], 2]
            v2dd = signs_2[10] * temp[site_2, components_2[10], 2]
            v2xd = signs_2[11] * temp[site_2, components_2[11], 2]
            v2yd = signs_2[12] * temp[site_2, components_2[12], 2]
            v2zd = signs_2[13] * temp[site_2, components_2[13], 2]
            v2dx = signs_2[14] * temp[site_2, components_2[14], 2]
            v2dy = signs_2[15] * temp[site_2, components_2[15], 2]
            v2dz = signs_2[16] * temp[site_2, components_2[16], 2]

            # compute contribution at inner site
            Γdd += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dd
            + 2.0 * v1dx * v2xd
            + 2.0 * v1dy * v2yd
            + 2.0 * v1dz * v2zd
            )

            Γdx += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dx
            - 2.0 * v1dx * v2xx
            - 2.0 * v1dy * v2yx
            - 2.0 * v1dz * v2zx
            )

            Γdy += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dy
            - 2.0 * v1dx * v2xy
            - 2.0 * v1dy * v2yy
            - 2.0 * v1dz * v2zy
            )

            Γdz += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dz
            - 2.0 * v1dx * v2xz
            - 2.0 * v1dy * v2yz
            - 2.0 * v1dz * v2zz
            )

            Γxd += -p * overlap_i[j][3] * (- 2.0 * v1xd * v2dd
            - 2.0 * v1xx * v2xd
            - 2.0 * v1xy * v2yd
            - 2.0 * v1xz * v2zd
            )

            Γxx += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dx
            - 2.0 * v1xx * v2xx
            - 2.0 * v1xy * v2yx
            - 2.0 * v1xz * v2zx
            )

            Γxy += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dy
            - 2.0 * v1xx * v2xy
            - 2.0 * v1xy * v2yy
            - 2.0 * v1xz * v2zy
            )

            Γxz += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dz
            - 2.0 * v1xx * v2xz
            - 2.0 * v1xy * v2yz
            - 2.0 * v1xz * v2zz
            )

            Γyd += -p * overlap_i[j][3] * (- 2.0 * v1yd * v2dd
            - 2.0 * v1yx * v2xd
            - 2.0 * v1yy * v2yd
            - 2.0 * v1yz * v2zd
            )

            Γyx += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dx
            - 2.0 * v1yx * v2xx
            - 2.0 * v1yy * v2yx
            - 2.0 * v1yz * v2zx
            )

            Γyy += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dy
            - 2.0 * v1yx * v2xy
            - 2.0 * v1yy * v2yy
            - 2.0 * v1yz * v2zy
            )

            Γyz += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dz
            - 2.0 * v1yx * v2xz
            - 2.0 * v1yy * v2yz
            - 2.0 * v1yz * v2zz
            )

            Γzd += -p * overlap_i[j][3] * (- 2.0 * v1zd * v2dd
            - 2.0 * v1zx * v2xd
            - 2.0 * v1zy * v2yd
            - 2.0 * v1zz * v2zd
            )

            Γzx += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dx
            - 2.0 * v1zx * v2xx
            - 2.0 * v1zy * v2yx
            - 2.0 * v1zz * v2zx
            )

            Γzy += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dy
            - 2.0 * v1zx * v2xy
            - 2.0 * v1zy * v2yy
            - 2.0 * v1zz * v2zy
            )

            Γzz += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dz
            - 2.0 * v1zx * v2xz
            - 2.0 * v1zy * v2yz
            - 2.0 * v1zz * v2zz) 
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
        Γdd = -p * (+ 1.0 * v1dd * v4dd
        + 1.0 * v1dd * v4xx
        + 1.0 * v1dd * v4yy
        + 1.0 * v1dd * v4zz
        - 1.0 * v1dx * v4dx
        - 1.0 * v1dx * v4xd        
        - 1.0 * v1dx * v4yz
        + 1.0 * v1dx * v4zy
        - 1.0 * v1dy * v4dy
        + 1.0 * v1dy * v4xz
        - 1.0 * v1dy * v4yd
        - 1.0 * v1dy * v4zx
        - 1.0 * v1dz * v4dz
        - 1.0 * v1dz * v4xy
        + 1.0 * v1dz * v4yx
        - 1.0 * v1dz * v4zd
        + 1.0 * v3dd * v2dd
        - 1.0 * v3dx * v2xd
        - 1.0 * v3dy * v2yd
        - 1.0 * v3dz * v2zd
        - 1.0 * v3xd * v2xd
        + 1.0 * v3xx * v2dd
        + 1.0 * v3xy * v2zd
        - 1.0 * v3xz * v2yd
        - 1.0 * v3yd * v2yd
        - 1.0 * v3yx * v2zd
        + 1.0 * v3yy * v2dd
        + 1.0 * v3yz * v2xd
        - 1.0 * v3zd * v2zd
        + 1.0 * v3zx * v2yd
        - 1.0 * v3zy * v2xd
        + 1.0 * v3zz * v2dd
        )

        Γdx = -p * (+ 1.0 * v1dd * v4dx
        + 1.0 * v1dd * v4xd
        - 1.0 * v1dd * v4yz
        + 1.0 * v1dd * v4zy
        + 1.0 * v1dx * v4dd
        + 1.0 * v1dx * v4xx
        - 1.0 * v1dx * v4yy
        - 1.0 * v1dx * v4zz
        + 1.0 * v1dy * v4dz
        + 1.0 * v1dy * v4xy
        + 1.0 * v1dy * v4yx
        - 1.0 * v1dy * v4zd
        - 1.0 * v1dz * v4dy
        + 1.0 * v1dz * v4xz
        + 1.0 * v1dz * v4yd
        + 1.0 * v1dz * v4zx
        + 1.0 * v3dd * v2dx
        + 1.0 * v3dx * v2xx
        + 1.0 * v3dy * v2yx
        + 1.0 * v3dz * v2zx
        + 1.0 * v3xd * v2xx
        + 1.0 * v3xx * v2dx
        - 1.0 * v3xy * v2zx
        + 1.0 * v3xz * v2yx
        + 1.0 * v3yd * v2yx
        + 1.0 * v3yx * v2zx
        + 1.0 * v3yy * v2dx
        - 1.0 * v3yz * v2xx
        + 1.0 * v3zd * v2zx
        - 1.0 * v3zx * v2yx
        + 1.0 * v3zy * v2xx
        + 1.0 * v3zz * v2dx
        )

        Γdy = -p * (+ 1.0 * v1dd * v4dy
        + 1.0 * v1dd * v4xz
        + 1.0 * v1dd * v4yd
        - 1.0 * v1dd * v4zx
        - 1.0 * v1dx * v4dz
        + 1.0 * v1dx * v4xy
        + 1.0 * v1dx * v4yx
        + 1.0 * v1dx * v4zd
        + 1.0 * v1dy * v4dd
        - 1.0 * v1dy * v4xx
        + 1.0 * v1dy * v4yy
        - 1.0 * v1dy * v4zz
        + 1.0 * v1dz * v4dx
        - 1.0 * v1dz * v4xd
        + 1.0 * v1dz * v4yz
        + 1.0 * v1dz * v4zy
        + 1.0 * v3dd * v2dy
        + 1.0 * v3dx * v2xy
        + 1.0 * v3dy * v2yy
        + 1.0 * v3dz * v2zy
        + 1.0 * v3xd * v2xy
        + 1.0 * v3xx * v2dy
        - 1.0 * v3xy * v2zy
        + 1.0 * v3xz * v2yy
        + 1.0 * v3yd * v2yy
        + 1.0 * v3yx * v2zy
        + 1.0 * v3yy * v2dy
        - 1.0 * v3yz * v2xy
        + 1.0 * v3zd * v2zy
        - 1.0 * v3zx * v2yy
        + 1.0 * v3zy * v2xy
        + 1.0 * v3zz * v2dy
        )

        Γdz = -p * (+ 1.0 * v1dd * v4dz
        - 1.0 * v1dd * v4xy
        + 1.0 * v1dd * v4yx
        + 1.0 * v1dd * v4zd
        + 1.0 * v1dx * v4dy
        + 1.0 * v1dx * v4xz
        - 1.0 * v1dx * v4yd
        + 1.0 * v1dx * v4zx
        - 1.0 * v1dy * v4dx
        + 1.0 * v1dy * v4xd
        + 1.0 * v1dy * v4yz
        + 1.0 * v1dy * v4zy
        + 1.0 * v1dz * v4dd
        - 1.0 * v1dz * v4xx
        - 1.0 * v1dz * v4yy
        + 1.0 * v1dz * v4zz
        + 1.0 * v3dd * v2dz
        + 1.0 * v3dx * v2xz
        + 1.0 * v3dy * v2yz
        + 1.0 * v3dz * v2zz
        + 1.0 * v3xd * v2xz
        + 1.0 * v3xx * v2dz
        - 1.0 * v3xy * v2zz
        + 1.0 * v3xz * v2yz
        + 1.0 * v3yd * v2yz
        + 1.0 * v3yx * v2zz
        + 1.0 * v3yy * v2dz
        - 1.0 * v3yz * v2xz
        + 1.0 * v3zd * v2zz
        - 1.0 * v3zx * v2yz
        + 1.0 * v3zy * v2xz
        + 1.0 * v3zz * v2dz
        )

        Γxd = -p * (+ 1.0 * v1xd * v4dd
        + 1.0 * v1xd * v4xx
        + 1.0 * v1xd * v4yy
        + 1.0 * v1xd * v4zz
        + 1.0 * v1xx * v4dx
        + 1.0 * v1xx * v4xd
        + 1.0 * v1xx * v4yz
        - 1.0 * v1xx * v4zy
        + 1.0 * v1xy * v4dy
        - 1.0 * v1xy * v4xz
        + 1.0 * v1xy * v4yd
        + 1.0 * v1xy * v4zx
        + 1.0 * v1xz * v4dz
        + 1.0 * v1xz * v4xy
        - 1.0 * v1xz * v4yx
        + 1.0 * v1xz * v4zd
        + 1.0 * v3dd * v2xd
        + 1.0 * v3dx * v2dd
        + 1.0 * v3dy * v2zd
        - 1.0 * v3dz * v2yd
        + 1.0 * v3xd * v2dd
        + 1.0 * v3xx * v2xd
        + 1.0 * v3xy * v2yd
        + 1.0 * v3xz * v2zd
        - 1.0 * v3yd * v2zd
        + 1.0 * v3yx * v2yd
        - 1.0 * v3yy * v2xd
        + 1.0 * v3yz * v2dd
        + 1.0 * v3zd * v2yd
        + 1.0 * v3zx * v2zd
        - 1.0 * v3zy * v2dd
        - 1.0 * v3zz * v2xd
        )

        Γxx = -p * (- 1.0 * v1xd * v4dx
        - 1.0 * v1xd * v4xd
        + 1.0 * v1xd * v4yz
        - 1.0 * v1xd * v4zy
        + 1.0 * v1xx * v4dd
        + 1.0 * v1xx * v4xx
        - 1.0 * v1xx * v4yy
        - 1.0 * v1xx * v4zz
        + 1.0 * v1xy * v4dz
        + 1.0 * v1xy * v4xy
        + 1.0 * v1xy * v4yx
        - 1.0 * v1xy * v4zd
        - 1.0 * v1xz * v4dy
        + 1.0 * v1xz * v4xz
        + 1.0 * v1xz * v4yd
        + 1.0 * v1xz * v4zx
        + 1.0 * v3dd * v2xx
        - 1.0 * v3dx * v2dx
        + 1.0 * v3dy * v2zx
        - 1.0 * v3dz * v2yx
        - 1.0 * v3xd * v2dx
        + 1.0 * v3xx * v2xx
        + 1.0 * v3xy * v2yx
        + 1.0 * v3xz * v2zx
        - 1.0 * v3yd * v2zx
        + 1.0 * v3yx * v2yx
        - 1.0 * v3yy * v2xx
        - 1.0 * v3yz * v2dx
        + 1.0 * v3zd * v2yx
        + 1.0 * v3zx * v2zx
        + 1.0 * v3zy * v2dx
        - 1.0 * v3zz * v2xx
        )

        Γxy = -p * (- 1.0 * v1xd * v4dy
        - 1.0 * v1xd * v4xz
        - 1.0 * v1xd * v4yd
        + 1.0 * v1xd * v4zx
        - 1.0 * v1xx * v4dz
        + 1.0 * v1xx * v4xy
        + 1.0 * v1xx * v4yx
        + 1.0 * v1xx * v4zd
        + 1.0 * v1xy * v4dd
        - 1.0 * v1xy * v4xx
        + 1.0 * v1xy * v4yy
        - 1.0 * v1xy * v4zz
        + 1.0 * v1xz * v4dx
        - 1.0 * v1xz * v4xd
        + 1.0 * v1xz * v4yz
        + 1.0 * v1xz * v4zy
        + 1.0 * v3dd * v2xy
        - 1.0 * v3dx * v2dy
        + 1.0 * v3dy * v2zy
        - 1.0 * v3dz * v2yy
        - 1.0 * v3xd * v2dy
        + 1.0 * v3xx * v2xy
        + 1.0 * v3xy * v2yy
        + 1.0 * v3xz * v2zy
        - 1.0 * v3yd * v2zy
        + 1.0 * v3yx * v2yy
        - 1.0 * v3yy * v2xy
        - 1.0 * v3yz * v2dy
        + 1.0 * v3zd * v2yy
        + 1.0 * v3zx * v2zy
        + 1.0 * v3zy * v2dy
        - 1.0 * v3zz * v2xy
        )

        Γxz = -p * (- 1.0 * v1xd * v4dz
        + 1.0 * v1xd * v4xy
        - 1.0 * v1xd * v4yx
        - 1.0 * v1xd * v4zd
        + 1.0 * v1xx * v4dy
        + 1.0 * v1xx * v4xz
        - 1.0 * v1xx * v4yd
        + 1.0 * v1xx * v4zx
        - 1.0 * v1xy * v4dx
        + 1.0 * v1xy * v4xd
        + 1.0 * v1xy * v4yz
        + 1.0 * v1xy * v4zy
        + 1.0 * v1xz * v4dd
        - 1.0 * v1xz * v4xx
        - 1.0 * v1xz * v4yy
        + 1.0 * v1xz * v4zz
        + 1.0 * v3dd * v2xz
        - 1.0 * v3dx * v2dz
        + 1.0 * v3dy * v2zz
        - 1.0 * v3dz * v2yz
        - 1.0 * v3xd * v2dz
        + 1.0 * v3xx * v2xz
        + 1.0 * v3xy * v2yz
        + 1.0 * v3xz * v2zz
        - 1.0 * v3yd * v2zz
        + 1.0 * v3yx * v2yz
        - 1.0 * v3yy * v2xz
        - 1.0 * v3yz * v2dz
        + 1.0 * v3zd * v2yz
        + 1.0 * v3zx * v2zz
        + 1.0 * v3zy * v2dz
        - 1.0 * v3zz * v2xz
        )

        Γyd = -p * (+ 1.0 * v1yd * v4dd
        + 1.0 * v1yd * v4xx
        + 1.0 * v1yd * v4yy
        + 1.0 * v1yd * v4zz
        + 1.0 * v1yx * v4dx
        + 1.0 * v1yx * v4xd
        + 1.0 * v1yx * v4yz
        - 1.0 * v1yx * v4zy
        + 1.0 * v1yy * v4dy
        - 1.0 * v1yy * v4xz
        + 1.0 * v1yy * v4yd
        + 1.0 * v1yy * v4zx
        + 1.0 * v1yz * v4dz
        + 1.0 * v1yz * v4xy
        - 1.0 * v1yz * v4yx
        + 1.0 * v1yz * v4zd
        + 1.0 * v3dd * v2yd
        - 1.0 * v3dx * v2zd
        + 1.0 * v3dy * v2dd
        + 1.0 * v3dz * v2xd
        + 1.0 * v3xd * v2zd
        - 1.0 * v3xx * v2yd
        + 1.0 * v3xy * v2xd
        - 1.0 * v3xz * v2dd
        + 1.0 * v3yd * v2dd
        + 1.0 * v3yx * v2xd
        + 1.0 * v3yy * v2yd
        + 1.0 * v3yz * v2zd
        - 1.0 * v3zd * v2xd
        + 1.0 * v3zx * v2dd
        + 1.0 * v3zy * v2zd
        - 1.0 * v3zz * v2yd
        )

        Γyx = -p * (- 1.0 * v1yd * v4dx
        - 1.0 * v1yd * v4xd
        + 1.0 * v1yd * v4yz
        - 1.0 * v1yd * v4zy
        + 1.0 * v1yx * v4dd
        + 1.0 * v1yx * v4xx
        - 1.0 * v1yx * v4yy
        - 1.0 * v1yx * v4zz
        + 1.0 * v1yy * v4dz
        + 1.0 * v1yy * v4xy
        + 1.0 * v1yy * v4yx
        - 1.0 * v1yy * v4zd
        - 1.0 * v1yz * v4dy
        + 1.0 * v1yz * v4xz
        + 1.0 * v1yz * v4yd
        + 1.0 * v1yz * v4zx
        + 1.0 * v3dd * v2yx
        - 1.0 * v3dx * v2zx
        - 1.0 * v3dy * v2dx
        + 1.0 * v3dz * v2xx
        + 1.0 * v3xd * v2zx
        - 1.0 * v3xx * v2yx
        + 1.0 * v3xy * v2xx
        + 1.0 * v3xz * v2dx
        - 1.0 * v3yd * v2dx
        + 1.0 * v3yx * v2xx
        + 1.0 * v3yy * v2yx
        + 1.0 * v3yz * v2zx
        - 1.0 * v3zd * v2xx
        - 1.0 * v3zx * v2dx
        + 1.0 * v3zy * v2zx
        - 1.0 * v3zz * v2yx
        )

        Γyy = -p * (- 1.0 * v1yd * v4dy
        - 1.0 * v1yd * v4xz
        - 1.0 * v1yd * v4yd
        + 1.0 * v1yd * v4zx
        - 1.0 * v1yx * v4dz
        + 1.0 * v1yx * v4xy
        + 1.0 * v1yx * v4yx
        + 1.0 * v1yx * v4zd
        + 1.0 * v1yy * v4dd
        - 1.0 * v1yy * v4xx
        + 1.0 * v1yy * v4yy
        - 1.0 * v1yy * v4zz
        + 1.0 * v1yz * v4dx
        - 1.0 * v1yz * v4xd
        + 1.0 * v1yz * v4yz
        + 1.0 * v1yz * v4zy
        + 1.0 * v3dd * v2yy
        - 1.0 * v3dx * v2zy
        - 1.0 * v3dy * v2dy
        + 1.0 * v3dz * v2xy
        + 1.0 * v3xd * v2zy
        - 1.0 * v3xx * v2yy
        + 1.0 * v3xy * v2xy
        + 1.0 * v3xz * v2dy
        - 1.0 * v3yd * v2dy
        + 1.0 * v3yx * v2xy
        + 1.0 * v3yy * v2yy
        + 1.0 * v3yz * v2zy
        - 1.0 * v3zd * v2xy
        - 1.0 * v3zx * v2dy
        + 1.0 * v3zy * v2zy
        - 1.0 * v3zz * v2yy
        )

        Γyz = -p * (- 1.0 * v1yd * v4dz
        + 1.0 * v1yd * v4xy
        - 1.0 * v1yd * v4yx
        - 1.0 * v1yd * v4zd
        + 1.0 * v1yx * v4dy
        + 1.0 * v1yx * v4xz
        - 1.0 * v1yx * v4yd
        + 1.0 * v1yx * v4zx
        - 1.0 * v1yy * v4dx
        + 1.0 * v1yy * v4xd
        + 1.0 * v1yy * v4yz
        + 1.0 * v1yy * v4zy
        + 1.0 * v1yz * v4dd
        - 1.0 * v1yz * v4xx
        - 1.0 * v1yz * v4yy
        + 1.0 * v1yz * v4zz
        + 1.0 * v3dd * v2yz
        - 1.0 * v3dx * v2zz
        - 1.0 * v3dy * v2dz
        + 1.0 * v3dz * v2xz
        + 1.0 * v3xd * v2zz
        - 1.0 * v3xx * v2yz
        + 1.0 * v3xy * v2xz
        + 1.0 * v3xz * v2dz
        - 1.0 * v3yd * v2dz
        + 1.0 * v3yx * v2xz
        + 1.0 * v3yy * v2yz
        + 1.0 * v3yz * v2zz
        - 1.0 * v3zd * v2xz
        - 1.0 * v3zx * v2dz
        + 1.0 * v3zy * v2zz
        - 1.0 * v3zz * v2yz
        )

        Γzd = -p * (+ 1.0 * v1zd * v4dd
        + 1.0 * v1zd * v4xx
        + 1.0 * v1zd * v4yy
        + 1.0 * v1zd * v4zz
        + 1.0 * v1zx * v4dx
        + 1.0 * v1zx * v4xd
        + 1.0 * v1zx * v4yz
        - 1.0 * v1zx * v4zy
        + 1.0 * v1zy * v4dy
        - 1.0 * v1zy * v4xz
        + 1.0 * v1zy * v4yd
        + 1.0 * v1zy * v4zx
        + 1.0 * v1zz * v4dz
        + 1.0 * v1zz * v4xy
        - 1.0 * v1zz * v4yx
        + 1.0 * v1zz * v4zd
        + 1.0 * v3dd * v2zd
        + 1.0 * v3dx * v2yd
        - 1.0 * v3dy * v2xd
        + 1.0 * v3dz * v2dd
        - 1.0 * v3xd * v2yd
        - 1.0 * v3xx * v2zd
        + 1.0 * v3xy * v2dd
        + 1.0 * v3xz * v2xd
        + 1.0 * v3yd * v2xd
        - 1.0 * v3yx * v2dd
        - 1.0 * v3yy * v2zd
        + 1.0 * v3yz * v2yd
        + 1.0 * v3zd * v2dd
        + 1.0 * v3zx * v2xd
        + 1.0 * v3zy * v2yd
        + 1.0 * v3zz * v2zd
        )

        Γzx = -p * (- 1.0 * v1zd * v4dx
        - 1.0 * v1zd * v4xd
        + 1.0 * v1zd * v4yz
        - 1.0 * v1zd * v4zy
        + 1.0 * v1zx * v4dd
        + 1.0 * v1zx * v4xx
        - 1.0 * v1zx * v4yy
        - 1.0 * v1zx * v4zz
        + 1.0 * v1zy * v4dz
        + 1.0 * v1zy * v4xy
        + 1.0 * v1zy * v4yx
        - 1.0 * v1zy * v4zd
        - 1.0 * v1zz * v4dy
        + 1.0 * v1zz * v4xz
        + 1.0 * v1zz * v4yd
        + 1.0 * v1zz * v4zx
        + 1.0 * v3dd * v2zx
        + 1.0 * v3dx * v2yx
        - 1.0 * v3dy * v2xx
        - 1.0 * v3dz * v2dx
        - 1.0 * v3xd * v2yx
        - 1.0 * v3xx * v2zx
        - 1.0 * v3xy * v2dx
        + 1.0 * v3xz * v2xx
        + 1.0 * v3yd * v2xx
        + 1.0 * v3yx * v2dx
        - 1.0 * v3yy * v2zx
        + 1.0 * v3yz * v2yx
        - 1.0 * v3zd * v2dx
        + 1.0 * v3zx * v2xx
        + 1.0 * v3zy * v2yx
        + 1.0 * v3zz * v2zx
        )

        Γzy = -p * (- 1.0 * v1zd * v4dy
        - 1.0 * v1zd * v4xz
        - 1.0 * v1zd * v4yd
        + 1.0 * v1zd * v4zx
        - 1.0 * v1zx * v4dz
        + 1.0 * v1zx * v4xy
        + 1.0 * v1zx * v4yx
        + 1.0 * v1zx * v4zd
        + 1.0 * v1zy * v4dd
        - 1.0 * v1zy * v4xx
        + 1.0 * v1zy * v4yy
        - 1.0 * v1zy * v4zz
        + 1.0 * v1zz * v4dx
        - 1.0 * v1zz * v4xd
        + 1.0 * v1zz * v4yz
        + 1.0 * v1zz * v4zy
        + 1.0 * v3dd * v2zy
        + 1.0 * v3dx * v2yy
        - 1.0 * v3dy * v2xy
        - 1.0 * v3dz * v2dy
        - 1.0 * v3xd * v2yy
        - 1.0 * v3xx * v2zy
        - 1.0 * v3xy * v2dy
        + 1.0 * v3xz * v2xy
        + 1.0 * v3yd * v2xy
        + 1.0 * v3yx * v2dy
        - 1.0 * v3yy * v2zy
        + 1.0 * v3yz * v2yy
        - 1.0 * v3zd * v2dy
        + 1.0 * v3zx * v2xy
        + 1.0 * v3zy * v2yy
        + 1.0 * v3zz * v2zy
        )

        Γzz = -p * (- 1.0 * v1zd * v4dz
        + 1.0 * v1zd * v4xy
        - 1.0 * v1zd * v4yx
        - 1.0 * v1zd * v4zd
        + 1.0 * v1zx * v4dy
        + 1.0 * v1zx * v4xz
        - 1.0 * v1zx * v4yd
        + 1.0 * v1zx * v4zx
        - 1.0 * v1zy * v4dx
        + 1.0 * v1zy * v4xd
        + 1.0 * v1zy * v4yz
        + 1.0 * v1zy * v4zy
        + 1.0 * v1zz * v4dd
        - 1.0 * v1zz * v4xx
        - 1.0 * v1zz * v4yy
        + 1.0 * v1zz * v4zz
        + 1.0 * v3dd * v2zz
        + 1.0 * v3dx * v2yz
        - 1.0 * v3dy * v2xz
        - 1.0 * v3dz * v2dz
        - 1.0 * v3xd * v2yz
        - 1.0 * v3xx * v2zz
        - 1.0 * v3xy * v2dz
        + 1.0 * v3xz * v2xz
        + 1.0 * v3yd * v2xz
        + 1.0 * v3yx * v2dz
        - 1.0 * v3yy * v2zz
        + 1.0 * v3yz * v2yz
        - 1.0 * v3zd * v2dz
        + 1.0 * v3zx * v2xz
        + 1.0 * v3zy * v2yz
        + 1.0 * v3zz * v2zz
        )

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
            Γdd += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dd
            + 2.0 * v1dx * v2xd
            + 2.0 * v1dy * v2yd
            + 2.0 * v1dz * v2zd
            )

            Γdx += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dx
            - 2.0 * v1dx * v2xx
            - 2.0 * v1dy * v2yx
            - 2.0 * v1dz * v2zx
            )

            Γdy += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dy
            - 2.0 * v1dx * v2xy
            - 2.0 * v1dy * v2yy
            - 2.0 * v1dz * v2zy
            )

            Γdz += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dz
            - 2.0 * v1dx * v2xz
            - 2.0 * v1dy * v2yz
            - 2.0 * v1dz * v2zz
            )

            Γxd += -p * overlap_i[j][3] * (- 2.0 * v1xd * v2dd
            - 2.0 * v1xx * v2xd
            - 2.0 * v1xy * v2yd
            - 2.0 * v1xz * v2zd
            )

            Γxx += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dx
            - 2.0 * v1xx * v2xx
            - 2.0 * v1xy * v2yx
            - 2.0 * v1xz * v2zx
            )

            Γxy += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dy
            - 2.0 * v1xx * v2xy
            - 2.0 * v1xy * v2yy
            - 2.0 * v1xz * v2zy
            )

            Γxz += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dz
            - 2.0 * v1xx * v2xz
            - 2.0 * v1xy * v2yz
            - 2.0 * v1xz * v2zz
            )

            Γyd += -p * overlap_i[j][3] * (- 2.0 * v1yd * v2dd
            - 2.0 * v1yx * v2xd
            - 2.0 * v1yy * v2yd
            - 2.0 * v1yz * v2zd
            )

            Γyx += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dx
            - 2.0 * v1yx * v2xx
            - 2.0 * v1yy * v2yx
            - 2.0 * v1yz * v2zx
            )

            Γyy += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dy
            - 2.0 * v1yx * v2xy
            - 2.0 * v1yy * v2yy
            - 2.0 * v1yz * v2zy
            )

            Γyz += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dz
            - 2.0 * v1yx * v2xz
            - 2.0 * v1yy * v2yz
            - 2.0 * v1yz * v2zz
            )

            Γzd += -p * overlap_i[j][3] * (- 2.0 * v1zd * v2dd
            - 2.0 * v1zx * v2xd
            - 2.0 * v1zy * v2yd
            - 2.0 * v1zz * v2zd
            )

            Γzx += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dx
            - 2.0 * v1zx * v2xx
            - 2.0 * v1zy * v2yx
            - 2.0 * v1zz * v2zx
            )

            Γzy += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dy
            - 2.0 * v1zx * v2xy
            - 2.0 * v1zy * v2yy
            - 2.0 * v1zz * v2zy
            )

            Γzz += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dz
            - 2.0 * v1zx * v2xz
            - 2.0 * v1zy * v2yz
            - 2.0 * v1zz * v2zz) 
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
                Γdd = -p * (+ 1.0 * v1dd * v4dd
        + 1.0 * v1dd * v4xx
        + 1.0 * v1dd * v4yy
        + 1.0 * v1dd * v4zz
        - 1.0 * v1dx * v4dx
        - 1.0 * v1dx * v4xd        
        - 1.0 * v1dx * v4yz
        + 1.0 * v1dx * v4zy
        - 1.0 * v1dy * v4dy
        + 1.0 * v1dy * v4xz
        - 1.0 * v1dy * v4yd
        - 1.0 * v1dy * v4zx
        - 1.0 * v1dz * v4dz
        - 1.0 * v1dz * v4xy
        + 1.0 * v1dz * v4yx
        - 1.0 * v1dz * v4zd
        + 1.0 * v3dd * v2dd
        - 1.0 * v3dx * v2xd
        - 1.0 * v3dy * v2yd
        - 1.0 * v3dz * v2zd
        - 1.0 * v3xd * v2xd
        + 1.0 * v3xx * v2dd
        + 1.0 * v3xy * v2zd
        - 1.0 * v3xz * v2yd
        - 1.0 * v3yd * v2yd
        - 1.0 * v3yx * v2zd
        + 1.0 * v3yy * v2dd
        + 1.0 * v3yz * v2xd
        - 1.0 * v3zd * v2zd
        + 1.0 * v3zx * v2yd
        - 1.0 * v3zy * v2xd
        + 1.0 * v3zz * v2dd
        )

        Γdx = -p * (+ 1.0 * v1dd * v4dx
        + 1.0 * v1dd * v4xd
        - 1.0 * v1dd * v4yz
        + 1.0 * v1dd * v4zy
        + 1.0 * v1dx * v4dd
        + 1.0 * v1dx * v4xx
        - 1.0 * v1dx * v4yy
        - 1.0 * v1dx * v4zz
        + 1.0 * v1dy * v4dz
        + 1.0 * v1dy * v4xy
        + 1.0 * v1dy * v4yx
        - 1.0 * v1dy * v4zd
        - 1.0 * v1dz * v4dy
        + 1.0 * v1dz * v4xz
        + 1.0 * v1dz * v4yd
        + 1.0 * v1dz * v4zx
        + 1.0 * v3dd * v2dx
        + 1.0 * v3dx * v2xx
        + 1.0 * v3dy * v2yx
        + 1.0 * v3dz * v2zx
        + 1.0 * v3xd * v2xx
        + 1.0 * v3xx * v2dx
        - 1.0 * v3xy * v2zx
        + 1.0 * v3xz * v2yx
        + 1.0 * v3yd * v2yx
        + 1.0 * v3yx * v2zx
        + 1.0 * v3yy * v2dx
        - 1.0 * v3yz * v2xx
        + 1.0 * v3zd * v2zx
        - 1.0 * v3zx * v2yx
        + 1.0 * v3zy * v2xx
        + 1.0 * v3zz * v2dx
        )

        Γdy = -p * (+ 1.0 * v1dd * v4dy
        + 1.0 * v1dd * v4xz
        + 1.0 * v1dd * v4yd
        - 1.0 * v1dd * v4zx
        - 1.0 * v1dx * v4dz
        + 1.0 * v1dx * v4xy
        + 1.0 * v1dx * v4yx
        + 1.0 * v1dx * v4zd
        + 1.0 * v1dy * v4dd
        - 1.0 * v1dy * v4xx
        + 1.0 * v1dy * v4yy
        - 1.0 * v1dy * v4zz
        + 1.0 * v1dz * v4dx
        - 1.0 * v1dz * v4xd
        + 1.0 * v1dz * v4yz
        + 1.0 * v1dz * v4zy
        + 1.0 * v3dd * v2dy
        + 1.0 * v3dx * v2xy
        + 1.0 * v3dy * v2yy
        + 1.0 * v3dz * v2zy
        + 1.0 * v3xd * v2xy
        + 1.0 * v3xx * v2dy
        - 1.0 * v3xy * v2zy
        + 1.0 * v3xz * v2yy
        + 1.0 * v3yd * v2yy
        + 1.0 * v3yx * v2zy
        + 1.0 * v3yy * v2dy
        - 1.0 * v3yz * v2xy
        + 1.0 * v3zd * v2zy
        - 1.0 * v3zx * v2yy
        + 1.0 * v3zy * v2xy
        + 1.0 * v3zz * v2dy
        )

        Γdz = -p * (+ 1.0 * v1dd * v4dz
        - 1.0 * v1dd * v4xy
        + 1.0 * v1dd * v4yx
        + 1.0 * v1dd * v4zd
        + 1.0 * v1dx * v4dy
        + 1.0 * v1dx * v4xz
        - 1.0 * v1dx * v4yd
        + 1.0 * v1dx * v4zx
        - 1.0 * v1dy * v4dx
        + 1.0 * v1dy * v4xd
        + 1.0 * v1dy * v4yz
        + 1.0 * v1dy * v4zy
        + 1.0 * v1dz * v4dd
        - 1.0 * v1dz * v4xx
        - 1.0 * v1dz * v4yy
        + 1.0 * v1dz * v4zz
        + 1.0 * v3dd * v2dz
        + 1.0 * v3dx * v2xz
        + 1.0 * v3dy * v2yz
        + 1.0 * v3dz * v2zz
        + 1.0 * v3xd * v2xz
        + 1.0 * v3xx * v2dz
        - 1.0 * v3xy * v2zz
        + 1.0 * v3xz * v2yz
        + 1.0 * v3yd * v2yz
        + 1.0 * v3yx * v2zz
        + 1.0 * v3yy * v2dz
        - 1.0 * v3yz * v2xz
        + 1.0 * v3zd * v2zz
        - 1.0 * v3zx * v2yz
        + 1.0 * v3zy * v2xz
        + 1.0 * v3zz * v2dz
        )

        Γxd = -p * (+ 1.0 * v1xd * v4dd
        + 1.0 * v1xd * v4xx
        + 1.0 * v1xd * v4yy
        + 1.0 * v1xd * v4zz
        + 1.0 * v1xx * v4dx
        + 1.0 * v1xx * v4xd
        + 1.0 * v1xx * v4yz
        - 1.0 * v1xx * v4zy
        + 1.0 * v1xy * v4dy
        - 1.0 * v1xy * v4xz
        + 1.0 * v1xy * v4yd
        + 1.0 * v1xy * v4zx
        + 1.0 * v1xz * v4dz
        + 1.0 * v1xz * v4xy
        - 1.0 * v1xz * v4yx
        + 1.0 * v1xz * v4zd
        + 1.0 * v3dd * v2xd
        + 1.0 * v3dx * v2dd
        + 1.0 * v3dy * v2zd
        - 1.0 * v3dz * v2yd
        + 1.0 * v3xd * v2dd
        + 1.0 * v3xx * v2xd
        + 1.0 * v3xy * v2yd
        + 1.0 * v3xz * v2zd
        - 1.0 * v3yd * v2zd
        + 1.0 * v3yx * v2yd
        - 1.0 * v3yy * v2xd
        + 1.0 * v3yz * v2dd
        + 1.0 * v3zd * v2yd
        + 1.0 * v3zx * v2zd
        - 1.0 * v3zy * v2dd
        - 1.0 * v3zz * v2xd
        )

        Γxx = -p * (- 1.0 * v1xd * v4dx
        - 1.0 * v1xd * v4xd
        + 1.0 * v1xd * v4yz
        - 1.0 * v1xd * v4zy
        + 1.0 * v1xx * v4dd
        + 1.0 * v1xx * v4xx
        - 1.0 * v1xx * v4yy
        - 1.0 * v1xx * v4zz
        + 1.0 * v1xy * v4dz
        + 1.0 * v1xy * v4xy
        + 1.0 * v1xy * v4yx
        - 1.0 * v1xy * v4zd
        - 1.0 * v1xz * v4dy
        + 1.0 * v1xz * v4xz
        + 1.0 * v1xz * v4yd
        + 1.0 * v1xz * v4zx
        + 1.0 * v3dd * v2xx
        - 1.0 * v3dx * v2dx
        + 1.0 * v3dy * v2zx
        - 1.0 * v3dz * v2yx
        - 1.0 * v3xd * v2dx
        + 1.0 * v3xx * v2xx
        + 1.0 * v3xy * v2yx
        + 1.0 * v3xz * v2zx
        - 1.0 * v3yd * v2zx
        + 1.0 * v3yx * v2yx
        - 1.0 * v3yy * v2xx
        - 1.0 * v3yz * v2dx
        + 1.0 * v3zd * v2yx
        + 1.0 * v3zx * v2zx
        + 1.0 * v3zy * v2dx
        - 1.0 * v3zz * v2xx
        )

        Γxy = -p * (- 1.0 * v1xd * v4dy
        - 1.0 * v1xd * v4xz
        - 1.0 * v1xd * v4yd
        + 1.0 * v1xd * v4zx
        - 1.0 * v1xx * v4dz
        + 1.0 * v1xx * v4xy
        + 1.0 * v1xx * v4yx
        + 1.0 * v1xx * v4zd
        + 1.0 * v1xy * v4dd
        - 1.0 * v1xy * v4xx
        + 1.0 * v1xy * v4yy
        - 1.0 * v1xy * v4zz
        + 1.0 * v1xz * v4dx
        - 1.0 * v1xz * v4xd
        + 1.0 * v1xz * v4yz
        + 1.0 * v1xz * v4zy
        + 1.0 * v3dd * v2xy
        - 1.0 * v3dx * v2dy
        + 1.0 * v3dy * v2zy
        - 1.0 * v3dz * v2yy
        - 1.0 * v3xd * v2dy
        + 1.0 * v3xx * v2xy
        + 1.0 * v3xy * v2yy
        + 1.0 * v3xz * v2zy
        - 1.0 * v3yd * v2zy
        + 1.0 * v3yx * v2yy
        - 1.0 * v3yy * v2xy
        - 1.0 * v3yz * v2dy
        + 1.0 * v3zd * v2yy
        + 1.0 * v3zx * v2zy
        + 1.0 * v3zy * v2dy
        - 1.0 * v3zz * v2xy
        )

        Γxz = -p * (- 1.0 * v1xd * v4dz
        + 1.0 * v1xd * v4xy
        - 1.0 * v1xd * v4yx
        - 1.0 * v1xd * v4zd
        + 1.0 * v1xx * v4dy
        + 1.0 * v1xx * v4xz
        - 1.0 * v1xx * v4yd
        + 1.0 * v1xx * v4zx
        - 1.0 * v1xy * v4dx
        + 1.0 * v1xy * v4xd
        + 1.0 * v1xy * v4yz
        + 1.0 * v1xy * v4zy
        + 1.0 * v1xz * v4dd
        - 1.0 * v1xz * v4xx
        - 1.0 * v1xz * v4yy
        + 1.0 * v1xz * v4zz
        + 1.0 * v3dd * v2xz
        - 1.0 * v3dx * v2dz
        + 1.0 * v3dy * v2zz
        - 1.0 * v3dz * v2yz
        - 1.0 * v3xd * v2dz
        + 1.0 * v3xx * v2xz
        + 1.0 * v3xy * v2yz
        + 1.0 * v3xz * v2zz
        - 1.0 * v3yd * v2zz
        + 1.0 * v3yx * v2yz
        - 1.0 * v3yy * v2xz
        - 1.0 * v3yz * v2dz
        + 1.0 * v3zd * v2yz
        + 1.0 * v3zx * v2zz
        + 1.0 * v3zy * v2dz
        - 1.0 * v3zz * v2xz
        )

        Γyd = -p * (+ 1.0 * v1yd * v4dd
        + 1.0 * v1yd * v4xx
        + 1.0 * v1yd * v4yy
        + 1.0 * v1yd * v4zz
        + 1.0 * v1yx * v4dx
        + 1.0 * v1yx * v4xd
        + 1.0 * v1yx * v4yz
        - 1.0 * v1yx * v4zy
        + 1.0 * v1yy * v4dy
        - 1.0 * v1yy * v4xz
        + 1.0 * v1yy * v4yd
        + 1.0 * v1yy * v4zx
        + 1.0 * v1yz * v4dz
        + 1.0 * v1yz * v4xy
        - 1.0 * v1yz * v4yx
        + 1.0 * v1yz * v4zd
        + 1.0 * v3dd * v2yd
        - 1.0 * v3dx * v2zd
        + 1.0 * v3dy * v2dd
        + 1.0 * v3dz * v2xd
        + 1.0 * v3xd * v2zd
        - 1.0 * v3xx * v2yd
        + 1.0 * v3xy * v2xd
        - 1.0 * v3xz * v2dd
        + 1.0 * v3yd * v2dd
        + 1.0 * v3yx * v2xd
        + 1.0 * v3yy * v2yd
        + 1.0 * v3yz * v2zd
        - 1.0 * v3zd * v2xd
        + 1.0 * v3zx * v2dd
        + 1.0 * v3zy * v2zd
        - 1.0 * v3zz * v2yd
        )

        Γyx = -p * (- 1.0 * v1yd * v4dx
        - 1.0 * v1yd * v4xd
        + 1.0 * v1yd * v4yz
        - 1.0 * v1yd * v4zy
        + 1.0 * v1yx * v4dd
        + 1.0 * v1yx * v4xx
        - 1.0 * v1yx * v4yy
        - 1.0 * v1yx * v4zz
        + 1.0 * v1yy * v4dz
        + 1.0 * v1yy * v4xy
        + 1.0 * v1yy * v4yx
        - 1.0 * v1yy * v4zd
        - 1.0 * v1yz * v4dy
        + 1.0 * v1yz * v4xz
        + 1.0 * v1yz * v4yd
        + 1.0 * v1yz * v4zx
        + 1.0 * v3dd * v2yx
        - 1.0 * v3dx * v2zx
        - 1.0 * v3dy * v2dx
        + 1.0 * v3dz * v2xx
        + 1.0 * v3xd * v2zx
        - 1.0 * v3xx * v2yx
        + 1.0 * v3xy * v2xx
        + 1.0 * v3xz * v2dx
        - 1.0 * v3yd * v2dx
        + 1.0 * v3yx * v2xx
        + 1.0 * v3yy * v2yx
        + 1.0 * v3yz * v2zx
        - 1.0 * v3zd * v2xx
        - 1.0 * v3zx * v2dx
        + 1.0 * v3zy * v2zx
        - 1.0 * v3zz * v2yx
        )

        Γyy = -p * (- 1.0 * v1yd * v4dy
        - 1.0 * v1yd * v4xz
        - 1.0 * v1yd * v4yd
        + 1.0 * v1yd * v4zx
        - 1.0 * v1yx * v4dz
        + 1.0 * v1yx * v4xy
        + 1.0 * v1yx * v4yx
        + 1.0 * v1yx * v4zd
        + 1.0 * v1yy * v4dd
        - 1.0 * v1yy * v4xx
        + 1.0 * v1yy * v4yy
        - 1.0 * v1yy * v4zz
        + 1.0 * v1yz * v4dx
        - 1.0 * v1yz * v4xd
        + 1.0 * v1yz * v4yz
        + 1.0 * v1yz * v4zy
        + 1.0 * v3dd * v2yy
        - 1.0 * v3dx * v2zy
        - 1.0 * v3dy * v2dy
        + 1.0 * v3dz * v2xy
        + 1.0 * v3xd * v2zy
        - 1.0 * v3xx * v2yy
        + 1.0 * v3xy * v2xy
        + 1.0 * v3xz * v2dy
        - 1.0 * v3yd * v2dy
        + 1.0 * v3yx * v2xy
        + 1.0 * v3yy * v2yy
        + 1.0 * v3yz * v2zy
        - 1.0 * v3zd * v2xy
        - 1.0 * v3zx * v2dy
        + 1.0 * v3zy * v2zy
        - 1.0 * v3zz * v2yy
        )

        Γyz = -p * (- 1.0 * v1yd * v4dz
        + 1.0 * v1yd * v4xy
        - 1.0 * v1yd * v4yx
        - 1.0 * v1yd * v4zd
        + 1.0 * v1yx * v4dy
        + 1.0 * v1yx * v4xz
        - 1.0 * v1yx * v4yd
        + 1.0 * v1yx * v4zx
        - 1.0 * v1yy * v4dx
        + 1.0 * v1yy * v4xd
        + 1.0 * v1yy * v4yz
        + 1.0 * v1yy * v4zy
        + 1.0 * v1yz * v4dd
        - 1.0 * v1yz * v4xx
        - 1.0 * v1yz * v4yy
        + 1.0 * v1yz * v4zz
        + 1.0 * v3dd * v2yz
        - 1.0 * v3dx * v2zz
        - 1.0 * v3dy * v2dz
        + 1.0 * v3dz * v2xz
        + 1.0 * v3xd * v2zz
        - 1.0 * v3xx * v2yz
        + 1.0 * v3xy * v2xz
        + 1.0 * v3xz * v2dz
        - 1.0 * v3yd * v2dz
        + 1.0 * v3yx * v2xz
        + 1.0 * v3yy * v2yz
        + 1.0 * v3yz * v2zz
        - 1.0 * v3zd * v2xz
        - 1.0 * v3zx * v2dz
        + 1.0 * v3zy * v2zz
        - 1.0 * v3zz * v2yz
        )

        Γzd = -p * (+ 1.0 * v1zd * v4dd
        + 1.0 * v1zd * v4xx
        + 1.0 * v1zd * v4yy
        + 1.0 * v1zd * v4zz
        + 1.0 * v1zx * v4dx
        + 1.0 * v1zx * v4xd
        + 1.0 * v1zx * v4yz
        - 1.0 * v1zx * v4zy
        + 1.0 * v1zy * v4dy
        - 1.0 * v1zy * v4xz
        + 1.0 * v1zy * v4yd
        + 1.0 * v1zy * v4zx
        + 1.0 * v1zz * v4dz
        + 1.0 * v1zz * v4xy
        - 1.0 * v1zz * v4yx
        + 1.0 * v1zz * v4zd
        + 1.0 * v3dd * v2zd
        + 1.0 * v3dx * v2yd
        - 1.0 * v3dy * v2xd
        + 1.0 * v3dz * v2dd
        - 1.0 * v3xd * v2yd
        - 1.0 * v3xx * v2zd
        + 1.0 * v3xy * v2dd
        + 1.0 * v3xz * v2xd
        + 1.0 * v3yd * v2xd
        - 1.0 * v3yx * v2dd
        - 1.0 * v3yy * v2zd
        + 1.0 * v3yz * v2yd
        + 1.0 * v3zd * v2dd
        + 1.0 * v3zx * v2xd
        + 1.0 * v3zy * v2yd
        + 1.0 * v3zz * v2zd
        )

        Γzx = -p * (- 1.0 * v1zd * v4dx
        - 1.0 * v1zd * v4xd
        + 1.0 * v1zd * v4yz
        - 1.0 * v1zd * v4zy
        + 1.0 * v1zx * v4dd
        + 1.0 * v1zx * v4xx
        - 1.0 * v1zx * v4yy
        - 1.0 * v1zx * v4zz
        + 1.0 * v1zy * v4dz
        + 1.0 * v1zy * v4xy
        + 1.0 * v1zy * v4yx
        - 1.0 * v1zy * v4zd
        - 1.0 * v1zz * v4dy
        + 1.0 * v1zz * v4xz
        + 1.0 * v1zz * v4yd
        + 1.0 * v1zz * v4zx
        + 1.0 * v3dd * v2zx
        + 1.0 * v3dx * v2yx
        - 1.0 * v3dy * v2xx
        - 1.0 * v3dz * v2dx
        - 1.0 * v3xd * v2yx
        - 1.0 * v3xx * v2zx
        - 1.0 * v3xy * v2dx
        + 1.0 * v3xz * v2xx
        + 1.0 * v3yd * v2xx
        + 1.0 * v3yx * v2dx
        - 1.0 * v3yy * v2zx
        + 1.0 * v3yz * v2yx
        - 1.0 * v3zd * v2dx
        + 1.0 * v3zx * v2xx
        + 1.0 * v3zy * v2yx
        + 1.0 * v3zz * v2zx
        )

        Γzy = -p * (- 1.0 * v1zd * v4dy
        - 1.0 * v1zd * v4xz
        - 1.0 * v1zd * v4yd
        + 1.0 * v1zd * v4zx
        - 1.0 * v1zx * v4dz
        + 1.0 * v1zx * v4xy
        + 1.0 * v1zx * v4yx
        + 1.0 * v1zx * v4zd
        + 1.0 * v1zy * v4dd
        - 1.0 * v1zy * v4xx
        + 1.0 * v1zy * v4yy
        - 1.0 * v1zy * v4zz
        + 1.0 * v1zz * v4dx
        - 1.0 * v1zz * v4xd
        + 1.0 * v1zz * v4yz
        + 1.0 * v1zz * v4zy
        + 1.0 * v3dd * v2zy
        + 1.0 * v3dx * v2yy
        - 1.0 * v3dy * v2xy
        - 1.0 * v3dz * v2dy
        - 1.0 * v3xd * v2yy
        - 1.0 * v3xx * v2zy
        - 1.0 * v3xy * v2dy
        + 1.0 * v3xz * v2xy
        + 1.0 * v3yd * v2xy
        + 1.0 * v3yx * v2dy
        - 1.0 * v3yy * v2zy
        + 1.0 * v3yz * v2yy
        - 1.0 * v3zd * v2dy
        + 1.0 * v3zx * v2xy
        + 1.0 * v3zy * v2yy
        + 1.0 * v3zz * v2zy
        )

        Γzz = -p * (- 1.0 * v1zd * v4dz
        + 1.0 * v1zd * v4xy
        - 1.0 * v1zd * v4yx
        - 1.0 * v1zd * v4zd
        + 1.0 * v1zx * v4dy
        + 1.0 * v1zx * v4xz
        - 1.0 * v1zx * v4yd
        + 1.0 * v1zx * v4zx
        - 1.0 * v1zy * v4dx
        + 1.0 * v1zy * v4xd
        + 1.0 * v1zy * v4yz
        + 1.0 * v1zy * v4zy
        + 1.0 * v1zz * v4dd
        - 1.0 * v1zz * v4xx
        - 1.0 * v1zz * v4yy
        + 1.0 * v1zz * v4zz
        + 1.0 * v3dd * v2zz
        + 1.0 * v3dx * v2yz
        - 1.0 * v3dy * v2xz
        - 1.0 * v3dz * v2dz
        - 1.0 * v3xd * v2yz
        - 1.0 * v3xx * v2zz
        - 1.0 * v3xy * v2dz
        + 1.0 * v3xz * v2xz
        + 1.0 * v3yd * v2xz
        + 1.0 * v3yx * v2dz
        - 1.0 * v3yy * v2zz
        + 1.0 * v3yz * v2yz
        - 1.0 * v3zd * v2dz
        + 1.0 * v3zx * v2xz
        + 1.0 * v3zy * v2yz
        + 1.0 * v3zz * v2zz
        )

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
            Γdd += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dd
            + 2.0 * v1dx * v2xd
            + 2.0 * v1dy * v2yd
            + 2.0 * v1dz * v2zd
            )

            Γdx += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dx
            - 2.0 * v1dx * v2xx
            - 2.0 * v1dy * v2yx
            - 2.0 * v1dz * v2zx
            )

            Γdy += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dy
            - 2.0 * v1dx * v2xy
            - 2.0 * v1dy * v2yy
            - 2.0 * v1dz * v2zy
            )

            Γdz += -p * overlap_i[j][3] * (- 2.0 * v1dd * v2dz
            - 2.0 * v1dx * v2xz
            - 2.0 * v1dy * v2yz
            - 2.0 * v1dz * v2zz
            )

            Γxd += -p * overlap_i[j][3] * (- 2.0 * v1xd * v2dd
            - 2.0 * v1xx * v2xd
            - 2.0 * v1xy * v2yd
            - 2.0 * v1xz * v2zd
            )

            Γxx += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dx
            - 2.0 * v1xx * v2xx
            - 2.0 * v1xy * v2yx
            - 2.0 * v1xz * v2zx
            )

            Γxy += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dy
            - 2.0 * v1xx * v2xy
            - 2.0 * v1xy * v2yy
            - 2.0 * v1xz * v2zy
            )

            Γxz += -p * overlap_i[j][3] * (+ 2.0 * v1xd * v2dz
            - 2.0 * v1xx * v2xz
            - 2.0 * v1xy * v2yz
            - 2.0 * v1xz * v2zz
            )

            Γyd += -p * overlap_i[j][3] * (- 2.0 * v1yd * v2dd
            - 2.0 * v1yx * v2xd
            - 2.0 * v1yy * v2yd
            - 2.0 * v1yz * v2zd
            )

            Γyx += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dx
            - 2.0 * v1yx * v2xx
            - 2.0 * v1yy * v2yx
            - 2.0 * v1yz * v2zx
            )

            Γyy += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dy
            - 2.0 * v1yx * v2xy
            - 2.0 * v1yy * v2yy
            - 2.0 * v1yz * v2zy
            )

            Γyz += -p * overlap_i[j][3] * (+ 2.0 * v1yd * v2dz
            - 2.0 * v1yx * v2xz
            - 2.0 * v1yy * v2yz
            - 2.0 * v1yz * v2zz
            )

            Γzd += -p * overlap_i[j][3] * (- 2.0 * v1zd * v2dd
            - 2.0 * v1zx * v2xd
            - 2.0 * v1zy * v2yd
            - 2.0 * v1zz * v2zd
            )

            Γzx += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dx
            - 2.0 * v1zx * v2xx
            - 2.0 * v1zy * v2yx
            - 2.0 * v1zz * v2zx
            )

            Γzy += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dy
            - 2.0 * v1zx * v2xy
            - 2.0 * v1zy * v2yy
            - 2.0 * v1zz * v2zy
            )

            Γzz += -p * overlap_i[j][3] * (+ 2.0 * v1zd * v2dz
            - 2.0 * v1zx * v2xz
            - 2.0 * v1zy * v2yz
            - 2.0 * v1zz * v2zz) 
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