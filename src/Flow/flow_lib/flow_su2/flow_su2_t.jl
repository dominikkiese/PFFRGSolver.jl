# Katanin kernel
function compute_t_kat!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_su2,
    a    :: Action_su2,
    da   :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = get_propagator_kat(Λ, v + 0.5 * t, v - 0.5 * t, m, a, da) + get_propagator_kat(Λ, v - 0.5 * t, v + 0.5 * t, m, a, da)

    # get buffers for left non-local vertex
    bs1 = get_buffers_s( v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffers_t(t, vt, v, m)
    bu1 = get_buffers_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * ( t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffers_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffers_t(t, v, vtp, m)
    bu2 = get_buffers_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * ( t + v + vtp), m)    

    # get buffers for local left vertex
    bs3 = get_buffers_s( v + vt, 0.5 * (t + v - vt), 0.5 * (-t + v - vt), m)
    bt3 = get_buffers_t(-v + vt, 0.5 * (t + v + vt), 0.5 * (-t + v + vt), m)
    bu3 = get_buffers_u(t, v, vt, m)

    # get buffers for local right vertex
    bs4 = get_buffers_s(v + vtp, 0.5 * (t - v + vtp), 0.5 * (-t - v + vtp), m)
    bt4 = get_buffers_t(v - vtp, 0.5 * (t + v + vtp), 0.5 * (-t + v + vtp), m)
    bu4 = get_buffers_u(t, vtp, v, m)

    # cache local vertex values
    v3 = get_Γ(1, bs3, bt3, bu3, r, a)
    v4 = get_Γ(1, bs4, bt4, bu4, r, a)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1
        # cache vertex values for all lattice sites
        get_Γ_avx!(1, r, bs1, bt1, bu1, a, temp, 1)
        get_Γ_avx!(1, r, bs2, bt2, bu2, a, temp, 2)

        compute_t_kernel_spin!(buff, p, dv, v3, v4, a.S, r, temp)
    else 
        # cache vertex values for all lattice sites
        get_Γ_avx!(2, r, bs1, bt1, bu1, a, temp, 1)
        get_Γ_avx!(2, r, bs2, bt2, bu2, a, temp, 2)

        compute_t_kernel_dens!(buff, p, dv, v3, v4, a.S, r, temp)
    end

    return nothing
end





# left kernel (right part obtained by symmetries)
function compute_t_left!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_su2,
    a    :: Action_su2,
    da   :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)

    # get buffers for left non-local vertex
    bs1 = get_buffers_s( v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = ntuple(x -> get_buffer_empty(), 2)
    bu1 = get_buffers_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * ( t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffers_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffers_t(t, v, vtp, m)
    bu2 = get_buffers_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * ( t + v + vtp), m)

    # get buffers for local left vertex
    bs3 = get_buffers_s( v + vt, 0.5 * (t + v - vt), 0.5 * (-t + v - vt), m)
    bt3 = get_buffers_t(-v + vt, 0.5 * (t + v + vt), 0.5 * (-t + v + vt), m)
    bu3 = ntuple(x -> get_buffer_empty(), 2)
    
    # get buffers for local right vertex
    bs4 = get_buffers_s(v + vtp, 0.5 * (t - v + vtp), 0.5 * (-t - v + vtp), m)
    bt4 = get_buffers_t(v - vtp, 0.5 * (t + v + vtp), 0.5 * (-t + v + vtp), m)
    bu4 = get_buffers_u(t, vtp, v, m)

    # cache local vertex values
    v3 = get_Γ(1, bs3, bt3, bu3, r, da, ch_u = false)
    v4 = get_Γ(1, bs4, bt4, bu4, r,  a)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1
        # cache vertex values for all lattice sites
        get_Γ_avx!(1, r, bs1, bt1, bu1, da, temp, 1, ch_t = false)
        get_Γ_avx!(1, r, bs2, bt2, bu2,  a, temp, 2)

        compute_t_kernel_spin!(buff, p, dv, v3, v4, a.S, r, temp)
    else 
        # cache vertex values for all lattice sites
        get_Γ_avx!(2, r, bs1, bt1, bu1, da, temp, 1, ch_t = false)
        get_Γ_avx!(2, r, bs2, bt2, bu2,  a, temp, 2)

        compute_t_kernel_dens!(buff, p, dv, v3, v4, a.S, r, temp)
    end

    return nothing
end





# central kernel
function compute_t_central!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_su2,
    a    :: Action_su2,
    da_l :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)

    # get buffers for left non-local vertex
    bs1 = get_buffers_s( v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffers_t(t, vt, v, m)
    bu1 = get_buffers_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * ( t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = ntuple(x -> get_buffer_empty(), 2)
    bt2 = get_buffers_t(t, v, vtp, m)
    bu2 = ntuple(x -> get_buffer_empty(), 2)

    # get buffers for local left vertex
    bs3 = get_buffers_s( v + vt, 0.5 * (t + v - vt), 0.5 * (-t + v - vt), m)
    bt3 = get_buffers_t(-v + vt, 0.5 * (t + v + vt), 0.5 * (-t + v + vt), m)
    bu3 = get_buffers_u(t, v, vt, m)

    # get buffers for local right vertex
    bs4 = ntuple(x -> get_buffer_empty(), 2)
    bt4 = ntuple(x -> get_buffer_empty(), 2)
    bu4 = get_buffers_u(t, vtp, v, m)

    # cache local vertex values
    v3 = get_Γ(1, bs3, bt3, bu3, r, a)
    v4 = get_Γ(1, bs4, bt4, bu4, r, da_l, ch_s = false, ch_t = false)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1
        # cache vertex values for all lattice sites
        get_Γ_avx!(1, r, bs1, bt1, bu1,    a, temp, 1)
        get_Γ_avx!(1, r, bs2, bt2, bu2, da_l, temp, 2, ch_s = false, ch_u = false)

        compute_t_kernel_spin!(buff, p, dv, v3, v4, a.S, r, temp)
    else 
        # cache vertex values for all lattice sites
        get_Γ_avx!(2, r, bs1, bt1, bu1,    a, temp, 1)
        get_Γ_avx!(2, r, bs2, bt2, bu2, da_l, temp, 2, ch_s = false, ch_u = false)

        compute_t_kernel_dens!(buff, p, dv, v3, v4, a.S, r, temp)
    end

    return nothing
end