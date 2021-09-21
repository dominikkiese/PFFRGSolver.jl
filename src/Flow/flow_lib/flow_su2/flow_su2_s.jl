# Katanin kernel
function compute_s_kat!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_su2,
    a    :: Action_su2,
    da   :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = get_propagator_kat(Λ, v + 0.5 * s, 0.5 * s - v, m, a, da) + get_propagator_kat(Λ, 0.5 * s - v, v + 0.5 * s, m, a, da)

    # get buffers for left vertex
    bs1 = get_buffers_s(s, vs, -v, m)
    bt1 = get_buffers_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffers_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffers_s(s, v, vsp, m)
    bt2 = get_buffers_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffers_u( v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1
        compute_s_kernel_spin!(buff, p, dv, r, temp)
    else 
        compute_s_kernel_dens!(buff, p, dv, r, temp)
    end

    return nothing
end





# left kernel (right part obtained by symmetries)
function compute_s_left!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_su2,
    a    :: Action_su2,
    da   :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for left vertex
    bs1 = ntuple(x -> get_buffer_empty(), 2)
    bt1 = get_buffers_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffers_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffers_s(s, v, vsp, m)
    bt2 = get_buffers_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffers_u( v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites
    get_Γ_avx!(r, bs1, bt1, bu1, da, temp, 1, ch_s = false)
    get_Γ_avx!(r, bs2, bt2, bu2,  a, temp, 2)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1
        compute_s_kernel_spin!(buff, p, dv, r, temp)
    else 
        compute_s_kernel_dens!(buff, p, dv, r, temp)
    end

    return nothing
end





# central kernel
function compute_s_central!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_su2,
    a    :: Action_su2,
    da_l :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for left vertex
    bs1 = get_buffers_s(s, vs, -v, m)
    bt1 = get_buffers_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffers_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffers_s(s, v, vsp, m)
    bt2 = ntuple(x -> get_buffer_empty(), 2)
    bu2 = ntuple(x -> get_buffer_empty(), 2)

    # cache vertex values for all lattice sites
    get_Γ_avx!(r, bs1, bt1, bu1,    a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, da_l, temp, 2, ch_t = false, ch_u = false)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1
        compute_s_kernel_spin!(buff, p, dv, r, temp)
    else 
        compute_s_kernel_dens!(buff, p, dv, r, temp)
    end

    return nothing
end