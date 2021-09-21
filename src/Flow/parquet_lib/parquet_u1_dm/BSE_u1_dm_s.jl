# BSE kernel for the s channel
function compute_s_BSE!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_u1_dm,
    a    :: Action_u1_dm,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for left vertex
    bs1 = ntuple(x -> get_buffer_empty(), 6)
    bt1 = get_buffers_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffers_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffers_s(s, v, vsp, m)
    bt2 = get_buffers_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffers_u( v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, ch_s = false)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1 
        compute_s_kernel_xx!(buff, p, dv, r, temp)
    elseif comp == 2 
        compute_s_kernel_zz!(buff, p, dv, r, temp)
    elseif comp == 3 
        compute_s_kernel_DM!(buff, p, dv, r, temp)
    elseif comp == 4
        compute_s_kernel_dd!(buff, p, dv, r, temp)
    elseif comp == 5
        compute_s_kernel_zd!(buff, p, dv, r, temp)
    else 
        compute_s_kernel_dz!(buff, p, dv, r, temp)
    end

    return nothing
end