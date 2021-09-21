# BSE kernel for the u channel
function compute_u_BSE!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64,
    vu   :: Float64,
    vup  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_u1_dm,
    a    :: Action_u1_dm,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v - 0.5 * u, v + 0.5 * u, m, a)

    # get buffers for left vertex
    bs1 = get_buffers_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffers_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = ntuple(x -> get_buffer_empty(), 6)

    # get buffers for right vertex
    bs2 = get_buffers_s( v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m)
    bt2 = get_buffers_t(-v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m)
    bu2 = get_buffers_u(u, v, vup, m)

    # cache vertex values for all lattice sites
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, ch_u = false)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions to Γ[comp] for all lattice sites
    if comp == 1 
        compute_u_kernel_xx!(buff, p, dv, r, temp)
    elseif comp == 2 
        compute_u_kernel_zz!(buff, p, dv, r, temp)
    elseif comp == 3 
        compute_u_kernel_DM!(buff, p, dv, r, temp)
    elseif comp == 4
        compute_u_kernel_dd!(buff, p, dv, r, temp)
    elseif comp == 5
        compute_u_kernel_zd!(buff, p, dv, r, temp)
    else 
        compute_u_kernel_dz!(buff, p, dv, r, temp)
    end

    return nothing
end