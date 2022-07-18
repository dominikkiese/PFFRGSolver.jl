# generate correlation dummy for su2-hkg symmetry
function get_χ_su2_hkg_empty(
    r :: Reduced_lattice,
    m :: Mesh
    ) :: Vector{Matrix{Float64}}

    χxx = zeros(Float64, length(r.sites), m.num_χ)
    χyy = zeros(Float64, length(r.sites), m.num_χ)
    χzz = zeros(Float64, length(r.sites), m.num_χ)
    χxy = zeros(Float64, length(r.sites), m.num_χ)
    χxz = zeros(Float64, length(r.sites), m.num_χ)
    χyz = zeros(Float64, length(r.sites), m.num_χ)
    χyx = zeros(Float64, length(r.sites), m.num_χ)
    χzx = zeros(Float64, length(r.sites), m.num_χ)
    χzy = zeros(Float64, length(r.sites), m.num_χ)
    χ   = Matrix{Float64}[χxx, χyy, χzz, χxy, χxz, χyz, χyx, χzx, χzy]

    return χ 
end

# xx kernel for double integral
function compute_χ_kernel_xx!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(1, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vxx = get_Γ_comp(1, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyy = get_Γ_comp(2, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vdd = get_Γ_comp(10, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vzz = get_Γ_comp(3, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vyy + vzz - vxx - vdd) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# yy kernel for double integral
function compute_χ_kernel_yy!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(2, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vxx = get_Γ_comp(1, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyy = get_Γ_comp(2, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vdd = get_Γ_comp(10, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vzz = get_Γ_comp(3, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vzz - vyy + vxx - vdd) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# zz kernel for double integral
function compute_χ_kernel_zz!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(3, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vxx = get_Γ_comp(1, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyy = get_Γ_comp(2, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vdd = get_Γ_comp(10, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vzz = get_Γ_comp(3, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vyy - vzz + vxx - vdd) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# xy kernel for double integral
function compute_χ_kernel_xy!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(4, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vdz = get_Γ_comp(16, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vzd = get_Γ_comp(13, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vxy = get_Γ_comp(4, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyx = get_Γ_comp(7, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vzd - vdz - vxy - vyx) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# xz kernel for double integral
function compute_χ_kernel_xz!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(5, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vzx = get_Γ_comp(8, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyd = get_Γ_comp(12, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vdy = get_Γ_comp(15, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vxz = get_Γ_comp(5, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vdy - vxz - vyd - vzx) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# yz kernel for double integral
function compute_χ_kernel_yz!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(6, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vdx = get_Γ_comp(14, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vxd = get_Γ_comp(11, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyz = get_Γ_comp(6, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vzy = get_Γ_comp(9, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vxd - vdx - vyz - vzy) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# yx kernel for double integral
function compute_χ_kernel_yx!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(7, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vdz = get_Γ_comp(16, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vzd = get_Γ_comp(13, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vxy = get_Γ_comp(4, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyx = get_Γ_comp(7, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vdz - vzd - vxy - vyx) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# zx kernel for double integral
function compute_χ_kernel_zx!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(8, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vzx = get_Γ_comp(8, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyd = get_Γ_comp(12, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vdy = get_Γ_comp(15, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vxz = get_Γ_comp(5, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vyd - vzx - vdy - vxz) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# zy kernel for double integral
function compute_χ_kernel_zy!(
    Λ    :: Float64,
    site :: Int64,
    w    :: Float64,
    vv   :: Matrix{Float64},
    buff :: Vector{Float64},
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2_hkg
    )    :: Nothing

    for i in eachindex(buff)
        # map integration arguments and modify increments
        v   = (2.0 * vv[1, i] - 1.0) / ((1.0 - vv[1, i]) * vv[1, i])
        vp  = (2.0 * vv[2, i] - 1.0) / ((1.0 - vv[2, i]) * vv[2, i])
        dv  = (2.0 * vv[1, i] * vv[1, i] - 2.0 * vv[1, i] + 1) / ((1.0 - vv[1, i]) * (1.0 - vv[1, i]) * vv[1, i] * vv[1, i])
        dvp = (2.0 * vv[2, i] * vv[2, i] - 2.0 * vv[2, i] + 1) / ((1.0 - vv[2, i]) * (1.0 - vv[2, i]) * vv[2, i] * vv[2, i])

        # get buffers for non-local term
        bs1 = get_buffer_s(v + vp, 0.5 * (v - w - vp), 0.5 * (-v - w + vp), m)
        bt1 = get_buffer_t(w, v, vp, m)
        bu1 = get_buffer_u(v - vp, 0.5 * (v - w + vp), 0.5 * ( v + w + vp), m)
        
        # get buffers for local term
        bs2 = get_buffer_s( v + vp, 0.5 * (v - w - vp), 0.5 * (v + w - vp), m)
        bt2 = get_buffer_t(-v + vp, 0.5 * (v - w + vp), 0.5 * (v + w + vp), m)
        bu2 = get_buffer_u(-w, v, vp, m)

        # compute value
        buff[i] = -(2.0 * a.S)^2 * get_Γ_comp(9, site, bs1, bt1, bu1, r, a, apply_flags_su2_hkg) / (2.0 * pi)^2

        if site == 1
            vdx = get_Γ_comp(14, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vxd = get_Γ_comp(11, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vyz = get_Γ_comp(6, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            vzy = get_Γ_comp(9, site, bs2, bt2, bu2, r, a, apply_flags_su2_hkg)
            buff[i] -= (2.0 * a.S) * (vdx - vxd - vyz - vzy) / (2.0 * (2.0 * pi)^2)
        end

        buff[i] *= get_G(Λ,  v - 0.5 * w, m, a) * get_G(Λ,  v + 0.5 * w, m, a) * dv
        buff[i] *= get_G(Λ, vp - 0.5 * w, m, a) * get_G(Λ, vp + 0.5 * w, m, a) * dvp
    end
    
    return nothing
end

# compute correlations in real and Matsubara space
function compute_χ!(
    Λ     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action_su2_hkg,
    χ     :: Vector{Matrix{Float64}},
    χ_tol :: NTuple{2, Float64}
    )     :: Nothing

    @sync for w in 1 : m.num_χ
        for i in eachindex(r.sites)
            Threads.@spawn begin
                # determine reference scale 
                ref = Λ + 0.5 * m.χ[w]

                # compute xx vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_xx!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[1][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute yy vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_yy!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[2][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute zz vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_zz!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[3][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute xy vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_xy!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[4][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute xz vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_xz!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[5][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute yz vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_yz!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[6][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute yx vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_yx!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[7][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute zx vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_zx!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[8][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute zy vertex contribution
                integrand  = (vv, buff) -> compute_χ_kernel_zy!(Λ, i, m.χ[w], vv, buff, r, m, a)
                χ[9][i, w] = integrate_χ_boxes(integrand, 4.0 * ref, χ_tol)

                # compute propagator contribution
                if i == 1
                    integrand   = v -> (2.0 * a.S) * get_G(Λ, v - 0.5 * m.χ[w], m, a) * get_G(Λ, v + 0.5 * m.χ[w], m, a) / (4.0 * pi)
                    res         = quadgk(integrand, -Inf, -2.0 * ref, 0.0, 2.0 * ref, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]
                    χ[1][i, w] += res
                    χ[2][i, w] += res
                    χ[3][i, w] += res
                    
                end
            end
        end
    end

    return nothing
end