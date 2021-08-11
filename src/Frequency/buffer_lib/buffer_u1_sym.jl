# generate access buffer for s channel of Action_u1_sym struct
function get_u1_sym_buffer_s(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh
    )    :: Buffer

    # generate symmetry related flags and perform mappings
    w, v, vp, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν = get_flags_s(w, v, vp)

    # respect exchange flag for interpolation
    if exchange_flag
        if comp == 5 
            comp = 6 
        elseif comp == 6
            comp = 5 
        end 
    end

    # deref meshes for interpolation, respecting possible mapping to u channel
    Ω = m.Ωs[comp]
    ν = m.νs[comp]

    if map_flag
        Ω = m.Ωu[comp]
        ν = m.νu[comp]
    end

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν)
end

# generate access buffer for t channel of Action_u1_sym struct
function get_u1_sym_buffer_t(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh
    )    :: Buffer

    # generate symmetry related flags and perform mappings
    w, v, vp, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν = get_flags_t(w, v, vp)

    # respect exchange flag for interpolation
    if exchange_flag
        if comp == 5 
            comp = 6 
        elseif comp == 6
            comp = 5 
        end 
    end

    # deref meshes for interpolation
    Ω = m.Ωt[comp]
    ν = m.νt[comp]

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν)
end

# generate access buffer for u channel of Action_u1_sym struct
function get_u1_sym_buffer_u(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh
    )    :: Buffer

    # generate symmetry related flags and perform mappings
    w, v, vp, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν = get_flags_u(w, v, vp)

    # respect exchange flag for interpolation
    if exchange_flag
        if comp == 5 
            comp = 6 
        elseif comp == 6
            comp = 5 
        end 
    end

    # deref meshes for interpolation, respecting possible mapping to s channel
    Ω = m.Ωu[comp]
    ν = m.νu[comp]

    if map_flag
        Ω = m.Ωs[comp]
        ν = m.νs[comp]
    end

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν)
end