# generate access buffer for s channel of Action_u1_dm struct
function get_buffer_s(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh_u1_dm
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

# generate access buffer for t channel of Action_u1_dm struct
function get_buffer_t(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh_u1_dm
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

# generate access buffer for u channel of Action_u1_dm struct
function get_buffer_u(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh_u1_dm
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





# interface function to generate s channel buffers for all components
function get_buffers_s(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: Mesh_u1_dm
    )  :: NTuple{6, Buffer}

    buffs = ntuple(comp -> get_buffer_s(comp, w, v, vp, m), 6)

    return buffs 
end

# interface function to generate t channel buffers for all components
function get_buffers_t(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: Mesh_u1_dm
    )  :: NTuple{6, Buffer}

    buffs = ntuple(comp -> get_buffer_t(comp, w, v, vp, m), 6)

    return buffs 
end

# interface function to generate u channel buffers for all components
function get_buffers_u(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: Mesh_u1_dm
    )  :: NTuple{6, Buffer}

    buffs = ntuple(comp -> get_buffer_u(comp, w, v, vp, m), 6)

    return buffs 
end