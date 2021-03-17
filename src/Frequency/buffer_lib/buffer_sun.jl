"""
    buffer_sun <: buffer 

Access buffer for reading out vertices from action_sun struct. 
Contains symmetry related flags, (asymptotic) kernel specification and interpolation parameters.
"""
struct buffer_sun <: buffer  
    exchange_flag :: Bool 
    map_flag      :: Bool
    kernel        :: Int64
    p1            :: param 
    p2            :: param 
    p3            :: param 
end

"""
    get_buffer_sun_empty() :: buffer_sun

Generate buffer_sun struct with dummy fields.
"""
function get_buffer_sun_empty() :: buffer_sun 

    b = buffer_sun(false, false, 0, get_param_empty(), get_param_empty(), get_param_empty())

    return b 
end 





"""
    get_buffer_sun_s(
        w  :: Float64,
        v  :: Float64,
        vp :: Float64,
        m  :: mesh
        )  :: buffer_sun

Generate access buffer for s channel of action_sun struct.
Symmetries are applied to map all frequencies onto non-negative values:
1) -w  -> w  + site exchange 
2) -v  -> v  + site exchange + mapping to u channel (sign if density)
3) -vp -> vp + mapping to u channel (sign if density)
"""
function get_buffer_sun_s(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    b             = get_buffer_sun_empty()
    exchange_flag = false
    map_flag      = false

    # do -w -> w + site exchange 
    if w < 0.0 
        w             *= -1.0 
        exchange_flag  = set_flag(exchange_flag)
    end 

    # do -v -> v + site exchange + mapping to u channel (sign if density)
    if v < 0.0 
        v             *= -1.0 
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag) 
    end

    # do -vp -> vp + mapping to u channel (sign if density)
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag) 
    end

    # interpolation for q3
    if heavyside(m.Ωs[end] - abs(w)) * heavyside(m.νs[end] - abs(v)) * heavyside(m.νs[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, m.Ωs), get_param(v, m.νs), get_param(vp, m.νs))
        @goto exit
    end
    
    # interpolation for q2_2
    if heavyside(m.Ωs[end] - abs(w)) * heavyside(m.νs[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, m.Ωs), get_param_empty(), get_param(vp, m.νs)) 
        @goto exit
    end
    
    # interpolation for q2_1
    if heavyside(m.Ωs[end] - abs(w)) * heavyside(m.νs[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, m.Ωs), get_param(v, m.νs), get_param_empty()) 
        @goto exit
    end
    
    # interpolation for q1
    if heavyside(m.Ωs[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, m.Ωs), get_param_empty(), get_param_empty()) 
        @goto exit
    end

    @label exit

    return b
end

"""
    get_buffer_sun_t(
        w  :: Float64,
        v  :: Float64,
        vp :: Float64,
        m  :: mesh
        )  :: buffer_sun

Generate access buffer for t channel of action_sun struct.
Symmetries are applied to map all frequencies onto non-negative values:
1) -w  -> w  
2) -v  -> v  (sign if density)
3) -vp -> vp (sign if density)
"""
function get_buffer_sun_t(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    b             = get_buffer_sun_empty()
    exchange_flag = false
    map_flag      = false

    # do -w -> w
    if w < 0.0
        w *= -1.0
    end
    
    # do -v -> v + sign in density
    if v < 0.0
        v        *= -1.0
        map_flag  = set_flag(map_flag)
    end
    
    # do -vp -> vp + sign in density
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag)
    end

    # interpolation for q3
    if heavyside(m.Ωt[end] - abs(w)) * heavyside(m.νt[end] - abs(v)) * heavyside(m.νt[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, m.Ωt), get_param(v, m.νt), get_param(vp, m.νt))
        @goto exit
    end
    
    # interpolation for q2_2
    if heavyside(m.Ωt[end] - abs(w)) * heavyside(m.νt[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, m.Ωt), get_param_empty(), get_param(vp, m.νt)) 
        @goto exit
    end
    
    # interpolation for q2_1
    if heavyside(m.Ωt[end] - abs(w)) * heavyside(m.νt[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, m.Ωt), get_param(v, m.νt), get_param_empty()) 
        @goto exit
    end
    
    # interpolation for q1
    if heavyside(m.Ωt[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, m.Ωt), get_param_empty(), get_param_empty()) 
        @goto exit
    end

    @label exit

    return b
end

"""
    get_buffer_sun_u(
        w  :: Float64,
        v  :: Float64,
        vp :: Float64,
        m  :: mesh
        )  :: buffer_sun

Generate access buffer for u channel of action_sun struct.
Symmetries are applied to map all frequencies onto non-negative values:
1) -w  -> w  + site exchange 
2) -v  -> v  + site exchange + mapping to s channel (sign if density)
3) -vp -> vp + mapping to s channel (sign if density)
"""
function get_buffer_sun_u(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    b             = get_buffer_sun_empty()
    exchange_flag = false
    map_flag      = false

    # do -w -> w + site exchange
    if w < 0.0
        w             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
    end

    # do -v -> v + site exchange + mapping to s channel (sign if density)
    if v < 0.0
        v             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag)
    end

    # do -vp -> vp + mapping to s channel (sign if density)
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag) 
    end

    # interpolation for q3
    if heavyside(m.Ωs[end] - abs(w)) * heavyside(m.νs[end] - abs(v)) * heavyside(m.νs[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, m.Ωs), get_param(v, m.νs), get_param(vp, m.νs))
        @goto exit
    end
    
    # interpolation for q2_2
    if heavyside(m.Ωs[end] - abs(w)) * heavyside(m.νs[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, m.Ωs), get_param_empty(), get_param(vp, m.νs)) 
        @goto exit
    end
    
    # interpolation for q2_1
    if heavyside(m.Ωs[end] - abs(w)) * heavyside(m.νs[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, m.Ωs), get_param(v, m.νs), get_param_empty()) 
        @goto exit
    end
    
    # interpolation for q1
    if heavyside(m.Ωs[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, m.Ωs), get_param_empty(), get_param_empty()) 
        @goto exit
    end
    
    @label exit

    return b
end





