# define buffer_sun struct 
struct buffer_sun <: buffer  
    exchange_flag :: Bool 
    map_flag      :: Bool
    kernel        :: Int64
    p1            :: param 
    p2            :: param 
    p3            :: param 
end

# generate empty buffer_sun struct 
function get_buffer_sun_empty() :: buffer_sun 

    b = buffer_sun(false, false, 0, get_param_empty(), get_param_empty(), get_param_empty())

    return b 
end 





# buffer for s channel
function get_buffer_sun_s(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    # init flags
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

    # generate an empty buffer 
    b = get_buffer_sun_empty()

    # interpolation for q3
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(v)) * heavyside(m.ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, m.Ω), get_param(v, m.ν), get_param(vp, m.ν))
        @goto exit
    end
    
    # interpolation for q2_2
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, m.Ω), get_param_empty(), get_param(vp, m.ν)) 
        @goto exit
    end
    
    # interpolation for q2_1
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, m.Ω), get_param(v, m.ν), get_param_empty()) 
        @goto exit
    end
    
    # interpolation for q1
    if heavyside(m.Ω[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, m.Ω), get_param_empty(), get_param_empty()) 
        @goto exit
    end

    @label exit

    return b
end

# buffer for t channel
function get_buffer_sun_t(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    # init flags
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

    # generate an empty buffer 
    b = get_buffer_sun_empty()

    # interpolation for q3
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(v)) * heavyside(m.ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, m.Ω), get_param(v, m.ν), get_param(vp, m.ν))
        @goto exit
    end
    
    # interpolation for q2_2
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, m.Ω), get_param_empty(), get_param(vp, m.ν)) 
        @goto exit
    end
    
    # interpolation for q2_1
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, m.Ω), get_param(v, m.ν), get_param_empty()) 
        @goto exit
    end
    
    # interpolation for q1
    if heavyside(m.Ω[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, m.Ω), get_param_empty(), get_param_empty()) 
        @goto exit
    end

    @label exit

    return b
end

# buffer for u channel
function get_buffer_sun_u(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    # init flags
    exchange_flag = false
    map_flag      = false

    # do -w -> w + site exchange
    if w < 0.0
        w             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
    end

    # do -v -> v + site exchange + mapping to s-channel (sign if density)
    if v < 0.0
        v             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag)
    end

    # do -vp -> vp + mapping to s-channel (sign if density)
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag) 
    end

    # generate an empty buffer 
    b = get_buffer_sun_empty()

    # interpolation for q3
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(v)) * heavyside(m.ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, m.Ω), get_param(v, m.ν), get_param(vp, m.ν))
        @goto exit
    end
    
    # interpolation for q2_2
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, m.Ω), get_param_empty(), get_param(vp, m.ν)) 
        @goto exit
    end
    
    # interpolation for q2_1
    if heavyside(m.Ω[end] - abs(w)) * heavyside(m.ν[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, m.Ω), get_param(v, m.ν), get_param_empty()) 
        @goto exit
    end
    
    # interpolation for q1
    if heavyside(m.Ω[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, m.Ω), get_param_empty(), get_param_empty()) 
        @goto exit
    end
    
    @label exit

    return b
end





