# define bare propagator
function get_G_bare(
    Λ :: Float64,
    w :: Float64
    ) :: Float64 

    val = 0.0

    if abs(w) > 1e-10
        val = -expm1(-w^2 / Λ^2) / w
    end

    return val
end

# define full propagator
function get_G(
    Λ :: Float64,
    w :: Float64,
    m :: mesh,
    a :: action
    ) :: Float64

    val = 0.0 

    if abs(w) > 1e-10
        denom = 1.0 / get_G_bare(Λ, w) + get_Σ(w, m, a)
        val   = 1.0 / denom
    end

    return val 
end

# define single scale propagator
function get_S(
    Λ :: Float64,
    w :: Float64,
    m :: mesh,
    a :: action
    ) :: Float64

    val = 0.0 

    if abs(w) > 1e-10
        val1 = get_G(Λ, w, m, a)^2 / get_G_bare(Λ, w)^2
        val2 = exp(-w^2 / Λ^2) * 2.0 * w / Λ^3 
        val  = val1 * val2 
    end 

    return val 
end

# define Katanin bubble
function get_propagator_kat(
    Λ  :: Float64,
    w1 :: Float64,
    w2 :: Float64,
    m  :: mesh,
    a  :: action,
    da :: action
    )  :: Float64 

    val1 = get_S(Λ, w1, m, a) + get_G(Λ, w1, m, a)^2 * get_Σ(w1, m, da)
    val2 = get_G(Λ, w2, m, a)
    val  = val1 * val2 / (2.0 * pi)

    return val 
end

# define propagator bubble
function get_propagator(
    Λ  :: Float64,
    w1 :: Float64,
    w2 :: Float64,
    m  :: mesh,
    a  :: action
    )  :: Float64

    val1 = get_G(Λ, w1, m, a)
    val2 = get_G(Λ, w2, m, a)
    val  = val1 * val2 / (2.0 * pi)

    return val
end