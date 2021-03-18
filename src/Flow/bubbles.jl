"""
    get_G_bare(
        Λ :: Float64,
        w :: Float64
        ) :: Float64 

Returns the bare propagator for smooth regulator R = (1 - exp(-w^2 / Λ^2)).
"""
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

"""
    get_G(
        Λ :: Float64,
        w :: Float64,
        m :: mesh,
        a :: action
        ) :: Float64

Returns the dressed propagator for smooth regulator R = (1 - exp(-w^2 / Λ^2)).
"""
function get_G(
    Λ :: Float64,
    w :: Float64,
    m :: mesh,
    a :: action
    ) :: Float64

    val = 0.0 

    if abs(w) > 1e-10
        Σ     = get_Σ(w, m, a)
        G0    = get_G_bare(Λ, w)
        denom = 0.0

        # ensure causality
        if w * Σ >= 0.0
            denom = 1.0 / G0 + Σ
        else
            denom = 1.0 / G0
        end 

        val = 1.0 / denom
    end

    return val 
end

"""
    get_S(
        Λ :: Float64,
        w :: Float64,
        m :: mesh,
        a :: action
        ) :: Float64

Returns the single-scale propagator for smooth regulator R = (1 - exp(-w^2 / Λ^2)).
"""
function get_S(
    Λ :: Float64,
    w :: Float64,
    m :: mesh,
    a :: action
    ) :: Float64

    val = 0.0 

    if abs(w) > 1e-10
        G   = get_G(Λ, w, m, a)
        G0  = get_G_bare(Λ, w)
        val = (G / G0)^2 * exp(-w^2 / Λ^2) * 2.0 * w / Λ^3 
    end 

    return val 
end

"""
    get_propagator_kat(
        Λ  :: Float64,
        w1 :: Float64,
        w2 :: Float64,
        m  :: mesh,
        a  :: action,
        da :: action
        )  :: Float64 

Returns the differentiated propagator bubble for smooth regulator R = (1 - exp(-w^2 / Λ^2)).
"""
function get_propagator_kat(
    Λ  :: Float64,
    w1 :: Float64,
    w2 :: Float64,
    m  :: mesh,
    a  :: action,
    da :: action
    )  :: Float64 

    dΣ  = get_Σ(w1, m, da)
    S1  = get_S(Λ, w1, m, a)
    G1  = get_G(Λ, w1, m, a)
    G2  = get_G(Λ, w2, m, a)
    val = (S1 + G1^2 * dΣ) * G2 / (2.0 * pi)

    return val 
end

"""
    get_propagator(
        Λ  :: Float64,
        w1 :: Float64,
        w2 :: Float64,
        m  :: mesh,
        a  :: action,
        da :: action
        )  :: Float64 

Returns the undifferentiated propagator bubble for smooth regulator R = (1 - exp(-w^2 / Λ^2)).
"""
function get_propagator(
    Λ  :: Float64,
    w1 :: Float64,
    w2 :: Float64,
    m  :: mesh,
    a  :: action
    )  :: Float64

    G1  = get_G(Λ, w1, m, a)
    G2  = get_G(Λ, w2, m, a)
    val = G1 * G2 / (2.0 * pi)

    return val
end