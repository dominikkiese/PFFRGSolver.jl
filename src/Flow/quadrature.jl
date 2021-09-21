# integrate inplace function over an interval [a, b], with b >= a, using an adaptive trapezoidal rule with Richardson extrapolation on the converged result
function trapz!(
    f!    :: Function,
    buff1 :: Matrix{Float64},
    buff2 :: Matrix{Float64},
    a     :: Float64,
    b     :: Float64,
    atol  :: Float64,
    rtol  :: Float64
    )     :: Nothing

    # reset buffer 
    @turbo buff1 .= 0.0

    # compute initial approximation
    f!(buff1, a, 0.5 * (b - a))
    f!(buff1, b, 0.5 * (b - a))

    # compute improved approximation 
    @turbo buff2 .= 0.5 .* buff1
    f!(buff2, 0.5 * (b + a), 0.5 * (b - a))

    # set number of intervals
    n = 4

    # continue computing improved approximations, until result converges within given tolerances or maximum number of subdivisions is reached
    while true
        # compute absolute and relative error
        norm1 = norm(buff1)
        norm2 = norm(buff2)
        @turbo buff1 .-= buff2
        adiff = norm(buff1)
        rdiff = adiff / min(norm1, norm2)

        if adiff < atol || rdiff < rtol || n > 1024
            # perform Richardson extrapolation for final result
            @turbo buff1 .+= buff2
            @turbo buff1 .*= -1.0 / 3.0
            @turbo buff1 .+=  4.0 / 3.0 .* buff2
            break
        end

        # initialize with current best guess
        @turbo buff1 .= buff2

        # compute improved approximation
        @turbo buff2 .= 0.5 .* buff1
        h = (b - a) / n

        for i in 1 : n - 1
            if isodd(i)
                f!(buff2, a + i * h, h)
            end
        end

        # double number of subdivisions
        n *= 2
    end

    return nothing
end 

# integrate inplace function over an interval [a, b], with b >= a, by pre-discretizing the integration domain linearly and applying an adaptive trapezoidal rule to each subdomain
# note: tbuff[1] is not reset for convenience in flow integration
function integrate_lin!(
    f!    :: Function, 
    tbuff :: NTuple{3, Matrix{Float64}},
    a     :: Float64, 
    b     :: Float64,
    eval  :: Int64,
    atol  :: Float64, 
    rtol  :: Float64
    )     :: Nothing

    @assert b >= a "Upper integration bound must be larger than or equal to lower bound."

    if b > a
        # split integration domain in subdomains of equal length 
        h = (b - a) / eval

        # iterate over subdomains and apply adaptive trapezoidal rule
        for i in 1 : eval 
            trapz!((b, x, dx) -> f!(b, x, dx), tbuff[2], tbuff[3], a + (i - 1) * h, a + i * h, atol, rtol)
            tbuff[1] .+= tbuff[2]
        end
    end

    return nothing  
end

# integrate inplace function over an interval [a, b], with b >= a and a > 0, by pre-discretizing the integration domain logarithmically and applying an adaptive trapezoidal rule to each subdomain
# note: tbuff[1] is not reset for convenience in flow integration
# note: the sgn keyword can be usend to map [a, b] -> [-b, -a]
function integrate_log!(
    f!    :: Function, 
    tbuff :: NTuple{3, Matrix{Float64}},
    a     :: Float64, 
    b     :: Float64,
    eval  :: Int64,
    atol  :: Float64, 
    rtol  :: Float64
    ;
    sgn   :: Float64 = 1.0
    )     :: Nothing

    @assert b >= a  "Upper integration bound must be larger than or equal to lower bound."
    @assert a > 0.0 "Lower bound must be larger zero."

    if b > a
        # determine logarithmic factor
        ξ = (b / a)^(1.0 / eval)

        # iterate over subdomains and apply adaptive trapezoidal rule
        for i in 1 : eval 
            trapz!((b, x, dx) -> f!(b, sgn * x, dx), tbuff[2], tbuff[3], ξ^(i - 1) * a, ξ^i * a, atol, rtol)
            tbuff[1] .+= tbuff[2]
        end
    end

    return nothing  
end