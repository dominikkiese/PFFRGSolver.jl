# auxiliary function to compute necessary contributions for refined Simpson rule
function residual!(
    f!   :: Function,
    buff :: Matrix{Float64},
    a    :: Float64,
    b    :: Float64,
    n    :: Int64
    )    :: Nothing

    # compute refined step size
    h = 0.5 * (b - a) / n

    # compute integral contributions for new nodes
    for i in 1 : n
        f!(buff, a + (2.0 * i - 1.0) * h, +4.0 * h / 3.0)
    end 

    # compute integral contributions for reweighted nodes
    for i in 1 : n ÷ 2 
        f!(buff, a + (4.0 * i - 2.0) * h, -2.0 * h / 3.0)
    end 

    return nothing
end

# integrate inplace function over an interval [a, b], with b >= a, using an adaptive Simpson rule with Richardson extrapolation on the converged result
function simps!(
    f!    :: Function,
    buff1 :: Matrix{Float64},
    buff2 :: Matrix{Float64},
    a     :: Float64,
    b     :: Float64,
    atol  :: Float64,
    rtol  :: Float64
    )     :: Nothing

    # reset result buffer
    @turbo buff1 .= 0.0

    # compute initial approximation
    m = 0.5 * (a + b)
    f!(buff1, a, 1.0 * (b - a) / 6.0)
    f!(buff1, m, 2.0 * (b - a) / 3.0)
    f!(buff1, b, 1.0 * (b - a) / 6.0)

    # compute improved approximation
    @turbo buff2 .= 0.5 .* buff1
    residual!((b, x, dx) -> f!(b, x, dx), buff2, a, b, 2)
    n = 4

    # continue improving the integral approximation until convergence is reached
    while true
        # perform Richardson extrapolation
        @turbo buff1 .*= -1.0 / 15.0
        @turbo buff1 .+= 16.0 / 15.0 .* buff2

        # compute errors
        norm1 = norm(buff1)
        norm2 = norm(buff2)
        @turbo buff1 .-= buff2
        adiff = norm(buff1)
        rdiff = adiff / max(norm1, norm2)
        
        # if result has converged, terminate
        if adiff < atol || rdiff < rtol
            @turbo buff1 .+= buff2
            break
        end

        # update current solution
        @turbo buff1 .= buff2

        # compute improved approximation
        @turbo buff2 .= 0.5 .* buff1
        residual!((b, x, dx) -> f!(b, x, dx), buff2, a, b, n)
        n *= 2
    end 

    return nothing 
end

# integrate inplace function over an interval [a, b], with b >= a, by pre-discretizing the integration domain linearly and applying an adaptive Simpson rule to each subdomain
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

        # iterate over subdomains and apply adaptive Simpson rule
        for i in 1 : eval 
            simps!((b, x, dx) -> f!(b, x, dx), tbuff[2], tbuff[3], a + (i - 1) * h, a + i * h, atol, rtol)
            tbuff[1] .+= tbuff[2]
        end
    end

    return nothing  
end

# integrate inplace function over an interval [a, b], with b >= a and a > 0, by pre-discretizing the integration domain logarithmically and applying an adaptive Simpson rule to each subdomain
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

        # iterate over subdomains and apply adaptive Simpson rule
        for i in 1 : eval 
            simps!((b, x, dx) -> f!(b, sgn * x, dx), tbuff[2], tbuff[3], ξ^(i - 1) * a, ξ^i * a, atol, rtol)
            tbuff[1] .+= tbuff[2]
        end
    end

    return nothing  
end