# integrate inplace function over an interval (a, b) using an adaptive trapezoidal rule (b > a)
function trapz!(
    f!    :: Function,
    buff1 :: Matrix{Float64},
    buff2 :: Matrix{Float64},
    a     :: Float64,
    b     :: Float64,
    atol  :: Float64,
    rtol  :: Float64,
    n_max :: Int64
    )     :: Nothing

    # reset buffer 
    buff1 .= 0.0

    # compute initial approximation
    f!(buff1, a, 0.5 * (b - a))
    f!(buff1, b, 0.5 * (b - a))

    # compute improved approximation 
    buff2 .= 0.5 .* buff1
    f!(buff2, 0.5 * (b + a), 0.5 * (b - a))

    # compute absolute and relative error
    norm1   = norm(buff1)
    norm2   = norm(buff2)
    buff1 .-= buff2
    adiff   = norm(buff1)
    rdiff   = adiff / min(norm1, norm2)
    buff1  .= buff2

    # set number of intervals
    n = 4

    # continue computing improved guesses, until result converges within given tolerances
    while adiff > atol && rdiff > rtol && n < n_max
        # compute improved approximation
        buff2 .= 0.5 .* buff1
        h      = (b - a) / n

        for i in 1 : n - 1
            if isodd(i)
                f!(buff2, a + i * h, h)
            end
        end

        # compute absolute and relative error
        norm1   = norm(buff1)
        norm2   = norm(buff2)
        buff1 .-= buff2
        adiff   = norm(buff1)
        rdiff   = adiff / min(norm1, norm2)
        buff1  .= buff2

        # double number of intervals
        n *= 2
    end

    return nothing
end 

# integrate inplace function over an interval (a, b) by pre-discretizing the integration domain and applying an adaptive trapezoidal rule to each subdomain (b > a)
# note: tbuff[1] is not reset for convenience in flow integration
function integrate!(
    f!    :: Function, 
    tbuff :: NTuple{3, Matrix{Float64}},
    a     :: Float64, 
    b     :: Float64,
    eval  :: Int64
    ;
    atol  :: Float64 = 1e-10, 
    rtol  :: Float64 = 1e-2,
    n_max :: Int64   = 10^5
    )     :: Nothing

    # split integration domain in subdomains of equal length 
    h = (b - a) / eval

    for i in 1 : eval 
        trapz!((b, x, dx) -> f!(b, x, dx), tbuff[2], tbuff[3], a + (i - 1) * h, a + i * h, atol, rtol, n_max)
        tbuff[1] .+= tbuff[2]
    end

    return nothing  
end