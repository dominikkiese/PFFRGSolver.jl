# integrate inplace function over an interval [a, b] with b > a, using an adaptive trapezoidal rule with Richardson extrapolation on the converged result
function trapz!(
    f!    :: Function,
    buff1 :: Matrix{Float64},
    buff2 :: Matrix{Float64},
    a     :: Float64,
    b     :: Float64,
    eval  :: Int64,
    atol  :: Float64,
    rtol  :: Float64,
    n_max :: Int64
    )     :: Nothing

    # reset buffer 
    @turbo buff1 .= 0.0

    # determine initial step width
    h = (b - a) / eval

    # compute initial approximation
    f!(buff1, a, 0.5 * h)
    f!(buff1, b, 0.5 * h)

    for i in 1 : eval - 1 
        f!(buff1, a + i * h, h)
    end 

    # perform biscetion
    h    *= 0.5 
    eval *= 2

    # compute improved approximation 
    @turbo buff2 .= 0.5 .* buff1
    
    for i in 1 : eval - 1
        if isodd(i)
            f!(buff2, a + i * h, h)
        end
    end

    # continue computing improved approximations, until result converges within given tolerances or maximum number of subdivisions is reached
    while true 
        # compute absolute and relative error
        norm1 = norm(buff1)
        norm2 = norm(buff2)
        @turbo buff1 .-= buff2
        adiff = norm(buff1)
        rdiff = adiff / min(norm1, norm2)

        if adiff <= atol || rdiff <= rtol || eval >= n_max
            # perform Richardson extrapolation for final result
            @turbo buff1 .+= buff2
            @turbo buff1 .*= -1.0 / 3.0
            @turbo buff1 .+=  4.0 / 3.0 .* buff2
            break
        end

        # initialize with current best guess
        @turbo buff1 .= buff2

        # perform biscetion
        h    *= 0.5 
        eval *= 2

        # compute improved approximation
        @turbo buff2 .= 0.5 .* buff1

        for i in 1 : eval - 1
            if isodd(i)
                f!(buff2, a + i * h, h)
            end
        end
    end

    return nothing
end 