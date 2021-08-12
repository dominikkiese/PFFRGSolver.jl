# integrate inplace function over an interval (a, b >= a) using the trapezoidal rule with (eval) linear subdivisions
# note: buff is not reset for convenience in flow integration
function integrate_lin!(
    f!    :: Function, 
    buff  :: Matrix{Float64},
    a     :: Float64, 
    b     :: Float64,
    eval  :: Int64
    )     :: Nothing

    @assert b >= a "Upper integration bound must be larger than or equal to lower bound."

    if b > a
        # determine linear step width
        h = (b - a) / eval

        # compute boundary terms 
        f!(buff, a, 0.5 * h)
        f!(buff, b, 0.5 * h) 

        # compute bulk terms 
        for i in 1 : eval - 1 
            f!(buff, a + i * h, h)
        end 
    end

    return nothing  
end

# integrate inplace function over an interval (a > 0, b >= a) using the trapezoidal rule with (eval) logarithmic subdivisions
# note: integral can be mapped onto negative domain (-b, -a) using the sgn keyword
# note: buff is not reset for convenience in flow integration
function integrate_log!(
    f!    :: Function, 
    buff  :: Matrix{Float64},
    a     :: Float64, 
    b     :: Float64,
    eval  :: Int64
    ;
    sgn   :: Float64 = 1.0
    )     :: Nothing

    @assert b >= a  "Upper integration bound must be larger than or equal to lower bound."
    @assert a > 0.0 "Lower bound must be larger than zero."

    if b > a
        # determine logarithmic factor
        ξ = (b / a)^(1.0 / eval)

        # compute boundary terms 
        f!(buff, sgn * a, 0.5 * (ξ - 1.0) * a)
        f!(buff, sgn * b, 0.5 * (ξ - 1.0) * a * ξ^(eval - 1)) 

        # compute intersection terms 
        for i in 1 : eval - 1 
            f!(buff, sgn * ξ^i * a, 0.5 * (ξ^2 - 1.0) * a * ξ^(i - 1))
        end 
    end

    return nothing  
end