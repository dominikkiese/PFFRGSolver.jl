# integrate inplace function over an interval (a, b >= a) using Simpson's rule with (eval) linear subdivisions
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
        # split integration domain in subdomains of equal length 
        h = (b - a) / eval

        # compute boundary terms 
        f!(buff, a, h / 6.0)
        f!(buff, b, h / 6.0) 

        # compute intersection terms 
        for i in 1 : eval - 1 
            f!(buff, a + i * h, h / 3.0)
        end 

        # compute center terms 
        for i in 0 : eval - 1
            f!(buff, a + i * h + 0.5 * h, 2.0 * h / 3.0)
        end
    end

    return nothing  
end

# integrate inplace function over an interval (a > 0, b >= a) using Simpson's rule with (eval) logarithmic subdivisions
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
        f!(buff, sgn * a, (ξ - 1.0) * a / 6.0)
        f!(buff, sgn * b, (b - ξ^(eval - 1) * a) / 6.0) 

        # compute intersection terms 
        for i in 1 : eval - 1 
            f!(buff, sgn * ξ^i * a, (ξ^2 - 1.0) * ξ^(i - 1) * a / 6.0)
        end 

        # compute center terms 
        for i in 0 : eval - 1
            f!(buff, sgn * 0.5 * (ξ + 1.0) * ξ^i * a, 2.0 * (ξ - 1.0) * ξ^i * a / 3.0)
        end
    end

    return nothing  
end