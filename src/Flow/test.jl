# define test kernel
function test_kernel!(
    b  :: Matrix{Float64},
    x  :: Float64,
    dx :: Float64
    )  :: Nothing

    for i in eachindex(b)
        b[i] += dx * x^i * exp(-x^2)
    end

    return nothing
end

# define benchmark kernel
function bench_kernel!(
    b :: Matrix{Float64},
    x :: Float64
    ) :: Nothing

    for i in eachindex(b)
        b[i] = x^i * exp(-x^2)
    end

    return nothing
end

"""
    test_flow() :: Nothing

Run consistency checks for flow equations by testing integrators.
"""
function test_flow() :: Nothing

    # benchmark handcrafted integrators against QuadGK
    test_integrators()

    return nothing
end

"""
    test_integrators() :: Nothing

Run consistency checks for integrators by computing test integrals and comparing to QuadGK.
"""
function test_integrators() :: Nothing

    # init test dummys
    b1 = zeros(Float64, 10, 10)
    b2 = (copy(b1), copy(b1), copy(b1))
    b3 = (copy(b1), copy(b1), copy(b1))

    @testset "quadrature" begin
        # compute integral with QuadGK
        quadgk!((b, x) -> bench_kernel!(b, x), b1, 1.0, 5.0, atol = 1e-8, rtol = 1e-8, maxevals = 10^6)

        # compute integral with trapz!
        integrate_lin!((b, x, dx) -> test_kernel!(b, x, dx), b2, 1.0, 5.0, 50, 1e-8, 1e-8, n_max = 10^6)
        integrate_log!((b, x, dx) -> test_kernel!(b, x, dx), b3, 1.0, 5.0, 50, 1e-8, 1e-8, n_max = 10^6)

        @test b1 ≈ b2[1]
        @test b1 ≈ b3[1]
    end

    return nothing
end