"""
    Bond 

Struct encapsulating the interactions between two lattice sites in matrix form.
* `sites    :: Tuple{Int64, Int64}` : indices of interacting lattice sites
* `exchange :: Matrix{Float64}`     : interaction matrix
"""
struct Bond 
    sites    :: Tuple{Int64, Int64}
    exchange :: Matrix{Float64}
end

# generate a bond with vanishing interactions
function get_bond_empty(
    i :: Int64, 
    j :: Int64
    ) :: Bond 

    b = Bond((i, j), zeros(Float64, 3, 3))

    return b 
end 

# modify bond matrix in place 
function add_bond!(
    J :: Float64,
    b :: Bond,
    μ :: Int64,
    ν :: Int64
    ) :: Nothing

    b.exchange[μ, ν] += J

    return nothing
end

# check if bonds are equal 
function are_equal(
    b1 :: Bond, 
    b2 :: Bond
    )  :: Bool 

    equal = norm(b1.exchange .- b2.exchange) <= 1e-8

    return equal 
end
