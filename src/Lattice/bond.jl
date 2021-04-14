"""
    bond 

Struct encapsulating the interactions between two lattice sites in matrix form.
* `sites    :: Tuple{Int64, Int64}` : indices of interacting lattice sites
* `exchange :: Matrix{Float64}`     : interaction matrix
"""
struct bond 
    sites    :: Tuple{Int64, Int64}
    exchange :: Matrix{Float64}
end

# generate a bond with vanishing interactions
function get_bond_empty(
    i :: Int64, 
    j :: Int64
    ) :: bond 

    b = bond((i, j), zeros(Float64, 3, 3))

    return b 
end 

# load bonds
include("bond_lib/bond_heisenberg.jl")

# check if bonds are equal 
function are_equal(
    b1 :: bond, 
    b2 :: bond
    )  :: Bool 

    equal = norm(b1.exchange .- b2.exchange) <= 1e-10

    return equal 
end