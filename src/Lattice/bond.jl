"""
    bond 

Struct encapsulating the interactions between two lattice sites in matrix form.
"""
struct bond 
    sites    :: Tuple{Int64, Int64}
    exchange :: Vector{Matrix{Float64}}
end

# generate a new, but empty bond 
function get_bond_empty(
    i :: Int64, 
    j :: Int64
    ) :: bond 

    b = bond((i, j), Matrix{Float64}[])

    return b 
end 

# load bonds
include("bond_lib/bond_heisenberg.jl")

# check if bonds are equal 
function are_equal(
    b1 :: bond, 
    b2 :: bond
    )  :: Bool 

    # init Bool
    equal = true

    # check if the bonds have the same interactions
    if length(b1.exchange) != length(b2.exchange)
        equal = false
    else
        for i in eachindex(b1.exchange)
            if maximum(abs.(b1.exchange[i] .- b2.exchange[i])) > 1e-10
                equal = false
                break
            end
        end
    end

    return equal
end