"""
    Action_su2_hkg <: Action

Struct containing self energy and vertex components for SU(2) HKG symmetric models.
* `S :: Float64`         : total spin quantum number 
* `Σ :: Vector{Float64}` : negative imaginary part of the self energy
* `Γ :: Vector{Vertex}`  : spin and density component of the full vertex
"""
struct Action_su2_hkg <: Action
    S :: Float64
    Σ :: Vector{Float64}
    Γ :: Vector{Vertex}
end 

# generate action_su2_hkg dummy
function get_action_su2_hkg_empty(
    S :: Float64,
    r :: Reduced_lattice,
    m :: Mesh,
    ) :: Action_su2

    # init self energy
    Σ = zeros(Float64, length(m.σ))

    # init vertices
    Γ = Vertex[get_vertex_empty(r, m) for i in 1 : 13]

    # build action
    a = Action_su2_hkg(S, Σ, Γ)

    return a
end

# init action for su2_hkg symmerty
function init_action!(
    l :: Lattice,
    r :: Reduced_lattice,
    a :: Action_su2
    ) :: Nothing

    # init bare action for spin component  Γxx, Γyy, Γzz, Γxy, Γxz, Γyz
    ref_int = SVector{4, Int64}(0, 0, 0, 1)
    ref     = Site(ref_int, get_vec(ref_int, l.uc))

    for i in eachindex(r.sites)
        # get bond from lattice
        b = get_bond(ref, r.sites[i], l)

        #set Γxx bare according to spin exchange
        

        #set Γyy bare according to spin exchange

        #set Γzz bare according to spin exchange

        #set Γxy bare according to spin exchange

        #set Γxz bare according to spin exchange

        #set Γyz bare according to spin exchange
       

    end

    return nothing
end