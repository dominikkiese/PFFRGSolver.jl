
# struct to encode a combined real and spin space symmetry
struct Mapping
    site :: Int64
    components :: Vector{Int64}
    signs :: Vector{Float64}
end

# functions to translate a spin transformation (permutation + sign changes on S = [Sx, Sy, Sz]) 
# into a mapping that directly works on the vertex components
# Note: Spin transformations apply signs first and then permute. 
#       Mappings map components first and then apply signs.

# heisenberg -- no spin symmetries needed
function trafoToMapping_su2(site, p_trafo, s_trafo)
    return Mapping(site, p_trafo, s_trafo)
end

# u1-dm -- no spin symmetries needed
function trafoToMapping_u1_dm(site, p_trafo, s_trafo)
    return Mapping(site, p_trafo, s_trafo)
end

# offdiagonal interactions -- spin symmetries needed
function trafoToMapping_su2_hkg(site, p_trafo, s_trafo)    
    p = [p_trafo; 4]
    s = [s_trafo; 1]
    
    Γxx = 1
    Γyy = 2
    Γzz = 3
    Γxy = 4
    Γxz = 5
    Γyz = 6
    Γyx = 7
    Γzx = 8
    Γzy = 9
    Γdd = 10
    Γxd = 11
    Γyd = 12
    Γzd = 13
    Γdx = 14
    Γdy = 15
    Γdz = 16
    
    Γmatrix = [
        Γxx     Γxy     Γxz     Γxd
        Γyx     Γyy     Γyz     Γyd
        Γzx     Γzy     Γzz     Γzd     
        Γdx     Γdy     Γdz     Γdd  
    ]
    
    Γmatrix_trafo = apply_spin_trafo(Γmatrix, p, s)
    
    p_map = zeros(Int64, 16)
    s_map = ones(Int64, 16)
    
    for i in eachindex(p_map)
        #index in Γ vector
        index = Γmatrix[i]

        #index in after transformation
        Γ_transformed = Γmatrix_trafo[i]

        #save in vector
        p_map[index] = abs(Γ_transformed)
        s_map[index] = sign(Γ_transformed)
    end
    
    return Mapping(site, p_map, s_map)
end

# interface function
function trafoToMapping(site, p_trafo, s_trafo, symmetry)
    if symmetry == "su2"
        return trafoToMapping_su2(site, p_trafo, s_trafo)
    elseif symmetry == "u1-dm"
        return trafoToMapping_u1_dm(site, p_trafo, s_trafo) 
    elseif symmetry == "su2-hkg"
        return trafoToMapping_su2_hkg(site, p_trafo, s_trafo)
    end
end

# define functions necessary for the unique() function to work on mappings
function Base.:(==)(a :: Mapping, b :: Mapping) 
    return (a.site == b.site) && (a.components == b.components) && (a.signs == b.signs)
end

function Base.hash(a :: Mapping, h :: UInt)
    return hash([a.site; a.components; a.signs], h)
end