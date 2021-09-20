"""
    Mesh_su2 <: Mesh

Struct containing frequency meshes for the self energy and vertices of Action_su2.
* `num_σ :: Int64`                       : total number of frequencies in the self energy mesh
* `num_Ω :: Int64`                       : total number of frequencies in the bosonic meshes
* `num_ν :: Int64`                       : total number of frequencies in the fermionic meshes 
* `σ     :: Vector{Float64}`             : self energy mesh
* `Ωs    :: SVector{2, Vector{Float64}}` : bosonic meshes for the s channel
* `νs    :: SVector{2, Vector{Float64}}` : fermionic meshes for the s channel
* `Ωt    :: SVector{2, Vector{Float64}}` : bosonic meshes for the t channel 
* `νt    :: SVector{2, Vector{Float64}}` : fermionic meshes for the t channel
* `Ωu    :: SVector{2, Vector{Float64}}` : bosonic meshes for the u channel 
* `νu    :: SVector{2, Vector{Float64}}` : fermionic meshes for the u channel
"""
struct Mesh_su2 <: Mesh
    num_σ :: Int64 
    num_Ω :: Int64 
    num_ν :: Int64
    σ     :: Vector{Float64}
    Ωs    :: SVector{2, Vector{Float64}}
    νs    :: SVector{2, Vector{Float64}}
    Ωt    :: SVector{2, Vector{Float64}}
    νt    :: SVector{2, Vector{Float64}}
    Ωu    :: SVector{2, Vector{Float64}}
    νu    :: SVector{2, Vector{Float64}}
end

# generate a Mesh_su2 struct at given initial scale and distribution parameters
function get_mesh_su2(
    initial :: Float64,
    num_σ   :: Int64,
    num_Ω   :: Int64,
    num_ν   :: Int64,
    p_σ     :: Float64,
    p_Ω     :: Float64,
    p_ν     :: Float64
    )       :: Mesh_su2 

    m = Mesh_su2(num_σ + 1, 
                 num_Ω + 1, 
                 num_ν + 1, 
                 get_mesh(5.0 * initial, 500.0 * max(initial, 0.5), num_σ, p_σ), 
                 SVector(ntuple(comp -> get_mesh(5.0 * initial, 250.0 * max(initial, 0.5), num_Ω, p_Ω), 2)), 
                 SVector(ntuple(comp -> get_mesh(5.0 * initial, 150.0 * max(initial, 0.5), num_ν, p_ν), 2)), 
                 SVector(ntuple(comp -> get_mesh(5.0 * initial, 250.0 * max(initial, 0.5), num_Ω, p_Ω), 2)), 
                 SVector(ntuple(comp -> get_mesh(5.0 * initial, 150.0 * max(initial, 0.5), num_ν, p_ν), 2)),
                 SVector(ntuple(comp -> get_mesh(5.0 * initial, 250.0 * max(initial, 0.5), num_Ω, p_Ω), 2)), 
                 SVector(ntuple(comp -> get_mesh(5.0 * initial, 150.0 * max(initial, 0.5), num_ν, p_ν), 2)))

    return m 
end