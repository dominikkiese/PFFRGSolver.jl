""" 
    Param 

Struct containing interpolation parameters for a single point in a mesh.
* `lower_index  :: Int64`   : index of nearest neighbor in mesh with smaller value
* `upper_index  :: Int64`   : index of nearest neighbor in mesh with larger value
* `lower_weight :: Float64` : interpolation weight for mesh[lower_index]
* `upper_weight :: Float64` : interpolation weight for mesh[upper_index]
"""
struct Param 
    lower_index  :: Int64 
    upper_index  :: Int64  
    lower_weight :: Float64 
    upper_weight :: Float64 
end

# generate param dummy
function get_param_empty() :: Param 
    
    p = Param(0, 0, 0.0, 0.0)

    return p 
end

# find nearest neighbor (lower and upper) indices in sorted list including zero
function get_indices(
    val  :: Float64, 
    list :: Vector{Float64}
    )    :: NTuple{2, Int64}

    # init indices
    lower_index = 0 
    upper_index = 0

    # check if in bounds otherwise search list
    if val >= list[end]
        lower_index, upper_index = length(list), length(list)
    else
        # iterate over list until upper_index is found (lower_index is then also determined)
        index_current = 1

        while val > list[index_current]
            index_current += 1
        end 

        lower_index = index_current 
        upper_index = index_current 

        if val < list[index_current]
            lower_index -= 1 
        end 
    end

    return lower_index, upper_index 
end

"""
    get_param(
        val  :: Float64, 
        list :: Vector{Float64}
        )    :: Param
        
Compute interpolation parameters of val in a set of discrete points (list) and buffer result in Param struct.
"""
function get_param(
    val  :: Float64, 
    list :: Vector{Float64}
    )    :: Param 

    # get neighbors and init weights 
    lower_index,  upper_index  = get_indices(val, list)
    lower_weight, upper_weight = 0.0, 0.0

    # compute weights
    if lower_index < upper_index
        d            = 1.0 / (list[upper_index] - list[lower_index])
        lower_weight = d * (list[upper_index] - val)
        upper_weight = d * (val - list[lower_index])
    else
        lower_weight = 1.0 
    end

    # build param
    p = Param(lower_index, upper_index, lower_weight, upper_weight)

    return p 
end