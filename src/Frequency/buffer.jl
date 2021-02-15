abstract type buffer end

# function to invert flag 
function set_flag(
    flag :: Bool 
    )    :: Bool

    if flag 
        flag = false 
    else
        flag = true 
    end 

    return flag 
end

# Heavyside function for kernel weights
function heavyside( 
    val :: Float64
    )   :: Float64 
    
    w = 0.0

    if val >= 0.0
        w = 1.0 
    end 

    return w
end

# load buffers for different symmetries 
include("buffer_lib/buffer_sun.jl")
