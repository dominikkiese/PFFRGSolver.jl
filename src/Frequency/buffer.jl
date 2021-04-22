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

# load buffers for different symmetries
include("buffer_lib/buffer_sun.jl")