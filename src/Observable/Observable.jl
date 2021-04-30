# load code
include("momentum.jl")
include("disk.jl")

# load correlations for different symmetries 
include("correlation_lib/correlation_sun.jl")

# load tests and timers
include("test.jl")
include("timers.jl")