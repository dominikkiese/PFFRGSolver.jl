# load propagator bubbles
include("bubbles.jl")

# load parquet equations for different symmetries
include("parquet_lib/parquet_su2/parquet_su2.jl")

# load flow equations for different symmetries 
include("flow_lib/flow_su2/flow_su2.jl")

# load quadrature rule for vertex integration 
include("quadrature.jl")

# load BSE computation of channels for frequency tuples (w1, w2, w3) and kernels k 
include("BSE_s.jl")
include("BSE_t.jl")
include("BSE_u.jl")

# load flow computation of channels for frequency tuples (w1, w2, w3) and kernels k 
include("flow_s.jl")
include("flow_t.jl")
include("flow_u.jl")

# load full BSE and flow computation
include("compute.jl")