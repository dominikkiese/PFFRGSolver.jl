# load modules
using LinearAlgebra 
using HDF5
using QuadGK
using LoopVectorization
using Dates
using Test
using TimerOutputs

# load source code
include(joinpath(@__DIR__, "src/Lattice/Lattice.jl"))
include(joinpath(@__DIR__, "src/Frequency/Frequency.jl"))
include(joinpath(@__DIR__, "src/Action/Action.jl"))
include(joinpath(@__DIR__, "src/Flow/Flow.jl"))
include(joinpath(@__DIR__, "src/Observable/Observable.jl"))
include(joinpath(@__DIR__, "src/Launcher/Launcher.jl"))