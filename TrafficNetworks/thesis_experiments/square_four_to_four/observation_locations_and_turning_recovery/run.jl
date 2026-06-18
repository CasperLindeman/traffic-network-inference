import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "plots.jl"))

make_square_observation_locations_turning_recovery_figure()
