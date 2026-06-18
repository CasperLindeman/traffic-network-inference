import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

previous_skip = get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_REPRESENTATIVE", nothing)
ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_REPRESENTATIVE"] = "1"

include(joinpath(@__DIR__, "sources", "reconstruction_figures.jl"))

if previous_skip === nothing
    delete!(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_REPRESENTATIVE")
else
    ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_REPRESENTATIVE"] = previous_skip
end
