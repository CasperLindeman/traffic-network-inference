import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

previous_skip = get(ENV, "TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO", nothing)
ENV["TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO"] = "1"

include(joinpath(@__DIR__, "inference.jl"))
include(joinpath(@__DIR__, "plots.jl"))

if previous_skip === nothing
    delete!(ENV, "TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO")
else
    ENV["TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO"] = previous_skip
end

results = run_two_to_two_inference()
print_summary(results)
make_two_to_two_posterior_figure(results)
