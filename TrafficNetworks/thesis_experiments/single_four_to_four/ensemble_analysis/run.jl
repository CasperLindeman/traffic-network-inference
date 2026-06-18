import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

# Compatibility entrypoint; implementation lives in ../common/four_to_four_method_core.jl.
include(joinpath(@__DIR__, "..", "common", "four_to_four_method_core.jl"))

if get(ENV, "TRAFFICNETWORKS_SKIP_FOUR_TO_FOUR_DEMO", "0") != "1"
    all_results, best_result, nuts_cost = run_method_comparison()
end
