import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

previous_skip = get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT_REPLOT", nothing)
ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT_REPLOT"] = "1"

include(joinpath(@__DIR__, "sources", "plot_data_budget_figures.jl"))

if previous_skip === nothing
    delete!(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT_REPLOT")
else
    ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT_REPLOT"] = previous_skip
end

function make_square_data_amount_and_esmda_budget_figures()
    return run_square_data_budget_experiment_replot()
end
