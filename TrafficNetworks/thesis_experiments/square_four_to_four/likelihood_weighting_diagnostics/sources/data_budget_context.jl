import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

SQUARE_FOUR_TO_FOUR_ESMDA_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four", "square_four_to_four_esmda_vs_adam")
SQUARE_FOUR_TO_FOUR_SINGLE_SCENARIO_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_single_scenario")
SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_multi_scenario")
include(joinpath(@__DIR__, "..", "..", "data_amount_and_esmda_budget", "sources", "data_budget_experiment.jl"))
