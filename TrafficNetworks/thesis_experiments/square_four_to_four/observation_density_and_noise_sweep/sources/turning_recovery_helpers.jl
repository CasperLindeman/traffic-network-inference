import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

SQUARE_FOUR_TO_FOUR_ESMDA_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four", "square_four_to_four_esmda_vs_adam")
SQUARE_FOUR_TO_FOUR_SINGLE_SCENARIO_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_single_scenario")
SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_multi_scenario")
SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "turning_outlier_structure")
include(joinpath(@__DIR__, "..", "..", "common", "turning_recovery", "structure.jl"))
