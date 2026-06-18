import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

SQUARE_FOUR_TO_FOUR_ESMDA_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_esmda_vs_adam")
SQUARE_FOUR_TO_FOUR_SINGLE_SCENARIO_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs")
include(joinpath(@__DIR__, "..", "..", "..", "common", "single_scenario", "common.jl"))
