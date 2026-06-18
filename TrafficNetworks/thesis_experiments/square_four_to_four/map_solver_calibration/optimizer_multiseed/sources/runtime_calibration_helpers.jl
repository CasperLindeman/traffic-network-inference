import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

SQUARE_FOUR_TO_FOUR_ESMDA_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_esmda_vs_adam")
SQUARE_FOUR_TO_FOUR_FORWARDDIFF_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_esmda_vs_adam_forwarddiff")
SQUARE_FOUR_TO_FOUR_LONGRUN_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_longrun_comparison")
SQUARE_FOUR_TO_FOUR_BUDGET_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_budget_comparison")
SQUARE_FOUR_TO_FOUR_FOLLOWUP_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_followups")
SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFY_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_runtime_verification")
include(joinpath(@__DIR__, "..", "..", "..", "common", "map_calibration", "runtime_verification.jl"))
