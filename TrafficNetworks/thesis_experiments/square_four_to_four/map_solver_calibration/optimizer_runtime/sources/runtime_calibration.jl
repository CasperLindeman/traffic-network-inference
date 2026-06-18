import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

SQUARE_FOUR_TO_FOUR_ESMDA_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_esmda_vs_adam")
SQUARE_FOUR_TO_FOUR_FORWARDDIFF_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_esmda_vs_adam_forwarddiff")
SQUARE_FOUR_TO_FOUR_LONGRUN_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_longrun_comparison")
SQUARE_FOUR_TO_FOUR_BUDGET_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_budget_comparison")
SQUARE_FOUR_TO_FOUR_FOLLOWUP_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "square_four_to_four_followups")
SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFY_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs")
ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFICATION_DEMO"] = "1"
include(joinpath(@__DIR__, "..", "..", "..", "common", "map_calibration", "runtime_verification.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    run_optimizer_runtime_verification(; output_dir=SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFY_OUTPUT_DIR)
end
