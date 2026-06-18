if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using Statistics
using LinearAlgebra
using ForwardDiff
using Plots
using Distributions
using SimulationBasedInference

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_SINGLE_SCENARIO_DEMO"] = "1"
if !isdefined(@__MODULE__, :SquareSingleScenarioSetup)
    include(joinpath(@__DIR__, "..", "single_scenario", "common.jl"))
end

MULTI_SCENARIO_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "representative_reconstruction", "generated", "square_four_to_four_multi_scenario"),
)

const MULTI_SCENARIO_PEAK_NOISE_SIGMA = DEFAULT_PEAK_NOISE_SIGMA
const MULTI_SCENARIO_PRIOR_SCALE = DEFAULT_PRIOR_SCALE
const MULTI_SCENARIO_ADAM_LEARNING_RATE = 0.02
const MULTI_SCENARIO_ADAM_GRAD_CLIP = Inf
const MULTI_SCENARIO_ADAM_DECAY_START = 0.65
const MULTI_SCENARIO_ADAM_FINAL_LR_SCALE = 0.02
const MULTI_SCENARIO_ESMDA_ENSEMBLE_SIZE = 192
const MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS = 6
const MULTI_SCENARIO_ESMDA_COVARIANCE_INFLATION = 1.02

struct MultiScenarioScenarioSpec
    label::String
    bc_levels::NTuple{5, Float64}
    bc_mode::Symbol
    bc_mode_strength::Float64
    ic_levels::NTuple{5, Float64}
    ic_mode::Symbol
    ic_mode_strength::Float64
end

struct MultiScenarioDataRegime
    label::String
    horizon_factor::Int
    scenario_count::Int
end

struct MultiScenarioDataset
    regime::MultiScenarioDataRegime
    labels::Vector{String}
    setups::Vector{SquareSingleScenarioSetup}
end

const MULTI_SCENARIO_BC_KNOTS = (0.0, 0.18, 0.42, 0.72, 1.0)
const MULTI_SCENARIO_IC_KNOTS = (0.0, 0.22, 0.50, 0.78, 1.0)


include(joinpath(@__DIR__, "scenarios.jl"))
include(joinpath(@__DIR__, "dataset.jl"))
include(joinpath(@__DIR__, "inference.jl"))
include(joinpath(@__DIR__, "metrics.jl"))
include(joinpath(@__DIR__, "plots.jl"))
