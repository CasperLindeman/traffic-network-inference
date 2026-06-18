if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Statistics
using LinearAlgebra
using ForwardDiff
using Plots

using TrafficNetworks
import TrafficNetworks: table_value_string, write_namedtuple_table, write_config_file
import TrafficNetworks: parabolic_noise_shape, physical_noise_sigma, inference_sigma_from_observation

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_ESMDA_ADAM_DEMO"] = "1"
if !isdefined(@__MODULE__, :SquareFourToFourSetup)
    include(joinpath(@__DIR__, "..", "base", "esmda_vs_adam.jl"))
end

SINGLE_SCENARIO_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_SINGLE_SCENARIO_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "representative_reconstruction", "generated", "square_four_to_four_single_scenario"),
)
const ALL_SQUARE_ROAD_IDS = collect(1:length(ROAD_LABELS))
const SINGLE_SCENARIO_LEGACY_TEMPLATE = square_stress_setup()
const DEFAULT_SENSOR_FRACTIONS = [
    (cell_id - 0.5) / SINGLE_SCENARIO_LEGACY_TEMPLATE.n_cells
    for cell_id in default_sensor_cells(SINGLE_SCENARIO_LEGACY_TEMPLATE.n_cells; n_sensors=4)
]
const DEFAULT_BASE_LENGTH_KM = 0.05
const DEFAULT_CELLS_PER_BASE_LENGTH = 2
const DEFAULT_PRIOR_SCALE = 1.25
const DEFAULT_PEAK_NOISE_SIGMA = 0.08
const DEFAULT_SIGMA_FLOOR_FRACTION = 0.03

struct SquareSingleScenarioSetup
    T::Float64
    CFL::Float64
    base_length_km::Float64
    cells_per_base_length::Int
    control_times::Vector{Float64}
    observed_road_ids::Vector{Int}
    sensor_fractions::Any
    boundary_road_ids::Vector{Int}
    inflows::Vector{Function}
    road_profiles::Vector{Function}
    road_length_multipliers::Vector{Int}
    speed_limits::Vector{Int}
    physical_noise_peak_sigma::Float64
end

struct SingleScenarioObservationData
    y_true::Vector{Float64}
    y_obs::Vector{Float64}
    sigma_true::Vector{Float64}
    sigma_model::Vector{Float64}
    clip_fraction::Float64
end


include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "simulation.jl"))
include(joinpath(@__DIR__, "inference.jl"))
include(joinpath(@__DIR__, "metrics.jl"))
include(joinpath(@__DIR__, "plots.jl"))
