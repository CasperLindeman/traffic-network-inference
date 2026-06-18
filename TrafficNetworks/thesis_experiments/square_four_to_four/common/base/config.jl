if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using Statistics
using LinearAlgebra
using Distributions
using SimulationBasedInference
using TrafficNetworks
using Test
import TrafficNetworks: seconds_to_hours, minutes_to_hours, make_profile_sum, make_inflow_signal, make_piecewise_profile, make_piecewise_signal
import TrafficNetworks: normalize_weights, weighted_quantile, weighted_column_mean, sample_summary

DEFAULT_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_ESMDA_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "representative_reconstruction", "generated", "square_four_to_four", "square_four_to_four_esmda_vs_adam"),
)

const N_JUNCTIONS = 4
const ROADS_PER_JUNCTION_SIDE = 4
const LOGITS_PER_ROW = 3
const N_PARAMS = N_JUNCTIONS * ROADS_PER_JUNCTION_SIDE * LOGITS_PER_ROW

const SQUARE_NETWORK_SPEC_PATH = joinpath(@__DIR__, "..", "..", "..", "network_specs", "square_four_to_four.toml")
const SQUARE_NETWORK_SPEC = TrafficNetworks.load_experiment_network_spec(SQUARE_NETWORK_SPEC_PATH)
square_road_group(name::String) = Int.(SQUARE_NETWORK_SPEC.metadata["road_groups"][name])

const JUNCTION_LABELS = ["J1 (top-left)", "J2 (top-right)", "J3 (bottom-left)", "J4 (bottom-right)"]
const EXTERNAL_INCOMING_ROADS = square_road_group("external_incoming")
const EXTERNAL_OUTGOING_ROADS = square_road_group("external_outgoing")
const CONNECTOR_ROADS = square_road_group("connectors")
const OBSERVED_INCOMING_ROADS = square_road_group("observed_incoming")
const OBSERVED_OUTGOING_ROADS = square_road_group("observed_outgoing")
const OBSERVED_INTERNAL_ROADS = square_road_group("observed_internal")
const DEFAULT_OBSERVED_ROADS = vcat(OBSERVED_INCOMING_ROADS, OBSERVED_OUTGOING_ROADS, OBSERVED_INTERNAL_ROADS)

const JUNCTION_INCOMING_ROADS = [copy(junction.incoming) for junction in SQUARE_NETWORK_SPEC.junctions]
const JUNCTION_OUTGOING_ROADS = [copy(junction.outgoing) for junction in SQUARE_NETWORK_SPEC.junctions]
const ROAD_LABELS = TrafficNetworks.road_labels(SQUARE_NETWORK_SPEC)

const LATENT_NAMES = Tuple(
    Symbol("z_j$(junction)_r$(row)_k$(logit)")
    for junction in 1:N_JUNCTIONS for row in 1:ROADS_PER_JUNCTION_SIDE for logit in 1:LOGITS_PER_ROW
)
const SQUARE_TURNING_PARAMETERIZATION = TrafficNetworks.RowSoftmaxTurningParameterization(
    N_JUNCTIONS,
    ROADS_PER_JUNCTION_SIDE,
    ROADS_PER_JUNCTION_SIDE;
    latent_names=LATENT_NAMES,
)

struct SquareFourToFourSetup
    T::Float64
    CFL::Float64
    n_cells::Int
    control_times::Vector{Float64}
    observed_road_ids::Vector{Int}
    observed_cell_ids::Vector{Int}
    boundary_road_ids::Vector{Int}
    inflows::Vector{Function}
    road_profiles::Vector{Function}
    road_lengths::Vector{Float64}
    speed_limits::Vector{Int}
    generated_noise_sigma::Float64
    likelihood_sigma::Float64
end

struct FinalStateSummary
    mean::Matrix{Float64}
    lower::Matrix{Float64}
    upper::Matrix{Float64}
end

function default_sensor_cells(n_cells; n_sensors=3)
    return unique(round.(Int, collect(range(2, n_cells - 1; length=n_sensors))))
end

road_label(road_id::Int) = ROAD_LABELS[road_id]
uniform_turning_matrix() = fill(0.25, 4, 4)

function road_role_label(road_id::Int)
    if road_id in EXTERNAL_INCOMING_ROADS
        return "incoming"
    elseif road_id in EXTERNAL_OUTGOING_ROADS
        return "outgoing"
    end
    return "internal"
end

function square_stress_setup(; network_spec=SQUARE_NETWORK_SPEC)
    return SquareFourToFourSetup(
        network_spec.T,
        network_spec.CFL,
        network_spec.n_cells::Int,
        copy(network_spec.control_times),
        TrafficNetworks.observed_road_ids(network_spec),
        TrafficNetworks.observed_cell_ids(network_spec),
        TrafficNetworks.boundary_road_ids(network_spec),
        TrafficNetworks.boundary_inflows(network_spec),
        TrafficNetworks.road_profiles(network_spec),
        TrafficNetworks.road_lengths(network_spec),
        TrafficNetworks.speed_limits(network_spec),
        0.0,
        0.003,
    )
end
