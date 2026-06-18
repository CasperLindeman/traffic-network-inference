"""
Reusable traffic-network simulation and inference tools.

The thesis experiments are organized around this workflow:

1. Define or load an `ExperimentNetworkSpec`.
2. Build a `RoadNetwork` with junction rules and boundary inflows.
3. Simulate the network with `simulate!`.
4. Flatten selected sensor histories and add physical observation noise.
5. Infer turning fractions using row-softmax parameterizations and weighted residuals.
6. Summarize recovery with RMSE, coverage, interval width, and table helpers.
"""
module TrafficNetworks

# LWR flux model
include("fluxes.jl")
using .Fluxes

# Roads, junctions, and boundaries
include("roads.jl")
using .Roads: Road, make_road, cfl_dt, update_road!

include("junctions.jl")
using .Junctions:
    Junction,
    compute_junction_fluxes,
    TurningFractionRule,
    AbstractJunctionRule,
    update_turning_fractions!

include("boundaries.jl")
using .Boundaries: Boundary, boundary_flux, boundary_cfl_flux, boundary_flux!, make_inflow_signal, make_piecewise_signal

# Network container
include("road_network.jl")
using .RoadNetworks: RoadNetwork

# Time stepping
include("solver.jl")
using .Solvers: simulate!, SimulationHistory

# Time, profile, and signal helpers
include("utils.jl")
using .Utils:
    clamp01,
    seconds_to_hours,
    minutes_to_hours,
    control_times_seconds,
    regular_control_times,
    make_profile,
    scale_profile_domain,
    make_profile_sum,
    make_piecewise_profile

# Statistical summaries, noisy observations, and table output
include("experiment_utils.jl")
using .ExperimentUtils:
    normalize_weights,
    weighted_quantile,
    weighted_column_mean,
    sample_summary,
    weighted_std,
    trapezoid_integral,
    normalize_importance_weights,
    gaussian_kernel1d,
    smooth_vector_clamped,
    smooth_matrix_clamped,
    weighted_kde_unit_interval,
    weighted_histogram_1d,
    weighted_histogram_2d,
    rmse,
    normalized_rmse,
    interval_coverage_mask,
    interval_coverage_rate,
    mean_interval_width,
    recovery_summary,
    simulate_noisy_observations,
    parabolic_noise_shape,
    physical_noise_sigma,
    inference_sigma_from_observation,
    generate_physical_observations,
    table_value_string,
    write_namedtuple_table,
    write_config_file

# Turning-fraction parameterizations and observation vectors
include("inference.jl")
using .InferenceTools:
    RowSoftmaxTurningParameterization,
    parameter_count,
    parameter_vector,
    stable_row_softmax,
    validate_row_stochastic_matrix,
    validate_row_stochastic_matrices,
    turning_matrix_from_logits,
    turning_matrices_from_logits,
    turning_entries,
    turning_entry_samples,
    entry_vector_to_turning_matrices,
    cell_observation_length,
    flatten_cell_observations,
    flatten_observation_blocks,
    paired_cell_observation_length,
    flatten_paired_cell_observations,
    reshape_cell_observations,
    weighted_residual,
    prediction_ensemble_mean

# TOML-backed experiment network construction
include("experiment_networks.jl")
using .ExperimentNetworks:
    DEFAULT_EXPERIMENT_CFL,
    ExperimentRoadSpec,
    ExperimentJunctionSpec,
    ExperimentBoundarySpec,
    ExperimentNetworkSpec,
    load_experiment_network_spec,
    road_ids,
    road_labels,
    road_blocks,
    road_lengths,
    speed_limits,
    road_profiles,
    boundary_road_ids,
    boundary_inflows,
    observed_road_ids,
    observed_cell_ids,
    road_cell_count,
    road_length_km,
    cell_center_distance_km,
    sensor_cell_id,
    paired_sensor_cell_ids,
    build_experiment_network,
    build_block_experiment_network,
    simulation_step_count,
    make_boundaries,
    make_roads_from_blocks,
    make_roads_from_lengths

# Visualization
include("visualization.jl")
using .RoadNetworkViz: RoadGeom, NetworkGeom, plot_network, plot_history, plot_road_history, animate_network

export
    # LWR flux model
    flux,
    dflux,
    godunov_flux,
    demand,
    supply,
    maxwavespeed,
    endpoint_wavespeed,

    # Core model components
    Road,
    make_road,
    cfl_dt,
    update_road!,
    Junction,
    compute_junction_fluxes,
    Boundary,
    boundary_flux,
    boundary_cfl_flux,
    boundary_flux!,
    RoadNetwork,
    simulate!,
    SimulationHistory,

    # Junction rules
    TurningFractionRule,
    AbstractJunctionRule,
    update_turning_fractions!,

    # Time, boundary, and initial-condition helpers
    clamp01,
    seconds_to_hours,
    minutes_to_hours,
    control_times_seconds,
    regular_control_times,
    make_inflow_signal,
    make_piecewise_signal,
    make_profile,
    scale_profile_domain,
    make_profile_sum,
    make_piecewise_profile,

    # Experiment network specifications and builders
    DEFAULT_EXPERIMENT_CFL,
    ExperimentRoadSpec,
    ExperimentJunctionSpec,
    ExperimentBoundarySpec,
    ExperimentNetworkSpec,
    load_experiment_network_spec,
    road_ids,
    road_labels,
    road_blocks,
    road_lengths,
    speed_limits,
    road_profiles,
    boundary_road_ids,
    boundary_inflows,
    observed_road_ids,
    observed_cell_ids,
    road_cell_count,
    road_length_km,
    cell_center_distance_km,
    sensor_cell_id,
    paired_sensor_cell_ids,
    build_experiment_network,
    build_block_experiment_network,
    simulation_step_count,
    make_boundaries,
    make_roads_from_blocks,
    make_roads_from_lengths,

    # Inference parameterizations and observation vectors
    RowSoftmaxTurningParameterization,
    parameter_count,
    parameter_vector,
    stable_row_softmax,
    validate_row_stochastic_matrix,
    validate_row_stochastic_matrices,
    turning_matrix_from_logits,
    turning_matrices_from_logits,
    turning_entries,
    turning_entry_samples,
    entry_vector_to_turning_matrices,
    cell_observation_length,
    flatten_cell_observations,
    flatten_observation_blocks,
    paired_cell_observation_length,
    flatten_paired_cell_observations,
    reshape_cell_observations,
    weighted_residual,
    prediction_ensemble_mean,

    # Statistical summaries, observation noise, and diagnostics
    normalize_weights,
    weighted_quantile,
    weighted_column_mean,
    sample_summary,
    weighted_std,
    trapezoid_integral,
    normalize_importance_weights,
    gaussian_kernel1d,
    smooth_vector_clamped,
    smooth_matrix_clamped,
    weighted_kde_unit_interval,
    weighted_histogram_1d,
    weighted_histogram_2d,
    rmse,
    normalized_rmse,
    interval_coverage_mask,
    interval_coverage_rate,
    mean_interval_width,
    recovery_summary,
    parabolic_noise_shape,
    physical_noise_sigma,
    inference_sigma_from_observation,
    generate_physical_observations,
    simulate_noisy_observations,
    table_value_string,
    write_namedtuple_table,
    write_config_file,

    # Visualization
    NetworkGeom,
    RoadGeom,
    plot_network,
    plot_history,
    plot_road_history,
    animate_network

end
