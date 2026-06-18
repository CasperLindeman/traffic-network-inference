# Shared helpers for square-four-to-four multi-scenario experiments.

const MULTI_SCENARIO_MAX_BOUNDARY_INFLOW = 0.25

function scenario_inflow(_base_fun::Function, road_id::Int, spec::MultiScenarioScenarioSpec, base_horizon::Real)
    road_offset = spec.bc_mode_strength * incoming_mode_weight(road_id, spec.bc_mode)
    road_levels = [clamp(level + road_offset, 0.0, MULTI_SCENARIO_MAX_BOUNDARY_INFLOW) for level in spec.bc_levels]

    return t -> begin
        tau = wrap_time_to_horizon(t, base_horizon) / base_horizon
        clamp(piecewise_linear_value(tau, MULTI_SCENARIO_BC_KNOTS, road_levels), 0.0, MULTI_SCENARIO_MAX_BOUNDARY_INFLOW)
    end
end

function scenario_profile(_base_fun::Function, road_id::Int, spec::MultiScenarioScenarioSpec)
    road_bias = spec.ic_mode_strength * profile_mode_weight(road_id, spec.ic_mode)
    road_levels = [clamp(level + road_bias, 0.0, 0.95) for level in spec.ic_levels]

    return x -> begin
        x_unit = clamp(x, 0.0, 1.0)
        clamp(piecewise_linear_value(x_unit, MULTI_SCENARIO_IC_KNOTS, road_levels), 0.0, 0.95)
    end
end

function extend_control_times(base_setup::SquareSingleScenarioSetup, horizon_factor::Int)
    dt = first(base_setup.control_times)
    return collect(dt:dt:(horizon_factor * base_setup.T))
end

function build_multi_scenario_setup(base_setup::SquareSingleScenarioSetup, spec::MultiScenarioScenarioSpec, regime::MultiScenarioDataRegime)
    base_horizon = base_setup.T
    inflows = [
        scenario_inflow(base_setup.inflows[idx], base_setup.boundary_road_ids[idx], spec, base_horizon)
        for idx in eachindex(base_setup.inflows)
    ]
    profiles = [scenario_profile(base_setup.road_profiles[i], i, spec) for i in eachindex(base_setup.road_profiles)]

    return SquareSingleScenarioSetup(
        regime.horizon_factor * base_setup.T,
        base_setup.CFL,
        base_setup.base_length_km,
        base_setup.cells_per_base_length,
        extend_control_times(base_setup, regime.horizon_factor),
        copy(base_setup.observed_road_ids),
        copy_sensor_spec(base_setup.sensor_fractions),
        copy(base_setup.boundary_road_ids),
        inflows,
        profiles,
        copy(base_setup.road_length_multipliers),
        copy(base_setup.speed_limits),
        base_setup.physical_noise_peak_sigma,
    )
end

function summarize_multi_scenario_scenarios(; scenario_count=12, peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA)
    preview_regime = MultiScenarioDataRegime("preview", 1, scenario_count)
    dataset = build_multi_scenario_dataset(preview_regime; peak_noise_sigma=peak_noise_sigma)
    base_horizon = dataset.setups[1].T
    time_grid = collect(range(0.0, base_horizon; length=9))
    x_grid = collect(range(0.0, 1.0; length=9))
    rows = NamedTuple[]

    for (label, setup) in zip(dataset.labels, dataset.setups)
        inflow_start = [inflow(0.0) for inflow in setup.inflows]
        inflow_values = reduce(vcat, [inflow.(time_grid) for inflow in setup.inflows])
        profile_front = [profile(0.0) for profile in setup.road_profiles]
        profile_back = [profile(1.0) for profile in setup.road_profiles]
        profile_values = reduce(vcat, [profile.(x_grid) for profile in setup.road_profiles])

        push!(
            rows,
            (
                label=label,
                bc_start_min=minimum(inflow_start),
                bc_start_max=maximum(inflow_start),
                bc_overall_min=minimum(inflow_values),
                bc_overall_max=maximum(inflow_values),
                ic_front_mean=mean(profile_front),
                ic_back_mean=mean(profile_back),
                ic_overall_min=minimum(profile_values),
                ic_overall_max=maximum(profile_values),
            ),
        )
    end

    return rows
end

function build_multi_scenario_dataset(regime::MultiScenarioDataRegime; peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA)
    base_setup = square_single_scenario_setup(peak_noise_sigma=peak_noise_sigma)
    specs = multi_scenario_scenario_specs(regime.scenario_count)
    labels = [spec.label for spec in specs]
    setups = [build_multi_scenario_setup(base_setup, spec, regime) for spec in specs]
    return MultiScenarioDataset(regime, labels, setups)
end

dataset_observation_length(dataset::MultiScenarioDataset) = sum(observation_length(setup) for setup in dataset.setups)
observation_multiplier(dataset::MultiScenarioDataset) = dataset_observation_length(dataset) / observation_length(square_single_scenario_setup(peak_noise_sigma=dataset.setups[1].physical_noise_peak_sigma))

function generate_physical_dataset_observations(P_true, dataset::MultiScenarioDataset; seed=1, floor_fraction=DEFAULT_SIGMA_FLOOR_FRACTION)
    y_true_blocks = Vector{Vector{Float64}}(undef, length(dataset.setups))
    y_obs_blocks = Vector{Vector{Float64}}(undef, length(dataset.setups))
    sigma_true_blocks = Vector{Vector{Float64}}(undef, length(dataset.setups))
    sigma_model_blocks = Vector{Vector{Float64}}(undef, length(dataset.setups))
    clip_fracs = Float64[]

    for (idx, setup) in enumerate(dataset.setups)
        rng = MersenneTwister(10_000 * seed + idx)
        y_true = simulator(P_true, setup)
        noisy = generate_physical_observations(y_true, setup.physical_noise_peak_sigma, rng; floor_fraction=floor_fraction)
        y_true_blocks[idx] = noisy.y_true
        y_obs_blocks[idx] = noisy.y_obs
        sigma_true_blocks[idx] = noisy.sigma_true
        sigma_model_blocks[idx] = noisy.sigma_model
        push!(clip_fracs, noisy.clip_fraction)
    end

    return (
        y_true=vcat(y_true_blocks...),
        y_obs=vcat(y_obs_blocks...),
        sigma_true=vcat(sigma_true_blocks...),
        sigma_model=vcat(sigma_model_blocks...),
        y_true_blocks=y_true_blocks,
        y_obs_blocks=y_obs_blocks,
        sigma_true_blocks=sigma_true_blocks,
        sigma_model_blocks=sigma_model_blocks,
        mean_clip_fraction=mean(clip_fracs),
        max_clip_fraction=maximum(clip_fracs),
    )
end

function simulator_dataset(p, dataset::MultiScenarioDataset)
    return vcat([simulator(p, setup) for setup in dataset.setups]...)
end

function simulator_dataset_forwarddiff(p, dataset::MultiScenarioDataset)
    blocks = [simulator_forwarddiff(p, setup) for setup in dataset.setups]
    T = promote_type(map(eltype, blocks)...)
    y = Vector{T}(undef, sum(length, blocks))
    write_pos = 1

    for block in blocks
        copyto!(y, write_pos, block, 1, length(block))
        write_pos += length(block)
    end

    return y
end

function weighted_residual_dataset(p, dataset::MultiScenarioDataset, y_obs::AbstractVector, sigma_model::AbstractVector)
    return (simulator_dataset(p, dataset) .- y_obs) ./ sigma_model
end

function weighted_residual_dataset_forwarddiff(p, dataset::MultiScenarioDataset, y_obs::AbstractVector, sigma_model::AbstractVector)
    return (simulator_dataset_forwarddiff(p, dataset) .- y_obs) ./ sigma_model
end

