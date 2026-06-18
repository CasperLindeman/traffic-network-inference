# Shared helpers for square-four-to-four single-scenario experiments.

function single_scenario_parameter_vector(p::AbstractVector)
    return TrafficNetworks.parameter_vector(p, SQUARE_TURNING_PARAMETERIZATION)
end

single_scenario_parameter_vector(p::NamedTuple) = TrafficNetworks.parameter_vector(p, SQUARE_TURNING_PARAMETERIZATION)

stable_row_softmax_single_scenario(z::AbstractVector) = TrafficNetworks.stable_row_softmax(z)

function turning_matrices_single_scenario_forwarddiff(p)
    return TrafficNetworks.turning_matrices_from_logits(
        p,
        SQUARE_TURNING_PARAMETERIZATION;
        validate=false,
    )
end
function single_scenario_state_eltype(p)
    return promote_type(eltype(single_scenario_parameter_vector(p)), Float64)
end

function build_square_network_from_matrices(Ps, setup::SquareSingleScenarioSetup; state_eltype::Type{<:Real}=Float64)
    road_ids = all_square_road_ids()
    road_blocks = [
        road_length_multiplier(setup, road_id)
        for road_id in road_ids
    ]
    scaled_profiles = [
        TrafficNetworks.scale_profile_domain(setup.road_profiles[road_id], road_length_multiplier(setup, road_id))
        for road_id in road_ids
    ]
    rules = [
        TurningFractionRule(Ps[junction_id])
        for junction_id in 1:N_JUNCTIONS
    ]

    return TrafficNetworks.build_block_experiment_network(
        SQUARE_NETWORK_SPEC,
        rules,
        road_blocks,
        setup.base_length_km,
        setup.cells_per_base_length,
        scaled_profiles,
        setup.speed_limits;
        T=setup.T,
        CFL=setup.CFL,
        boundary_inflow_values=setup.inflows,
        state_eltype=state_eltype,
    )
end

function build_square_network(p, setup::SquareSingleScenarioSetup)
    return build_square_network_from_matrices(turning_matrices(p), setup)
end

function build_square_network_forwarddiff(p, setup::SquareSingleScenarioSetup)
    return build_square_network_from_matrices(
        turning_matrices_single_scenario_forwarddiff(p),
        setup;
        state_eltype=single_scenario_state_eltype(p),
    )
end

function simulate_history(p, setup::SquareSingleScenarioSetup)
    net = build_square_network(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return hist
end

function simulate_history_forwarddiff(p, setup::SquareSingleScenarioSetup)
    net = build_square_network_forwarddiff(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return hist
end

function flatten_observations(hist::SimulationHistory, setup::SquareSingleScenarioSetup)
    return TrafficNetworks.flatten_observation_blocks(
        hist,
        observation_blocks(setup),
        observation_length(setup),
    )
end

function flatten_observations_forwarddiff(hist::SimulationHistory, setup::SquareSingleScenarioSetup)
    return TrafficNetworks.flatten_observation_blocks(
        hist,
        observation_blocks(setup),
        observation_length(setup),
    )
end
simulator(p, setup::SquareSingleScenarioSetup) = flatten_observations(simulate_history(p, setup), setup)
simulator_forwarddiff(p, setup::SquareSingleScenarioSetup) = flatten_observations_forwarddiff(simulate_history_forwarddiff(p, setup), setup)

function final_state_snapshot(p, setup::SquareSingleScenarioSetup)
    hist = simulate_history(p, setup)
    return [Float64.(hist.road_histories[road_id][:, end]) for road_id in all_square_road_ids()]
end

function flattened_state(snapshot::AbstractVector{<:AbstractVector}, road_ids::AbstractVector{Int})
    return vcat([Float64.(snapshot[road_id]) for road_id in road_ids]...)
end

function final_state_rmse(estimate::AbstractVector{<:AbstractVector}, truth::AbstractVector{<:AbstractVector}; road_ids=all_square_road_ids())
    est = flattened_state(estimate, road_ids)
    ref = flattened_state(truth, road_ids)
    return sqrt(mean((est .- ref) .^ 2))
end

observed_road_state_rmse(estimate::AbstractVector{<:AbstractVector}, truth::AbstractVector{<:AbstractVector}, road_ids::AbstractVector{Int}) =
    final_state_rmse(estimate, truth; road_ids=road_ids)

function generate_physical_observations(
    y_true::AbstractVector,
    peak_noise_sigma::Real,
    rng::AbstractRNG;
    floor_fraction=DEFAULT_SIGMA_FLOOR_FRACTION,
)
    sigma_true = physical_noise_sigma(y_true, peak_noise_sigma)
    y_raw = y_true .+ sigma_true .* randn(rng, length(y_true))
    y_obs = clamp.(y_raw, 0.0, 1.0)
    sigma_model = inference_sigma_from_observation(y_obs, peak_noise_sigma; floor_fraction=floor_fraction)

    return SingleScenarioObservationData(
        Float64.(y_true),
        Float64.(y_obs),
        Float64.(sigma_true),
        Float64.(sigma_model),
        mean((y_raw .< 0.0) .| (y_raw .> 1.0)),
    )
end

function single_scenario_problem_data(setup::SquareSingleScenarioSetup; seed=1, floor_fraction=DEFAULT_SIGMA_FLOOR_FRACTION)
    rng = MersenneTwister(seed)
    P_true = true_turning_matrices()
    y_true = simulator(P_true, setup)
    observations = generate_physical_observations(
        y_true,
        setup.physical_noise_peak_sigma,
        rng;
        floor_fraction=floor_fraction,
    )
    true_state = final_state_snapshot(P_true, setup)
    return (
        setup=setup,
        P_true=P_true,
        y_true=observations.y_true,
        y_obs=observations.y_obs,
        sigma_true=observations.sigma_true,
        sigma_model=observations.sigma_model,
        clip_fraction=observations.clip_fraction,
        true_state=true_state,
    )
end

