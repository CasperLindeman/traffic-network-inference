function build_square_network(p, setup::SquareFourToFourSetup)
    Ps = turning_matrices(p)
    rules = [TurningFractionRule(Ps[junction]) for junction in 1:N_JUNCTIONS]
    return TrafficNetworks.build_experiment_network(
        SQUARE_NETWORK_SPEC,
        rules;
        T=setup.T,
        CFL=setup.CFL,
        road_length_values=setup.road_lengths,
        road_profile_values=setup.road_profiles,
        speed_limit_values=setup.speed_limits,
        boundary_inflow_values=setup.inflows,
    )
end

observation_length(setup::SquareFourToFourSetup) =
    TrafficNetworks.cell_observation_length(
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )

function flatten_observations(hist::SimulationHistory, setup::SquareFourToFourSetup)
    return TrafficNetworks.flatten_cell_observations(
        hist,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end
function simulate_history(p, setup::SquareFourToFourSetup)
    net = build_square_network(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return hist
end

simulator(p, setup::SquareFourToFourSetup) = flatten_observations(simulate_history(p, setup), setup)

function final_state_snapshot(p, setup::SquareFourToFourSetup)
    hist = simulate_history(p, setup)
    return reduce(hcat, [road_history[:, end] for road_history in hist.road_histories])
end

function state_snapshot(p, setup::SquareFourToFourSetup, time_index::Int)
    hist = simulate_history(p, setup)
    return reduce(hcat, [road_history[:, time_index] for road_history in hist.road_histories])
end

function road_initial_profile(setup::SquareFourToFourSetup, road_id::Int)
    x_unit = collect(range(0.5 / setup.n_cells, 1.0 - 0.5 / setup.n_cells; length=setup.n_cells))
    return [setup.road_profiles[road_id](x) for x in x_unit]
end

function road_cell_centers_meters(setup::SquareFourToFourSetup, road_id::Int)
    road_length_m = 1000.0 * setup.road_lengths[road_id]
    dx = road_length_m / setup.n_cells
    return collect(range(dx / 2, road_length_m - dx / 2; length=setup.n_cells))
end

function reshape_observations(y::AbstractVector, setup::SquareFourToFourSetup)
    return TrafficNetworks.reshape_cell_observations(
        y,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end

function posterior_predictive_mean(sol, weights)
    pred_ens = Array(get_observables(sol).y)
    return TrafficNetworks.prediction_ensemble_mean(pred_ens, weights)
end
function final_state_samples(param_samples::AbstractMatrix, setup::SquareFourToFourSetup)
    n_samples = size(param_samples, 2)
    snapshots = Array{Float64}(undef, setup.n_cells, length(ROAD_LABELS), n_samples)

    for j in 1:n_samples
        snapshots[:, :, j] = final_state_snapshot(view(param_samples, :, j), setup)
    end

    return snapshots
end

function summarize_final_state_samples(samples::Array{Float64, 3}, weights::AbstractVector; level=0.90)
    @assert 0.0 < level < 1.0
    @assert size(samples, 3) == length(weights)

    weights_norm = normalize_weights(weights)
    lower_q = (1 - level) / 2
    upper_q = 1 - lower_q
    n_cells, n_roads, _ = size(samples)

    mean_state = Matrix{Float64}(undef, n_cells, n_roads)
    lower_state = Matrix{Float64}(undef, n_cells, n_roads)
    upper_state = Matrix{Float64}(undef, n_cells, n_roads)

    for road_id in 1:n_roads
        for cell in 1:n_cells
            draws = vec(samples[cell, road_id, :])
            mean_state[cell, road_id] = dot(draws, weights_norm)
            lower_state[cell, road_id] = weighted_quantile(draws, weights_norm, lower_q)
            upper_state[cell, road_id] = weighted_quantile(draws, weights_norm, upper_q)
        end
    end

    return FinalStateSummary(mean_state, lower_state, upper_state)
end

function summarize_final_states(param_samples::AbstractMatrix, setup::SquareFourToFourSetup, weights::AbstractVector; level=0.90)
    return summarize_final_state_samples(final_state_samples(param_samples, setup), weights; level=level)
end
