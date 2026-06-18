using LinearAlgebra

struct FinalStateSummary
    mean::Matrix{Float64}
    lower::Matrix{Float64}
    upper::Matrix{Float64}
end

function simulate_history(p, setup::FourToFourInferenceSetup)
    net = build_four_to_four_network(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return hist
end

function final_state_snapshot(p, setup::FourToFourInferenceSetup)
    hist = simulate_history(p, setup)
    return reduce(hcat, [road_history[:, end] for road_history in hist.road_histories])
end

function final_state_samples(param_samples::AbstractMatrix, setup::FourToFourInferenceSetup)
    n_cells = setup.n_cells
    n_roads = length(setup.road_lengths)
    n_samples = size(param_samples, 2)
    snapshots = Array{Float64}(undef, n_cells, n_roads, n_samples)

    for j in 1:n_samples
        snapshots[:, :, j] = final_state_snapshot(view(param_samples, :, j), setup)
    end

    return snapshots
end

function summarize_final_state_samples(samples::Array{Float64, 3}, weights::AbstractVector; level=0.90)
    @assert 0.0 < level < 1.0
    @assert size(samples, 3) == length(weights)

    weights_norm = TrafficNetworks.normalize_weights(weights)
    lower_q = (1 - level) / 2
    upper_q = 1 - lower_q

    n_cells, n_roads, _ = size(samples)
    mean_state = Matrix{Float64}(undef, n_cells, n_roads)
    lower_state = Matrix{Float64}(undef, n_cells, n_roads)
    upper_state = Matrix{Float64}(undef, n_cells, n_roads)

    for road in 1:n_roads
        for cell in 1:n_cells
            draws = vec(samples[cell, road, :])
            mean_state[cell, road] = dot(draws, weights_norm)
            lower_state[cell, road] = weighted_quantile(draws, weights_norm, lower_q)
            upper_state[cell, road] = weighted_quantile(draws, weights_norm, upper_q)
        end
    end

    return FinalStateSummary(mean_state, lower_state, upper_state)
end

function summarize_final_states(param_samples::AbstractMatrix, setup::FourToFourInferenceSetup, weights::AbstractVector; level=0.90)
    samples = final_state_samples(param_samples, setup)
    return summarize_final_state_samples(samples, weights; level=level)
end

function observed_road_label(road_id, setup::FourToFourInferenceSetup)
    return "r$(road_id)"
end

function road_cell_centers_meters(setup::FourToFourInferenceSetup, road_id::Int)
    road_length_m = 1000.0 * setup.road_lengths[road_id]
    dx = road_length_m / setup.n_cells
    return collect(range(dx / 2, road_length_m - dx / 2; length=setup.n_cells))
end

function road_initial_profile(setup::FourToFourInferenceSetup, road_id::Int)
    x_unit = collect(range(0.5 / setup.n_cells, 1.0 - 0.5 / setup.n_cells; length=setup.n_cells))
    return [setup.road_profiles[road_id](x) for x in x_unit]
end

