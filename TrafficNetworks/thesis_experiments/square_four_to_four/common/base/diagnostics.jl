function junction_turning_rmses(P_est, P_true)
    P_est_mats = turning_matrices(P_est)
    P_true_mats = turning_matrices(P_true)
    return [TrafficNetworks.rmse(vec(P_est_mats[junction]), vec(P_true_mats[junction])) for junction in 1:N_JUNCTIONS]
end

predictive_rmse(y_est::AbstractVector, y_true::AbstractVector) = TrafficNetworks.rmse(y_est, y_true)
final_state_rmse(estimate::AbstractMatrix, truth::AbstractMatrix) = TrafficNetworks.rmse(estimate, truth)

function observed_road_state_rmse(estimate::AbstractMatrix, truth::AbstractMatrix, road_ids::AbstractVector{Int})
    return TrafficNetworks.rmse(estimate[:, road_ids], truth[:, road_ids])
end

function variation_metrics(setup::SquareFourToFourSetup, y_true::AbstractVector)
    incoming_ic_spans = [maximum(road_initial_profile(setup, road_id)) - minimum(road_initial_profile(setup, road_id)) for road_id in EXTERNAL_INCOMING_ROADS]
    outgoing_ic_spans = [maximum(road_initial_profile(setup, road_id)) - minimum(road_initial_profile(setup, road_id)) for road_id in EXTERNAL_OUTGOING_ROADS]
    inflow_spans = [maximum(setup.inflows[k].(setup.control_times)) - minimum(setup.inflows[k].(setup.control_times)) for k in eachindex(setup.inflows)]
    observed_ic_spans = [maximum(road_initial_profile(setup, road_id)) - minimum(road_initial_profile(setup, road_id)) for road_id in setup.observed_road_ids]

    Y = reshape_observations(y_true, setup)
    observed_data_spans = [maximum(vec(Y[:, :, road_pos])) - minimum(vec(Y[:, :, road_pos])) for road_pos in 1:size(Y, 3)]

    return (
        incoming_ic_spans=incoming_ic_spans,
        outgoing_ic_spans=outgoing_ic_spans,
        inflow_spans=inflow_spans,
        observed_ic_spans=observed_ic_spans,
        observed_data_spans=observed_data_spans,
    )
end

function replace_junction_matrix(Ps, junction_id::Int, P_new::AbstractMatrix)
    updated = turning_matrices(Ps)
    updated[junction_id] = validate_turning_matrix(P_new)
    return updated
end

function sensitivity_diagnostics(P_true, setup::SquareFourToFourSetup)
    y_true = simulator(P_true, setup)
    all_uniform = [uniform_turning_matrix() for _ in 1:N_JUNCTIONS]
    y_all_uniform = simulator(all_uniform, setup)

    per_junction_uniform_rmse = Float64[]
    for junction in 1:N_JUNCTIONS
        y_j = simulator(replace_junction_matrix(P_true, junction, uniform_turning_matrix()), setup)
        push!(per_junction_uniform_rmse, predictive_rmse(y_j, y_true))
    end

    return (
        all_uniform_rmse=predictive_rmse(y_all_uniform, y_true),
        all_uniform_maxabs=maximum(abs.(y_all_uniform .- y_true)),
        per_junction_uniform_rmse=per_junction_uniform_rmse,
    )
end

function run_experiment_checks(setup::SquareFourToFourSetup, P_true)
    y_true = simulator(P_true, setup)
    variation = variation_metrics(setup, y_true)
    sensitivity = sensitivity_diagnostics(P_true, setup)

    incoming_usage = zeros(Int, length(ROAD_LABELS))
    outgoing_usage = zeros(Int, length(ROAD_LABELS))
    for roads in JUNCTION_INCOMING_ROADS, road_id in roads
        incoming_usage[road_id] += 1
    end
    for roads in JUNCTION_OUTGOING_ROADS, road_id in roads
        outgoing_usage[road_id] += 1
    end

    @testset "Square network experiment checks" begin
        @test length(y_true) == observation_length(setup)
        @test all(isfinite, y_true)
        @test length(setup.observed_road_ids) == 9
        @test length(intersect(setup.observed_road_ids, EXTERNAL_INCOMING_ROADS)) == 3
        @test length(intersect(setup.observed_road_ids, EXTERNAL_OUTGOING_ROADS)) == 3
        @test length(intersect(setup.observed_road_ids, CONNECTOR_ROADS)) == 3
        @test sort(unique(vcat(JUNCTION_INCOMING_ROADS...))) == sort(vcat(EXTERNAL_INCOMING_ROADS, CONNECTOR_ROADS))
        @test sort(unique(vcat(JUNCTION_OUTGOING_ROADS...))) == sort(vcat(EXTERNAL_OUTGOING_ROADS, CONNECTOR_ROADS))
        @test all(incoming_usage[road_id] == 1 for road_id in EXTERNAL_INCOMING_ROADS)
        @test all(outgoing_usage[road_id] == 0 for road_id in EXTERNAL_INCOMING_ROADS)
        @test all(incoming_usage[road_id] == 0 for road_id in EXTERNAL_OUTGOING_ROADS)
        @test all(outgoing_usage[road_id] == 1 for road_id in EXTERNAL_OUTGOING_ROADS)
        @test all(incoming_usage[road_id] == 1 for road_id in CONNECTOR_ROADS)
        @test all(outgoing_usage[road_id] == 1 for road_id in CONNECTOR_ROADS)
        @test minimum(variation.incoming_ic_spans) > 0.16
        @test minimum(variation.outgoing_ic_spans) > 0.11
        @test minimum(variation.inflow_spans) > 0.12
        @test minimum(variation.observed_data_spans) > 0.06
        @test sensitivity.all_uniform_rmse > 0.02
        @test minimum(sensitivity.per_junction_uniform_rmse) > 0.006
    end

    return (variation=variation, sensitivity=sensitivity, y_true=y_true)
end
