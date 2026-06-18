# Shared helpers for square-four-to-four multi-scenario experiments.

function average_state_metrics(P_est, setups::Vector{SquareSingleScenarioSetup}, P_true)
    predictive_vals = Float64[]
    full_vals = Float64[]
    observed_vals = Float64[]
    unobserved_vals = Float64[]

    for setup in setups
        y_true = simulator(P_true, setup)
        y_est = simulator(P_est, setup)
        true_state = final_state_snapshot(P_true, setup)
        est_state = final_state_snapshot(P_est, setup)

        push!(predictive_vals, predictive_rmse(y_est, y_true))
        push!(full_vals, final_state_rmse(est_state, true_state))
        push!(observed_vals, observed_road_state_rmse(est_state, true_state, setup.observed_road_ids))
        push!(unobserved_vals, observed_road_state_rmse(est_state, true_state, unobserved_square_road_ids(setup)))
    end

    return (
        predictive_rmse=mean(predictive_vals),
        final_state_rmse_all=mean(full_vals),
        final_state_rmse_observed=mean(observed_vals),
        final_state_rmse_unobserved=mean(unobserved_vals),
    )
end

function ensemble_mean_final_state_snapshot(
    param_samples::AbstractMatrix,
    weights::AbstractVector,
    setup::SquareSingleScenarioSetup,
)
    weights_norm = normalize_weights(weights)
    first_snapshot = final_state_snapshot(view(param_samples, :, 1), setup)
    mean_snapshot = [zeros(Float64, length(first_snapshot[road_id])) for road_id in eachindex(first_snapshot)]

    for member in 1:size(param_samples, 2)
        snapshot = member == 1 ? first_snapshot : final_state_snapshot(view(param_samples, :, member), setup)
        weight = weights_norm[member]
        for road_id in eachindex(mean_snapshot)
            mean_snapshot[road_id] .+= weight .* snapshot[road_id]
        end
    end

    return mean_snapshot
end

function average_ensemble_mean_state_metrics(
    param_samples::AbstractMatrix,
    weights::AbstractVector,
    setups::Vector{SquareSingleScenarioSetup},
    P_true,
)
    full_vals = Float64[]
    observed_vals = Float64[]
    unobserved_vals = Float64[]

    for setup in setups
        true_state = final_state_snapshot(P_true, setup)
        mean_state = ensemble_mean_final_state_snapshot(param_samples, weights, setup)

        push!(full_vals, final_state_rmse(mean_state, true_state))
        push!(observed_vals, observed_road_state_rmse(mean_state, true_state, setup.observed_road_ids))
        push!(unobserved_vals, observed_road_state_rmse(mean_state, true_state, unobserved_square_road_ids(setup)))
    end

    return (
        final_state_rmse_all=mean(full_vals),
        final_state_rmse_observed=mean(observed_vals),
        final_state_rmse_unobserved=mean(unobserved_vals),
    )
end

ensemble_mean_state_metrics(esmda, dataset::MultiScenarioDataset, P_true) =
    average_ensemble_mean_state_metrics(esmda.param_samples, esmda.weights, dataset.setups, P_true)

function multi_scenario_metric_row(
    regime::MultiScenarioDataset,
    method::Symbol,
    budget_label::String,
    requested_budget::Real,
    solve_seconds::Real,
    iterations::Int,
    P_est,
    y_est,
    P_true,
    y_true,
    ;
    state_metrics=nothing,
)
    resolved_state_metrics = state_metrics === nothing ? average_state_metrics(P_est, regime.setups, P_true) : state_metrics
    return (
        regime_label=regime.regime.label,
        scenario_count=regime.regime.scenario_count,
        horizon_factor=regime.regime.horizon_factor,
        observation_count=dataset_observation_length(regime),
        observation_multiplier=observation_multiplier(regime),
        method=string(method),
        budget_label=budget_label,
        requested_budget=Float64(requested_budget),
        solve_seconds=Float64(solve_seconds),
        iterations=iterations,
        turning_rmse=overall_turning_rmse(P_est, P_true),
        predictive_rmse=predictive_rmse(y_est, y_true),
        final_state_rmse_all=resolved_state_metrics.final_state_rmse_all,
        final_state_rmse_observed=resolved_state_metrics.final_state_rmse_observed,
        final_state_rmse_unobserved=resolved_state_metrics.final_state_rmse_unobserved,
    )
end

function choose_plateau_budget(rows; rel_tol=0.03)
    best_turning = minimum(getproperty.(rows, :turning_rmse))
    best_state = minimum(getproperty.(rows, :final_state_rmse_all))

    ordered = sort(rows; by=row -> row.solve_seconds)
    for row in ordered
        turning_ok = row.turning_rmse <= best_turning * (1.0 + rel_tol)
        state_ok = row.final_state_rmse_all <= best_state * (1.0 + rel_tol)
        if turning_ok && state_ok
            return (
                requested_budget=row.requested_budget,
                solve_seconds=row.solve_seconds,
                iterations=row.iterations,
                turning_rmse=row.turning_rmse,
                final_state_rmse_all=row.final_state_rmse_all,
                reached_plateau=true,
            )
        end
    end

    fallback = ordered[end]
    return (
        requested_budget=fallback.requested_budget,
        solve_seconds=fallback.solve_seconds,
        iterations=fallback.iterations,
        turning_rmse=fallback.turning_rmse,
        final_state_rmse_all=fallback.final_state_rmse_all,
        reached_plateau=false,
    )
end

