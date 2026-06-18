# Shared helpers for square-four-to-four single-scenario experiments.

function result_metric_row(
    label::String,
    loss_kind::Symbol,
    restart_label::String,
    restart_seed::Int,
    result,
    setup::SquareSingleScenarioSetup,
    P_true,
    y_true,
    y_obs,
    true_state,
)
    P_est = turning_matrices(result.z)
    y_est = simulator(result.z, setup)
    est_state = final_state_snapshot(result.z, setup)
    return (
        label=label,
        loss_kind=string(loss_kind),
        restart_label=restart_label,
        restart_seed=restart_seed,
        iterations=result.iterations,
        solve_seconds=result.solve_seconds,
        final_loss=result.final_loss,
        best_loss=result.best_loss,
        final_raw_grad_norm=result.final_raw_grad_norm,
        clip_fraction=result.clip_fraction,
        loss_tail_relspan=result.loss_tail_relspan,
        grad_tail_relspan=result.grad_tail_relspan,
        turning_rmse=overall_turning_rmse(P_est, P_true),
        fit_rmse=predictive_rmse(y_est, y_obs),
        predictive_rmse=predictive_rmse(y_est, y_true),
        final_state_rmse_all=final_state_rmse(est_state, true_state),
        final_state_rmse_observed=observed_road_state_rmse(est_state, true_state, setup.observed_road_ids),
        final_state_rmse_unobserved=observed_road_state_rmse(est_state, true_state, unobserved_square_road_ids(setup)),
    )
end

function summarize_metric(values::AbstractVector{<:Real})
    return (
        mean=mean(values),
        median=median(values),
        minimum=minimum(values),
        maximum=maximum(values),
        std=length(values) > 1 ? std(values) : 0.0,
    )
end

function aggregate_restart_rows(rows)
    summaries = NamedTuple[]
    for loss_kind in unique(getproperty.(rows, :loss_kind))
        group_rows = filter(row -> row.loss_kind == loss_kind, rows)
        turning = summarize_metric(getproperty.(group_rows, :turning_rmse))
        state = summarize_metric(getproperty.(group_rows, :final_state_rmse_all))
        fit = summarize_metric(getproperty.(group_rows, :fit_rmse))
        grad = summarize_metric(getproperty.(group_rows, :final_raw_grad_norm))
        clip = summarize_metric(getproperty.(group_rows, :clip_fraction))

        push!(
            summaries,
            (
                loss_kind=loss_kind,
                restarts=length(group_rows),
                turning_rmse_mean=turning.mean,
                turning_rmse_median=turning.median,
                turning_rmse_min=turning.minimum,
                turning_rmse_max=turning.maximum,
                final_state_rmse_all_mean=state.mean,
                final_state_rmse_all_median=state.median,
                final_state_rmse_all_min=state.minimum,
                final_state_rmse_all_max=state.maximum,
                fit_rmse_mean=fit.mean,
                fit_rmse_median=fit.median,
                fit_rmse_min=fit.minimum,
                fit_rmse_max=fit.maximum,
                final_raw_grad_norm_mean=grad.mean,
                final_raw_grad_norm_max=grad.maximum,
                clip_fraction_mean=clip.mean,
                clip_fraction_max=clip.maximum,
            ),
        )
    end

    return summaries
end

function gradient_component_rows(
    z::AbstractVector,
    y_obs::AbstractVector,
    setup::SquareSingleScenarioSetup,
    sigma_model::AbstractVector;
    prior_scale=DEFAULT_PRIOR_SCALE,
    point_label="point",
)
    group_indices = observation_group_indices(setup)
    rows = NamedTuple[]

    data_loss = p -> begin
        residual = (simulator_forwarddiff(p, setup) .- y_obs) ./ sigma_model
        0.5 * sum(abs2, residual)
    end

    prior_loss = p -> 0.5 * sum(abs2, p ./ prior_scale)

    total_data_grad = ForwardDiff.gradient(data_loss, z)
    push!(rows, (point_label=point_label, component="data_total", grad_norm=norm(total_data_grad), obs_count=length(y_obs), mean_sigma=mean(sigma_model)))

    for component in (:incoming, :outgoing, :internal)
        idx = group_indices[component]
        component_loss = p -> begin
            residual = (simulator_forwarddiff(p, setup)[idx] .- y_obs[idx]) ./ sigma_model[idx]
            0.5 * sum(abs2, residual)
        end
        grad = ForwardDiff.gradient(component_loss, z)
        push!(
            rows,
            (
                point_label=point_label,
                component=string(component),
                grad_norm=norm(grad),
                obs_count=length(idx),
                mean_sigma=mean(sigma_model[idx]),
            ),
        )
    end

    prior_grad = ForwardDiff.gradient(prior_loss, z)
    push!(rows, (point_label=point_label, component="prior", grad_norm=norm(prior_grad), obs_count=length(z), mean_sigma=NaN))

    full_loss = p -> map_loss_forwarddiff(p, y_obs, setup, sigma_model; prior_scale=prior_scale)
    full_grad = ForwardDiff.gradient(full_loss, z)
    push!(rows, (point_label=point_label, component="objective_total", grad_norm=norm(full_grad), obs_count=length(y_obs), mean_sigma=mean(sigma_model)))

    return rows
end

