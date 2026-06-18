# Shared helpers for square-four-to-four turning-outlier diagnostics.

function turning_entry_diagnostic_rows(
    dataset::MultiScenarioDataset,
    observations,
    esmda,
    P_true;
    sensitivity_step=1e-3,
)
    setup = first(dataset.setups)
    metadata = turning_entry_metadata(setup)
    activity = road_activity_summary(P_true, dataset)
    sensitivities = Dict(
        row.global_entry => row for row in turning_entry_sensitivity_rows(
            metadata,
            P_true,
            dataset,
            observations.sigma_model;
            step=sensitivity_step,
        )
    )
    stats = summarize_square_turning_fraction_samples(esmda.fraction_samples, esmda.weights)
    abs_errors = abs.(stats.means .- esmda.entry_true)
    large_error_cutoff = quantile(abs_errors, 0.90)
    rows = NamedTuple[]

    for meta in metadata
        idx = meta.global_entry
        truth = esmda.entry_true[idx]
        mean_est = stats.means[idx]
        lower = esmda.entry_ci_05[idx]
        upper = esmda.entry_ci_95[idx]
        width = upper - lower
        miss_gap = truth < lower ? lower - truth : truth > upper ? truth - upper : 0.0
        sens = sensitivities[idx]
        incoming_activity = activity[meta.incoming_road]
        outgoing_activity = activity[meta.outgoing_road]

        push!(
            rows,
            merge(
                meta,
                (
                    truth=truth,
                    posterior_mean=mean_est,
                    posterior_median=stats.medians[idx],
                    posterior_q1=stats.q1[idx],
                    posterior_q3=stats.q3[idx],
                    ci05=lower,
                    ci95=upper,
                    ci_width=width,
                    covered_by_ci90=miss_gap == 0.0,
                    signed_error=mean_est - truth,
                    abs_error=abs(mean_est - truth),
                    normalized_ci_gap=width > 0.0 ? miss_gap / width : (miss_gap > 0.0 ? Inf : 0.0),
                    large_error=abs(mean_est - truth) >= large_error_cutoff,
                    source_mean_density=incoming_activity.mean_density,
                    source_density_span=incoming_activity.density_span,
                    source_density_std=incoming_activity.density_std,
                    target_mean_density=outgoing_activity.mean_density,
                    target_density_span=outgoing_activity.density_span,
                    target_density_std=outgoing_activity.density_std,
                    raw_sensitivity=sens.raw_sensitivity,
                    normalized_sensitivity=sens.normalized_sensitivity,
                    max_abs_normalized_sensitivity=sens.max_abs_normalized_sensitivity,
                ),
            ),
        )
    end

    return rows
end

function group_metric_row(group_kind::String, group_label::String, rows)
    abs_errors = getproperty.(rows, :abs_error)
    ci_widths = getproperty.(rows, :ci_width)
    sensitivities = getproperty.(rows, :normalized_sensitivity)
    distances = getproperty.(rows, :target_observation_distance)
    finite_distances = [distance for distance in distances if isfinite(distance)]

    return (
        group_kind=group_kind,
        group_label=group_label,
        entry_count=length(rows),
        large_error_count=count(row -> row.large_error, rows),
        ci_miss_count=count(row -> !row.covered_by_ci90, rows),
        no_downstream_observation_count=count(row -> !isfinite(row.target_observation_distance), rows),
        mean_abs_error=mean(abs_errors),
        median_abs_error=quantile(abs_errors, 0.50),
        max_abs_error=maximum(abs_errors),
        mean_ci_width=mean(ci_widths),
        mean_normalized_sensitivity=mean(sensitivities),
        median_normalized_sensitivity=quantile(sensitivities, 0.50),
        finite_downstream_distance_mean=isempty(finite_distances) ? NaN : mean(finite_distances),
    )
end

function append_group_rows!(summary_rows, group_kind::String, rows, keyfun)
    groups = Dict{String, Vector{NamedTuple}}()
    for row in rows
        label = keyfun(row)
        push!(get!(groups, label, NamedTuple[]), row)
    end

    for label in sort(collect(keys(groups)))
        push!(summary_rows, group_metric_row(group_kind, label, groups[label]))
    end

    return summary_rows
end

function turning_entry_group_summary_rows(rows)
    summary_rows = NamedTuple[]
    append_group_rows!(summary_rows, "direct_observation", rows, row -> row.direct_observation_class)
    append_group_rows!(summary_rows, "downstream_observation", rows, row -> row.downstream_observation_class)
    append_group_rows!(summary_rows, "incoming_role", rows, row -> row.incoming_role)
    append_group_rows!(summary_rows, "outgoing_role", rows, row -> row.outgoing_role)
    append_group_rows!(summary_rows, "junction", rows, row -> row.junction_label)
    return summary_rows
end

function vector_summary_string(values::AbstractVector; digits=3)
    return join([@sprintf("%.*f", digits, Float64(value)) for value in values], ",")
end

function int_vector_string(values::AbstractVector)
    isempty(values) ? "" : join(string.(values), ",")
end

function bool_vector_string(values::AbstractVector)
    isempty(values) ? "" : join([value ? "true" : "false" for value in values], ",")
end

function turning_row_diagnostic_rows(
    entry_rows,
    dataset::MultiScenarioDataset,
    observations,
    esmda,
    P_true;
    sensitivity_step=1e-3,
)
    setup = first(dataset.setups)
    entry_by_position = Dict((row.junction, row.incoming_row, row.outgoing_col) => row for row in entry_rows)
    true_mats = turning_matrices(P_true)
    post_mats = turning_matrices(esmda.P_post_mean)
    rows = NamedTuple[]

    for junction in 1:N_JUNCTIONS
        for incoming_row in 1:4
            source_road = JUNCTION_INCOMING_ROADS[junction][incoming_row]
            target_roads = JUNCTION_OUTGOING_ROADS[junction]
            target_observed = [road_id in setup.observed_road_ids for road_id in target_roads]
            target_distances = [entry_by_position[(junction, incoming_row, col)].target_observation_distance for col in 1:4]
            true_row = Float64.(vec(true_mats[junction][incoming_row, :]))
            post_row = Float64.(vec(post_mats[junction][incoming_row, :]))
            signed_error = post_row .- true_row
            abs_error = abs.(signed_error)

            row_l1_error = sum(abs_error)
            row_tv_error = 0.5 * row_l1_error
            row_l2_error = sqrt(sum(signed_error .^ 2))
            max_abs_error = maximum(abs_error)
            dominant_true_col = argmax(true_row)
            dominant_post_col = argmax(post_row)
            largest_over_col = argmax(signed_error)
            largest_under_col = argmin(signed_error)
            largest_abs_col = argmax(abs_error)
            observed_target_cols = [col for col in 1:4 if target_observed[col]]
            unobserved_target_cols = [col for col in 1:4 if !target_observed[col]]

            row_replaced = replace_turning_row(P_true, junction, incoming_row, post_row)
            y_row_replaced = simulator_dataset(row_replaced, dataset)
            row_raw_observation_signal = sqrt(mean((y_row_replaced .- observations.y_true) .^ 2))
            row_normalized_observation_signal = sqrt(mean(((y_row_replaced .- observations.y_true) ./ observations.sigma_model) .^ 2))
            row_signal_per_tv_error = row_tv_error > 0.0 ? row_normalized_observation_signal / row_tv_error : 0.0

            entry_sensitivities = [entry_by_position[(junction, incoming_row, col)].normalized_sensitivity for col in 1:4]
            pair_values = Float64[]
            pair_labels = String[]
            for col_a in 1:3
                for col_b in (col_a + 1):4
                    push!(
                        pair_values,
                        pair_contrast_sensitivity(
                            P_true,
                            dataset,
                            observations.sigma_model,
                            junction,
                            incoming_row,
                            col_a,
                            col_b;
                            step=sensitivity_step,
                        ),
                    )
                    push!(pair_labels, @sprintf("%d<->%d", col_a, col_b))
                end
            end
            min_pair_idx = argmin(pair_values)
            under_over_pair_sensitivity = largest_under_col == largest_over_col ? NaN : pair_contrast_sensitivity(
                P_true,
                dataset,
                observations.sigma_model,
                junction,
                incoming_row,
                largest_over_col,
                largest_under_col;
                step=sensitivity_step,
            )

            push!(
                rows,
                (
                    row_label=@sprintf("J%d row %d", junction, incoming_row),
                    junction=junction,
                    junction_label=JUNCTION_LABELS[junction],
                    incoming_row=incoming_row,
                    source_road=source_road,
                    source_role=String(road_role_symbol(source_road)),
                    source_observed=source_road in setup.observed_road_ids,
                    target_roads=int_vector_string(target_roads),
                    target_road_values=copy(target_roads),
                    observed_target_cols=int_vector_string(observed_target_cols),
                    observed_target_col_values=copy(observed_target_cols),
                    unobserved_target_cols=int_vector_string(unobserved_target_cols),
                    unobserved_target_col_values=copy(unobserved_target_cols),
                    target_observed_flags=bool_vector_string(target_observed),
                    target_observed_values=copy(target_observed),
                    target_observed_count=count(identity, target_observed),
                    target_observation_distances=vector_summary_string(target_distances; digits=0),
                    target_observation_distance_values=copy(target_distances),
                    min_target_observation_distance=minimum(target_distances),
                    true_row=vector_summary_string(true_row),
                    true_row_values=copy(true_row),
                    posterior_mean_row=vector_summary_string(post_row),
                    posterior_mean_row_values=copy(post_row),
                    signed_error_row=vector_summary_string(signed_error),
                    signed_error_row_values=copy(signed_error),
                    abs_error_row=vector_summary_string(abs_error),
                    abs_error_row_values=copy(abs_error),
                    row_l1_error=row_l1_error,
                    row_tv_error=row_tv_error,
                    row_l2_error=row_l2_error,
                    max_abs_entry_error=max_abs_error,
                    largest_abs_col=largest_abs_col,
                    largest_abs_target_road=target_roads[largest_abs_col],
                    largest_abs_target_observed=target_observed[largest_abs_col],
                    dominant_true_col=dominant_true_col,
                    dominant_true_target_road=target_roads[dominant_true_col],
                    dominant_true_target_observed=target_observed[dominant_true_col],
                    dominant_posterior_col=dominant_post_col,
                    dominant_posterior_target_road=target_roads[dominant_post_col],
                    dominant_posterior_target_observed=target_observed[dominant_post_col],
                    dominant_target_changed=dominant_true_col != dominant_post_col,
                    largest_over_col=largest_over_col,
                    largest_over_target_road=target_roads[largest_over_col],
                    largest_over_target_observed=target_observed[largest_over_col],
                    largest_under_col=largest_under_col,
                    largest_under_target_road=target_roads[largest_under_col],
                    largest_under_target_observed=target_observed[largest_under_col],
                    mean_entry_normalized_sensitivity=mean(entry_sensitivities),
                    min_entry_normalized_sensitivity=minimum(entry_sensitivities),
                    max_entry_normalized_sensitivity=maximum(entry_sensitivities),
                    min_pair_contrast_sensitivity=minimum(pair_values),
                    mean_pair_contrast_sensitivity=mean(pair_values),
                    max_pair_contrast_sensitivity=maximum(pair_values),
                    weakest_pair_contrast=pair_labels[min_pair_idx],
                    under_over_pair_contrast_sensitivity=under_over_pair_sensitivity,
                    row_raw_observation_signal=row_raw_observation_signal,
                    row_normalized_observation_signal=row_normalized_observation_signal,
                    row_signal_per_tv_error=row_signal_per_tv_error,
                ),
            )
        end
    end

    return rows
end

function row_group_metric_row(group_kind::String, group_label::String, rows)
    tv_errors = getproperty.(rows, :row_tv_error)
    max_entry_errors = getproperty.(rows, :max_abs_entry_error)
    row_signals = getproperty.(rows, :row_normalized_observation_signal)
    pair_contrasts = getproperty.(rows, :min_pair_contrast_sensitivity)

    return (
        group_kind=group_kind,
        group_label=group_label,
        row_count=length(rows),
        dominant_target_changed_count=count(row -> row.dominant_target_changed, rows),
        source_unobserved_count=count(row -> !row.source_observed, rows),
        mean_observed_target_count=mean(getproperty.(rows, :target_observed_count)),
        mean_row_tv_error=mean(tv_errors),
        median_row_tv_error=quantile(tv_errors, 0.50),
        max_row_tv_error=maximum(tv_errors),
        mean_max_abs_entry_error=mean(max_entry_errors),
        max_abs_entry_error=maximum(max_entry_errors),
        mean_row_normalized_observation_signal=mean(row_signals),
        median_row_normalized_observation_signal=quantile(row_signals, 0.50),
        mean_min_pair_contrast_sensitivity=mean(pair_contrasts),
        median_min_pair_contrast_sensitivity=quantile(pair_contrasts, 0.50),
    )
end

function append_row_group_rows!(summary_rows, group_kind::String, rows, keyfun)
    groups = Dict{String, Vector{NamedTuple}}()
    for row in rows
        label = keyfun(row)
        push!(get!(groups, label, NamedTuple[]), row)
    end

    for label in sort(collect(keys(groups)))
        push!(summary_rows, row_group_metric_row(group_kind, label, groups[label]))
    end

    return summary_rows
end

function turning_row_group_summary_rows(row_rows)
    summary_rows = NamedTuple[]
    append_row_group_rows!(summary_rows, "source_observed", row_rows, row -> row.source_observed ? "source observed" : "source unobserved")
    append_row_group_rows!(summary_rows, "observed_target_count", row_rows, row -> @sprintf("%d observed targets", row.target_observed_count))
    append_row_group_rows!(summary_rows, "dominant_target_changed", row_rows, row -> row.dominant_target_changed ? "dominant target changed" : "dominant target preserved")
    append_row_group_rows!(summary_rows, "junction", row_rows, row -> row.junction_label)
    return summary_rows
end

