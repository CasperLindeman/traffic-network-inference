# Shared helpers for square-four-to-four turning-outlier diagnostics.

function turning_row_swap_fit_rows(row_rows, dataset::MultiScenarioDataset, observations, P_post, P_true)
    true_mats = turning_matrices(P_true)
    y_post = simulator_dataset(P_post, dataset)
    post_fit_norm = normalized_rmse_values(y_post, observations.y_obs, observations.sigma_model)
    post_truth_rmse = rmse_values(y_post, observations.y_true)
    true_fit_norm = normalized_rmse_values(observations.y_true, observations.y_obs, observations.sigma_model)
    rows = NamedTuple[]

    for row in row_rows
        junction = parse_table_int(row.junction)
        incoming_row = parse_table_int(row.incoming_row)
        P_swap = replace_turning_row(P_post, junction, incoming_row, true_mats[junction][incoming_row, :])
        y_swap = simulator_dataset(P_swap, dataset)
        swap_fit_norm = normalized_rmse_values(y_swap, observations.y_obs, observations.sigma_model)
        swap_truth_rmse = rmse_values(y_swap, observations.y_true)
        swap_visibility_norm = normalized_rmse_values(y_swap, y_post, observations.sigma_model)

        push!(
            rows,
            (
                row_label=string(row.row_label),
                junction=junction,
                incoming_row=incoming_row,
                source_road=parse_table_int(row.source_road),
                source_observed=parse_table_bool(row.source_observed),
                observed_target_cols=string(row.observed_target_cols),
                row_tv_error=parse_table_float(row.row_tv_error),
                max_abs_entry_error=parse_table_float(row.max_abs_entry_error),
                dominant_true_col=parse_table_int(row.dominant_true_col),
                dominant_posterior_col=parse_table_int(row.dominant_posterior_col),
                largest_under_col=parse_table_int(row.largest_under_col),
                largest_over_col=parse_table_int(row.largest_over_col),
                post_fit_normalized_rmse=post_fit_norm,
                swap_fit_normalized_rmse=swap_fit_norm,
                truth_fit_normalized_rmse=true_fit_norm,
                fit_normalized_improvement=post_fit_norm - swap_fit_norm,
                post_truth_rmse=post_truth_rmse,
                swap_truth_rmse=swap_truth_rmse,
                truth_rmse_improvement=post_truth_rmse - swap_truth_rmse,
                swap_visibility_normalized=swap_visibility_norm,
                improvement_per_tv_error=parse_table_float(row.row_tv_error) > 0.0 ? (post_truth_rmse - swap_truth_rmse) / parse_table_float(row.row_tv_error) : 0.0,
            ),
        )
    end

    return rows
end

function turning_row_swap_observed_road_rows(swap_rows, row_rows, dataset::MultiScenarioDataset, observations, P_post, P_true)
    true_mats = turning_matrices(P_true)
    y_post = simulator_dataset(P_post, dataset)
    indices_by_road = dataset_observed_road_indices(dataset)
    rows = NamedTuple[]
    row_lookup = Dict(string(row.row_label) => row for row in row_rows)

    for swap_row in swap_rows
        source_row = row_lookup[swap_row.row_label]
        target_roads = parse_int_list_string(source_row.target_roads)
        P_swap = replace_turning_row(P_post, swap_row.junction, swap_row.incoming_row, true_mats[swap_row.junction][swap_row.incoming_row, :])
        y_swap = simulator_dataset(P_swap, dataset)

        for road_id in sort(collect(keys(indices_by_road)))
            idx = indices_by_road[road_id]
            post_fit_norm = normalized_rmse_values(y_post[idx], observations.y_obs[idx], observations.sigma_model[idx])
            swap_fit_norm = normalized_rmse_values(y_swap[idx], observations.y_obs[idx], observations.sigma_model[idx])
            post_truth = rmse_values(y_post[idx], observations.y_true[idx])
            swap_truth = rmse_values(y_swap[idx], observations.y_true[idx])

            push!(
                rows,
                (
                    row_label=swap_row.row_label,
                    junction=swap_row.junction,
                    incoming_row=swap_row.incoming_row,
                    source_road=swap_row.source_road,
                    source_observed=swap_row.source_observed,
                    row_tv_error=swap_row.row_tv_error,
                    observed_road=road_id,
                    observed_road_role=String(road_role_symbol(road_id)),
                    observed_road_is_source=road_id == swap_row.source_road,
                    observed_road_is_target=road_id in target_roads,
                    post_fit_normalized_rmse=post_fit_norm,
                    swap_fit_normalized_rmse=swap_fit_norm,
                    fit_normalized_improvement=post_fit_norm - swap_fit_norm,
                    post_truth_rmse=post_truth,
                    swap_truth_rmse=swap_truth,
                    truth_rmse_improvement=post_truth - swap_truth,
                    swap_visibility_normalized=normalized_rmse_values(y_swap[idx], y_post[idx], observations.sigma_model[idx]),
                ),
            )
        end
    end

    return rows
end

function enrich_swap_rows_with_best_roads(swap_rows, road_rows)
    enriched = NamedTuple[]
    road_groups = Dict{String, Vector{NamedTuple}}()
    for row in road_rows
        push!(get!(road_groups, row.row_label, NamedTuple[]), row)
    end

    for row in swap_rows
        roads = road_groups[row.row_label]
        best_truth = first(sort(roads; by=road -> -road.truth_rmse_improvement))
        best_fit = first(sort(roads; by=road -> -road.fit_normalized_improvement))
        push!(
            enriched,
            merge(
                row,
                (
                    best_truth_improvement_observed_road=best_truth.observed_road,
                    best_truth_improvement_road_role=best_truth.observed_road_role,
                    best_truth_road_improvement=best_truth.truth_rmse_improvement,
                    best_fit_improvement_observed_road=best_fit.observed_road,
                    best_fit_improvement_road_role=best_fit.observed_road_role,
                    best_fit_road_improvement=best_fit.fit_normalized_improvement,
                ),
            ),
        )
    end

    return enriched
end

weighted_loss_contributions(y_pred::AbstractVector, y_obs::AbstractVector, sigma::AbstractVector) =
    0.5 .* ((Float64.(y_pred) .- Float64.(y_obs)) ./ Float64.(sigma)) .^ 2

weighted_mean_loss(y_pred::AbstractVector, y_obs::AbstractVector, sigma::AbstractVector) =
    mean(weighted_loss_contributions(y_pred, y_obs, sigma))

function row_swap_likelihood_variant_rows(row_rows, dataset::MultiScenarioDataset, observations, P_post, P_true; specs=sigma_variant_specs())
    true_mats = turning_matrices(P_true)
    y_post = simulator_dataset(P_post, dataset)
    rows = NamedTuple[]

    for spec in specs
        sigma = sigma_variant(observations.y_obs, dataset.setups[1].physical_noise_peak_sigma, spec.kind, spec.floor_fraction)
        post_fit_norm = normalized_rmse_values(y_post, observations.y_obs, sigma)
        post_loss = weighted_mean_loss(y_post, observations.y_obs, sigma)
        true_fit_norm = normalized_rmse_values(observations.y_true, observations.y_obs, sigma)
        true_loss = weighted_mean_loss(observations.y_true, observations.y_obs, sigma)

        for row in row_rows
            junction = parse_table_int(row.junction)
            incoming_row = parse_table_int(row.incoming_row)
            P_swap = replace_turning_row(P_post, junction, incoming_row, true_mats[junction][incoming_row, :])
            y_swap = simulator_dataset(P_swap, dataset)
            swap_fit_norm = normalized_rmse_values(y_swap, observations.y_obs, sigma)
            swap_loss = weighted_mean_loss(y_swap, observations.y_obs, sigma)

            push!(
                rows,
                (
                    row_label=string(row.row_label),
                    junction=junction,
                    incoming_row=incoming_row,
                    source_road=parse_table_int(row.source_road),
                    source_observed=parse_table_bool(row.source_observed),
                    row_tv_error=parse_table_float(row.row_tv_error),
                    variant_label=spec.label,
                    variant_kind=String(spec.kind),
                    floor_fraction=spec.floor_fraction,
                    post_fit_normalized_rmse=post_fit_norm,
                    swap_fit_normalized_rmse=swap_fit_norm,
                    truth_fit_normalized_rmse=true_fit_norm,
                    fit_normalized_improvement=post_fit_norm - swap_fit_norm,
                    post_mean_loss=post_loss,
                    swap_mean_loss=swap_loss,
                    truth_mean_loss=true_loss,
                    mean_loss_improvement=post_loss - swap_loss,
                    swap_improves_noisy_rmse=swap_fit_norm < post_fit_norm,
                    swap_improves_noisy_loss=swap_loss < post_loss,
                ),
            )
        end
    end

    return rows
end

function row_swap_likelihood_variant_summary_rows(variant_rows)
    labels = sort(unique(getproperty.(variant_rows, :variant_label)))
    rows = NamedTuple[]

    for label in labels
        group = [row for row in variant_rows if row.variant_label == label]
        push!(
            rows,
            (
                variant_label=label,
                variant_kind=first(group).variant_kind,
                floor_fraction=first(group).floor_fraction,
                row_count=length(group),
                noisy_rmse_improved_count=count(row -> row.swap_improves_noisy_rmse, group),
                noisy_loss_improved_count=count(row -> row.swap_improves_noisy_loss, group),
                mean_fit_normalized_improvement=mean(getproperty.(group, :fit_normalized_improvement)),
                min_fit_normalized_improvement=minimum(getproperty.(group, :fit_normalized_improvement)),
                max_fit_normalized_improvement=maximum(getproperty.(group, :fit_normalized_improvement)),
                mean_loss_improvement=mean(getproperty.(group, :mean_loss_improvement)),
                min_loss_improvement=minimum(getproperty.(group, :mean_loss_improvement)),
                max_loss_improvement=maximum(getproperty.(group, :mean_loss_improvement)),
            ),
        )
    end

    return rows
end

function row_swap_loss_contribution_rows(swap_rows, row_rows, dataset::MultiScenarioDataset, observations, P_post, P_true; variant_label="hard_floor_0p03", variant_kind=:hard_floor, floor_fraction=0.03, max_rows_per_swap=25)
    true_mats = turning_matrices(P_true)
    metadata = observation_metadata_lookup(dataset)
    row_lookup = Dict(string(row.row_label) => row for row in row_rows)
    sigma = sigma_variant(observations.y_obs, dataset.setups[1].physical_noise_peak_sigma, variant_kind, floor_fraction)
    y_post = simulator_dataset(P_post, dataset)
    post_contrib = weighted_loss_contributions(y_post, observations.y_obs, sigma)
    post_residual = (y_post .- observations.y_obs) ./ sigma
    true_residual = (observations.y_true .- observations.y_obs) ./ sigma
    rows = NamedTuple[]

    for swap_row in swap_rows
        source_row = row_lookup[swap_row.row_label]
        junction = swap_row.junction
        incoming_row = swap_row.incoming_row
        P_swap = replace_turning_row(P_post, junction, incoming_row, true_mats[junction][incoming_row, :])
        y_swap = simulator_dataset(P_swap, dataset)
        swap_contrib = weighted_loss_contributions(y_swap, observations.y_obs, sigma)
        swap_residual = (y_swap .- observations.y_obs) ./ sigma
        contribution_improvement = post_contrib .- swap_contrib
        order = sortperm(abs.(contribution_improvement); rev=true)

        for (rank, idx) in enumerate(first(order, min(max_rows_per_swap, length(order))))
            meta = metadata[idx]
            push!(
                rows,
                merge(
                    (
                        row_label=swap_row.row_label,
                        junction=junction,
                        incoming_row=incoming_row,
                        source_road=swap_row.source_road,
                        source_observed=swap_row.source_observed,
                        row_tv_error=swap_row.row_tv_error,
                        observed_target_cols=string(source_row.observed_target_cols),
                        rank_within_swap=rank,
                        variant_label=variant_label,
                        y_true=observations.y_true[idx],
                        y_obs=observations.y_obs[idx],
                        y_post=y_post[idx],
                        y_swap=y_swap[idx],
                        sigma_true=observations.sigma_true[idx],
                        sigma_model=sigma[idx],
                        flux_shape_obs=parabolic_noise_shape(observations.y_obs[idx]),
                        true_standardized_residual=true_residual[idx],
                        post_standardized_residual=post_residual[idx],
                        swap_standardized_residual=swap_residual[idx],
                        post_loss_contribution=post_contrib[idx],
                        swap_loss_contribution=swap_contrib[idx],
                        contribution_improvement=contribution_improvement[idx],
                        abs_contribution_improvement=abs(contribution_improvement[idx]),
                        swap_improves_observation=contribution_improvement[idx] > 0.0,
                    ),
                    meta,
                ),
            )
        end
    end

    return rows
end

