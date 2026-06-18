# Shared helpers for square-four-to-four turning-outlier diagnostics.

function write_likelihood_diagnostic_summary(
    noise_rows,
    noise_outlier_rows,
    variant_summary_rows,
    contribution_rows,
    output_path;
    current_variant_label="hard_floor_0p03",
)
    mkpath(dirname(output_path))
    current_noise = first([row for row in noise_rows if row.variant_label == current_variant_label])
    current_variant = first([row for row in variant_summary_rows if row.variant_label == current_variant_label])
    top_noise = first(noise_outlier_rows, min(8, length(noise_outlier_rows)))
    top_contrib = first(sort(contribution_rows; by=row -> -row.abs_contribution_improvement), min(12, length(contribution_rows)))

    open(output_path, "w") do io
        println(io, "Likelihood and noise-weight diagnostic")
        println(io, "======================================")
        println(io, @sprintf("Current hard floor fraction: %.3f", current_noise.floor_fraction))
        println(io, @sprintf("Current floor value: %.6g", current_noise.floor_value))
        println(io, @sprintf("Observations at floor: %d / %d (%.4f)", current_noise.at_floor_count, current_noise.observation_count, current_noise.at_floor_fraction))
        println(io, @sprintf("Max/min sigma ratio: %.3g", current_noise.max_to_min_sigma))
        println(io, @sprintf("Max/min weight ratio: %.3g", current_noise.max_to_min_weight))
        println(io, @sprintf("Max absolute standardized true-noise residual: %.3g", current_noise.max_abs_true_standardized))
        println(io, @sprintf("True-noise loss concentration top 1/5/10: %.4f / %.4f / %.4f", current_noise.true_loss_top1_fraction, current_noise.true_loss_top5_fraction, current_noise.true_loss_top10_fraction))
        println(io)

        println(io, "Alternative weighting checks")
        println(io, "----------------------------")
        for row in variant_summary_rows
            println(
                io,
                @sprintf(
                    "%-20s rows_where_swap_improves_loss=%2d/%2d mean_loss_improve=% .4g min=% .4g max=% .4g",
                    row.variant_label,
                    row.noisy_loss_improved_count,
                    row.row_count,
                    row.mean_loss_improvement,
                    row.min_loss_improvement,
                    row.max_loss_improvement,
                ),
            )
        end
        println(io)

        println(io, "Largest standardized residuals under true parameters")
        println(io, "----------------------------------------------------")
        for row in top_noise
            println(
                io,
                @sprintf(
                    "rank %2d idx=%4d scen=%d road=%d cell=%d time=%.2f min y_true=%.4f y_obs=%.4f sigma=%.4g r_true=% .3g at_floor=%s",
                    row.rank,
                    row.global_index,
                    row.scenario_index,
                    row.road_id,
                    row.sensor_cell,
                    row.time_min,
                    row.y_true,
                    row.y_obs,
                    row.sigma_model,
                    row.true_standardized_residual,
                    string(row.at_floor),
                ),
            )
        end
        println(io)

        println(io, "Largest row-swap loss-contribution changes")
        println(io, "------------------------------------------")
        for row in top_contrib
            println(
                io,
                @sprintf(
                    "%-8s rank=%2d idx=%4d scen=%d road=%d cell=%d time=%.2f min delta_loss=% .4g post_r=% .3g swap_r=% .3g true_r=% .3g sigma=%.4g",
                    row.row_label,
                    row.rank_within_swap,
                    row.global_index,
                    row.scenario_index,
                    row.road_id,
                    row.sensor_cell,
                    row.time_min,
                    row.contribution_improvement,
                    row.post_standardized_residual,
                    row.swap_standardized_residual,
                    row.true_standardized_residual,
                    row.sigma_model,
                ),
            )
        end
    end

    return output_path
end

function write_row_swap_summary(swap_rows, road_rows, output_path, dataset)
    ordered = sort(swap_rows; by=row -> -row.row_tv_error)
    improves_truth = count(row -> row.truth_rmse_improvement > 0.0, swap_rows)
    improves_fit = count(row -> row.fit_normalized_improvement > 0.0, swap_rows)
    top_rows = first(ordered, min(8, length(ordered)))

    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(io, "Turning-row swap-back fit diagnostic")
        println(io, "====================================")
        println(io, @sprintf("Regime: %s", dataset.regime.label))
        println(io, @sprintf("Rows tested: %d", length(swap_rows)))
        println(io, @sprintf("Rows where swapping row to truth improves prediction vs y_true: %d / %d", improves_truth, length(swap_rows)))
        println(io, @sprintf("Rows where swapping row to truth improves noisy-observation fit: %d / %d", improves_fit, length(swap_rows)))
        println(io)
        println(io, "Interpretation")
        println(io, "--------------")
        println(io, "Positive truth_rmse_improvement: that row error is genuinely hurting prediction.")
        println(io, "Near-zero truth_rmse_improvement: swapping the row barely matters for the simulated observations/states we compare.")
        println(io, "Positive noisy fit improvement is weaker evidence, because noisy observations do not equal truth.")
        println(io)
        println(io, "Worst row errors")
        println(io, "----------------")
        for row in top_rows
            println(
                io,
                @sprintf(
                    "%-8s TV=%.4f source=road %d observed=%s dominant=%d->%d truth_improve=%.6f fit_improve=%.6f visibility=%.3g best_truth_road=%d (%s, %.6f)",
                    row.row_label,
                    row.row_tv_error,
                    row.source_road,
                    string(row.source_observed),
                    row.dominant_true_col,
                    row.dominant_posterior_col,
                    row.truth_rmse_improvement,
                    row.fit_normalized_improvement,
                    row.swap_visibility_normalized,
                    row.best_truth_improvement_observed_road,
                    row.best_truth_improvement_road_role,
                    row.best_truth_road_improvement,
                ),
            )
        end
        println(io)
        println(io, "Observed-road effects for worst rows")
        println(io, "------------------------------------")
        for row in top_rows
            row_road_rows = sort([road for road in road_rows if road.row_label == row.row_label]; by=road -> -abs(road.truth_rmse_improvement))
            println(io, row.row_label)
            for road in first(row_road_rows, min(4, length(row_road_rows)))
                println(
                    io,
                    @sprintf(
                        "  road %d (%s): truth_improve=%.6f fit_improve=%.6f visibility=%.3g",
                        road.observed_road,
                        road.observed_road_role,
                        road.truth_rmse_improvement,
                        road.fit_normalized_improvement,
                        road.swap_visibility_normalized,
                    ),
                )
            end
        end
    end

    return output_path
end

function write_turning_row_swap_outputs(
    row_rows,
    dataset::MultiScenarioDataset,
    observations,
    P_post,
    P_true,
    output_dir;
    current_floor_fraction=DEFAULT_SIGMA_FLOOR_FRACTION,
)
    swap_rows_raw = turning_row_swap_fit_rows(row_rows, dataset, observations, P_post, P_true)
    road_rows = turning_row_swap_observed_road_rows(swap_rows_raw, row_rows, dataset, observations, P_post, P_true)
    swap_rows = enrich_swap_rows_with_best_roads(swap_rows_raw, road_rows)
    peak_noise_sigma = dataset.setups[1].physical_noise_peak_sigma
    noise_rows = noise_weight_summary_rows(observations, peak_noise_sigma)
    current_variant_label = hard_floor_variant_label(current_floor_fraction)
    noise_outlier_rows = noise_observation_outlier_rows(
        dataset,
        observations,
        peak_noise_sigma;
        variant_label=current_variant_label,
        variant_kind=:hard_floor,
        floor_fraction=current_floor_fraction,
    )
    variant_rows = row_swap_likelihood_variant_rows(row_rows, dataset, observations, P_post, P_true)
    variant_summary_rows = row_swap_likelihood_variant_summary_rows(variant_rows)
    contribution_rows = row_swap_loss_contribution_rows(
        swap_rows,
        row_rows,
        dataset,
        observations,
        P_post,
        P_true;
        variant_label=current_variant_label,
        variant_kind=:hard_floor,
        floor_fraction=current_floor_fraction,
    )

    metrics_path = write_namedtuple_table(swap_rows, joinpath(output_dir, "turning_row_swap_fit_metrics.tsv"))
    road_metrics_path = write_namedtuple_table(road_rows, joinpath(output_dir, "turning_row_swap_observed_road_metrics.tsv"))
    summary_path = write_row_swap_summary(swap_rows, road_rows, joinpath(output_dir, "turning_row_swap_fit_summary.txt"), dataset)
    noise_summary_path = write_namedtuple_table(noise_rows, joinpath(output_dir, "likelihood_noise_weight_summary.tsv"))
    noise_outlier_path = write_namedtuple_table(noise_outlier_rows, joinpath(output_dir, "likelihood_true_noise_outliers.tsv"))
    variant_path = write_namedtuple_table(variant_rows, joinpath(output_dir, "turning_row_swap_likelihood_variant_metrics.tsv"))
    variant_summary_path = write_namedtuple_table(variant_summary_rows, joinpath(output_dir, "turning_row_swap_likelihood_variant_summary.tsv"))
    contribution_path = write_namedtuple_table(contribution_rows, joinpath(output_dir, "turning_row_swap_loss_contributions.tsv"))
    likelihood_summary_path = write_likelihood_diagnostic_summary(
        noise_rows,
        noise_outlier_rows,
        variant_summary_rows,
        contribution_rows,
        joinpath(output_dir, "likelihood_noise_diagnostic_summary.txt"),
        current_variant_label=current_variant_label,
    )

    return (
        swap_rows=swap_rows,
        swap_road_rows=road_rows,
        noise_rows=noise_rows,
        noise_outlier_rows=noise_outlier_rows,
        variant_rows=variant_rows,
        variant_summary_rows=variant_summary_rows,
        contribution_rows=contribution_rows,
        swap_metrics_path=metrics_path,
        swap_road_metrics_path=road_metrics_path,
        swap_summary_path=summary_path,
        noise_summary_path=noise_summary_path,
        noise_outlier_path=noise_outlier_path,
        variant_path=variant_path,
        variant_summary_path=variant_summary_path,
        contribution_path=contribution_path,
        likelihood_summary_path=likelihood_summary_path,
    )
end

function distance_string(distance::Real)
    isfinite(distance) ? @sprintf("%.0f", distance) : "Inf"
end

function write_turning_outlier_summary(entry_rows, entry_group_rows, row_rows, row_group_rows, output_path, dataset, observations, esmda)
    rows = entry_rows
    sorted_rows = sort(rows; by=row -> -row.abs_error)
    top_rows = first(sorted_rows, min(12, length(sorted_rows)))
    threshold = 0.30
    threshold_rows = [row for row in sorted_rows if row.abs_error > threshold]
    sorted_row_rows = sort(row_rows; by=row -> -row.row_tv_error)
    top_row_rows = first(sorted_row_rows, min(8, length(sorted_row_rows)))
    top_row_count = min(5, length(sorted_row_rows))
    top_row_subset = first(sorted_row_rows, top_row_count)
    sensitivity_corr = finite_correlation(getproperty.(rows, :normalized_sensitivity), getproperty.(rows, :abs_error))
    direct_count_corr = finite_correlation(getproperty.(rows, :direct_observed_count), getproperty.(rows, :abs_error))
    distance_corr = finite_correlation(getproperty.(rows, :target_observation_distance), getproperty.(rows, :abs_error))
    row_signal_corr = finite_correlation(getproperty.(row_rows, :row_normalized_observation_signal), getproperty.(row_rows, :row_tv_error))
    row_source_corr = finite_correlation([row.source_observed ? 1.0 : 0.0 for row in row_rows], getproperty.(row_rows, :row_tv_error))
    large_rows = [row for row in rows if row.large_error]
    no_downstream_large = count(row -> !isfinite(row.target_observation_distance), large_rows)
    threshold_source_unobserved = count(row -> !row.source_observed, threshold_rows)
    threshold_target_unobserved = count(row -> !row.target_observed, threshold_rows)
    threshold_neither_observed = count(row -> !row.source_observed && !row.target_observed, threshold_rows)
    top_row_source_unobserved = count(row -> !row.source_observed, top_row_subset)
    top_row_dominant_changed = count(row -> row.dominant_target_changed, top_row_subset)

    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(io, "Square four-to-four turning-outlier structure diagnostic")
        println(io, "======================================================")
        println(io, @sprintf("Regime: %s", dataset.regime.label))
        println(io, @sprintf("Scenarios: %d", dataset.regime.scenario_count))
        println(io, @sprintf("Horizon factor: %d", dataset.regime.horizon_factor))
        println(io, @sprintf("Observation count: %d", dataset_observation_length(dataset)))
        println(io, @sprintf("Observation multiplier: %.2f", observation_multiplier(dataset)))
        println(io, @sprintf("ESMDA ensemble x iterations: %d x %d", esmda.ensemble_size, esmda.esmda_maxiters))
        println(io, @sprintf("Observation clip fraction mean/max: %.6f / %.6f", observations.mean_clip_fraction, observations.max_clip_fraction))
        println(io)

        println(io, "Correlation checks")
        println(io, "------------------")
        println(io, @sprintf("entry corr(abs_error, normalized_sensitivity): %.4f", sensitivity_corr))
        println(io, @sprintf("entry corr(abs_error, direct_observed_count): %.4f", direct_count_corr))
        println(io, @sprintf("entry corr(abs_error, downstream_observation_distance): %.4f", distance_corr))
        println(io, @sprintf("row corr(row_tv_error, row_normalized_observation_signal): %.4f", row_signal_corr))
        println(io, @sprintf("row corr(row_tv_error, source_observed): %.4f", row_source_corr))
        println(io, @sprintf("Top %d row misses with unobserved source roads: %d / %d", top_row_count, top_row_source_unobserved, top_row_count))
        println(io, @sprintf("Top %d row misses where dominant target changed: %d / %d", top_row_count, top_row_dominant_changed, top_row_count))
        println(io)

        println(io, "Terminology")
        println(io, "-----------")
        println(io, "Source road = the incoming road where traffic comes from before the junction.")
        println(io, "Target road = the outgoing road traffic enters after taking that turn.")
        println(io, "target observed means the outgoing road is observed; it does not mean the source road is observed.")
        println(io, "Row TV error = half the sum of absolute errors in one source road's four outgoing fractions.")
        println(io, "Row TV error is the traffic mass that would need to be moved between target roads to fix the row.")
        println(io)

        println(io, "Worst whole-row/source-road split errors")
        println(io, "----------------------------------------")
        for row in top_row_rows
            println(
                io,
                @sprintf(
                    "%-8s TV=%.4f L1=%.4f max_entry=%.4f source=road %d observed=%s observed_targets=[%s] dominant=%d->%d under=%d over=%d row_signal=%.3g pair_under_over=%.3g",
                    row.row_label,
                    row.row_tv_error,
                    row.row_l1_error,
                    row.max_abs_entry_error,
                    row.source_road,
                    string(row.source_observed),
                    row.observed_target_cols,
                    row.dominant_true_col,
                    row.dominant_posterior_col,
                    row.largest_under_col,
                    row.largest_over_col,
                    row.row_normalized_observation_signal,
                    row.under_over_pair_contrast_sensitivity,
                ),
            )
            println(io, @sprintf("  true      [%s]", row.true_row))
            println(io, @sprintf("  posterior [%s]", row.posterior_mean_row))
            println(io, @sprintf("  error     [%s]", row.signed_error_row))
        end
        println(io)

        println(io, @sprintf("Entry-level visual-threshold check: abs_error > %.2f", threshold))
        println(io, "------------------------------------------------")
        println(io, @sprintf("Entries above threshold: %d", length(threshold_rows)))
        println(io, @sprintf("  source road unobserved: %d / %d", threshold_source_unobserved, length(threshold_rows)))
        println(io, @sprintf("  target road unobserved: %d / %d", threshold_target_unobserved, length(threshold_rows)))
        println(io, @sprintf("  neither source nor target observed: %d / %d", threshold_neither_observed, length(threshold_rows)))
        println(io, @sprintf("  no downstream observed road: %d / %d", count(row -> !isfinite(row.target_observation_distance), threshold_rows), length(threshold_rows)))
        if isempty(threshold_rows)
            println(io, "None.")
        else
            for row in threshold_rows
                println(
                    io,
                    @sprintf(
                        "%-8s abs=%.4f truth=%.4f mean=%.4f source=road %d observed=%s target=road %d observed=%s class=%s downstream=%s sens=%.3g",
                        row.entry_label,
                        row.abs_error,
                        row.truth,
                        row.posterior_mean,
                        row.source_road,
                        string(row.source_observed),
                        row.target_road,
                        string(row.target_observed),
                        row.direct_observation_class,
                        distance_string(row.target_observation_distance),
                        row.normalized_sensitivity,
                    ),
                )
            end
        end
        println(io)

        println(io, "Largest individual posterior mean entry errors")
        println(io, "----------------------------------------------")
        for row in top_rows
            println(
                io,
                @sprintf(
                    "%-8s abs=%.4f truth=%.4f mean=%.4f ci90=[%.4f, %.4f] class=%s downstream=%s sens=%.3g",
                    row.entry_label,
                    row.abs_error,
                    row.truth,
                    row.posterior_mean,
                    row.ci05,
                    row.ci95,
                    row.direct_observation_class,
                    distance_string(row.target_observation_distance),
                    row.normalized_sensitivity,
                ),
            )
        end
        println(io)

        println(io, "Entry group summaries")
        println(io, "---------------------")
        for row in entry_group_rows
            println(
                io,
                @sprintf(
                    "%s | %s | n=%d large=%d ci_miss=%d mean_abs=%.4f max_abs=%.4f mean_sens=%.3g",
                    row.group_kind,
                    row.group_label,
                    row.entry_count,
                    row.large_error_count,
                    row.ci_miss_count,
                    row.mean_abs_error,
                    row.max_abs_error,
                    row.mean_normalized_sensitivity,
                ),
            )
        end
        println(io)

        println(io, "Row group summaries")
        println(io, "-------------------")
        for row in row_group_rows
            println(
                io,
                @sprintf(
                    "%s | %s | n=%d dominant_changed=%d source_unobserved=%d mean_TV=%.4f max_TV=%.4f mean_signal=%.3g mean_min_pair=%.3g",
                    row.group_kind,
                    row.group_label,
                    row.row_count,
                    row.dominant_target_changed_count,
                    row.source_unobserved_count,
                    row.mean_row_tv_error,
                    row.max_row_tv_error,
                    row.mean_row_normalized_observation_signal,
                    row.mean_min_pair_contrast_sensitivity,
                ),
            )
        end
    end

    return output_path
end

