if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Printf
using Statistics
using Plots
using DelimitedFiles

if !isdefined(@__MODULE__, :MultiScenarioDataset)
    include(joinpath(@__DIR__, "..", "multi_scenario", "common.jl"))
end

TURNING_OUTLIER_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "generated", "square_four_to_four", "square_four_to_four_turning_outlier_structure"),
)

include(joinpath(@__DIR__, "metadata.jl"))
include(joinpath(@__DIR__, "sensitivity.jl"))
include(joinpath(@__DIR__, "tables.jl"))
include(joinpath(@__DIR__, "noise.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
include(joinpath(@__DIR__, "plots.jl"))
include(joinpath(@__DIR__, "swaps.jl"))
include(joinpath(@__DIR__, "outputs.jl"))

function run_square_four_to_four_turning_outlier_structure(;
    seed=1,
    observation_seed=seed,
    esmda_seed=seed,
    regime=MultiScenarioDataRegime("structure_1x", 1, 1),
    ensemble_size=MULTI_SCENARIO_ESMDA_ENSEMBLE_SIZE,
    esmda_maxiters=MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
    prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
    peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA,
    floor_fraction=DEFAULT_SIGMA_FLOOR_FRACTION,
    sensitivity_step=1e-3,
    output_dir=TURNING_OUTLIER_OUTPUT_DIR,
)
    mkpath(output_dir)
    P_true = true_turning_matrices()
    dataset = build_multi_scenario_dataset(regime; peak_noise_sigma=peak_noise_sigma)
    observations = generate_physical_dataset_observations(P_true, dataset; seed=observation_seed, floor_fraction=floor_fraction)

    println("Square four-to-four turning-outlier structure diagnostic")
    println("--------------------------------------------------------")
    println(@sprintf("regime: %s", regime.label))
    println(@sprintf("observations: %d (%.2fx)", dataset_observation_length(dataset), observation_multiplier(dataset)))
    println(@sprintf("ESMDA: ensemble=%d iterations=%d", ensemble_size, esmda_maxiters))
    println(@sprintf("likelihood floor fraction: %.3f", floor_fraction))
    println(@sprintf("mean/max clip fraction: %.6f / %.6f", observations.mean_clip_fraction, observations.max_clip_fraction))
    println()

    esmda = run_esmda_multi_scenario(
        dataset,
        observations.y_obs,
        observations.sigma_model;
        seed=esmda_seed,
        prior_scale=prior_scale,
        ensemble_size=ensemble_size,
        esmda_maxiters=esmda_maxiters,
        P_true=P_true,
    )

    rows = turning_entry_diagnostic_rows(
        dataset,
        observations,
        esmda,
        P_true;
        sensitivity_step=sensitivity_step,
    )
    group_rows = turning_entry_group_summary_rows(rows)
    row_rows = turning_row_diagnostic_rows(
        rows,
        dataset,
        observations,
        esmda,
        P_true;
        sensitivity_step=sensitivity_step,
    )
    row_group_rows = turning_row_group_summary_rows(row_rows)
    large_miss_rows = [row for row in sort(rows; by=row -> -row.abs_error) if row.abs_error > 0.30]

    entry_metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "turning_entry_structure_metrics.tsv"))
    group_summary_path = write_namedtuple_table(group_rows, joinpath(output_dir, "turning_entry_structure_group_summary.tsv"))
    row_metrics_path = write_namedtuple_table(row_rows, joinpath(output_dir, "turning_row_structure_metrics.tsv"))
    row_group_summary_path = write_namedtuple_table(row_group_rows, joinpath(output_dir, "turning_row_structure_group_summary.tsv"))
    large_miss_path = write_namedtuple_table(large_miss_rows, joinpath(output_dir, "turning_entry_large_misses_abs_gt_0p30.tsv"))
    summary_path = write_turning_outlier_summary(
        rows,
        group_rows,
        row_rows,
        row_group_rows,
        joinpath(output_dir, "turning_entry_structure_summary.txt"),
        dataset,
        observations,
        esmda,
    )
    swap_outputs = write_turning_row_swap_outputs(
        row_rows,
        dataset,
        observations,
        esmda.P_post_mean,
        P_true,
        output_dir;
        current_floor_fraction=floor_fraction,
    )

    top_row_rows = first(sort(row_rows; by=row -> -row.row_tv_error), min(8, length(row_rows)))
    println("Largest whole-row/source-road split misses")
    println("------------------------------------------")
    for row in top_row_rows
        println(
            @sprintf(
                "%-8s TV %.4f | source road %d observed=%s | observed targets=[%s] | dominant %d -> %d",
                row.row_label,
                row.row_tv_error,
                row.source_road,
                string(row.source_observed),
                row.observed_target_cols,
                row.dominant_true_col,
                row.dominant_posterior_col,
            ),
        )
    end
    println()

    top_rows = first(sort(rows; by=row -> -row.abs_error), min(8, length(rows)))
    println("Largest absolute turning-fraction misses")
    println("----------------------------------------")
    for row in top_rows
        println(
            @sprintf(
                "%-8s abs %.4f | class=%s | downstream=%s | sensitivity %.3g",
                row.entry_label,
                row.abs_error,
                row.direct_observation_class,
                distance_string(row.target_observation_distance),
                row.normalized_sensitivity,
            ),
        )
    end
    println()
    println("Outputs")
    println("-------")
    println(entry_metrics_path)
    println(group_summary_path)
    println(row_metrics_path)
    println(row_group_summary_path)
    println(large_miss_path)
    println(summary_path)
    println(swap_outputs.swap_metrics_path)
    println(swap_outputs.swap_road_metrics_path)
    println(swap_outputs.swap_summary_path)
    println(swap_outputs.noise_summary_path)
    println(swap_outputs.noise_outlier_path)
    println(swap_outputs.variant_path)
    println(swap_outputs.variant_summary_path)
    println(swap_outputs.contribution_path)
    println(swap_outputs.likelihood_summary_path)

    return (
        dataset=dataset,
        observations=observations,
        esmda=esmda,
        rows=rows,
        group_rows=group_rows,
        row_rows=row_rows,
        row_group_rows=row_group_rows,
        large_miss_rows=large_miss_rows,
        entry_metrics_path=entry_metrics_path,
        group_summary_path=group_summary_path,
        row_metrics_path=row_metrics_path,
        row_group_summary_path=row_group_summary_path,
        large_miss_path=large_miss_path,
        summary_path=summary_path,
        swap_outputs=swap_outputs,
    )
end

function run_square_four_to_four_turning_outlier_swap_postprocess(;
    seed=1,
    observation_seed=seed,
    regime=MultiScenarioDataRegime("multi_scenario_12x", 1, 12),
    peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA,
    floor_fraction=DEFAULT_SIGMA_FLOOR_FRACTION,
    output_dir=joinpath(TURNING_OUTLIER_OUTPUT_DIR, "multi_scenario_12x_192x6_seed1"),
)
    row_metrics_path = joinpath(output_dir, "turning_row_structure_metrics.tsv")
    entry_metrics_path = joinpath(output_dir, "turning_entry_structure_metrics.tsv")
    isfile(row_metrics_path) || error("Missing row metrics file: $(row_metrics_path)")
    isfile(entry_metrics_path) || error("Missing entry metrics file: $(entry_metrics_path)")

    P_true = true_turning_matrices()
    dataset = build_multi_scenario_dataset(regime; peak_noise_sigma=peak_noise_sigma)
    observations = generate_physical_dataset_observations(P_true, dataset; seed=observation_seed, floor_fraction=floor_fraction)
    row_rows = read_namedtuple_table(row_metrics_path)
    entry_rows = read_namedtuple_table(entry_metrics_path)
    P_post = posterior_turning_matrices_from_entry_rows(entry_rows)

    outputs = write_turning_row_swap_outputs(
        row_rows,
        dataset,
        observations,
        P_post,
        P_true,
        output_dir;
        current_floor_fraction=floor_fraction,
    )

    println("Turning-row swap-back postprocess complete")
    println("------------------------------------------")
    println(outputs.swap_metrics_path)
    println(outputs.swap_road_metrics_path)
    println(outputs.swap_summary_path)
    println(outputs.noise_summary_path)
    println(outputs.noise_outlier_path)
    println(outputs.variant_path)
    println(outputs.variant_summary_path)
    println(outputs.contribution_path)
    println(outputs.likelihood_summary_path)

    return merge(
        (
            dataset=dataset,
            observations=observations,
            P_post=P_post,
            row_rows=row_rows,
            entry_rows=entry_rows,
        ),
        outputs,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_STRUCTURE", "0") != "1"
    run_square_four_to_four_turning_outlier_structure()
end
