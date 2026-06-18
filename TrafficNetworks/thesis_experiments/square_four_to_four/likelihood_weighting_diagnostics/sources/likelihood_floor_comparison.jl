import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using Printf
using Statistics
using Plots

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT"] = "1"
ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_STRUCTURE"] = "1"

include(joinpath(@__DIR__, "data_budget_context.jl"))
include(joinpath(@__DIR__, "turning_recovery_diagnostics.jl"))

const LIKELIHOOD_FLOOR_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "floor_comparison")
const LIKELIHOOD_FLOOR_FIGURE_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const LIKELIHOOD_FLOOR_REGIME = MultiScenarioDataRegime("likelihood_floor_12x", 1, 12)
const LIKELIHOOD_FLOOR_BUDGET = (
    label="reference_192x6",
    ensemble_size=MULTI_SCENARIO_ESMDA_ENSEMBLE_SIZE,
    esmda_iters=MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
)
const LIKELIHOOD_FLOOR_FRACTIONS = [0.03, 0.10, 0.15]
const LIKELIHOOD_FLOOR_SEEDS = [1, 2, 3]

function floor_label(floor_fraction::Real)
    return replace(@sprintf("hard_floor_%.2f", Float64(floor_fraction)), "." => "p")
end

function likelihood_floor_seed_tuple(seed::Int)
    return (
        run_seed=seed,
        observation_seed=DATA_BUDGET_OBSERVATION_SEED_OFFSET + seed,
        esmda_seed=DATA_BUDGET_ESMDA_SEED_OFFSET + seed,
    )
end

function row_swap_summary_metrics(dataset::MultiScenarioDataset, observations, P_post, P_true)
    true_mats = turning_matrices(P_true)
    y_post = simulator_dataset(P_post, dataset)
    post_fit_norm = normalized_rmse_values(y_post, observations.y_obs, observations.sigma_model)
    post_truth_rmse = rmse_values(y_post, observations.y_true)

    truth_improved = 0
    fit_improved = 0
    contradiction_count = 0
    max_truth_improvement = -Inf
    min_fit_improvement = Inf
    worst_label = ""
    worst_truth_improvement = NaN
    worst_fit_improvement = NaN

    for junction in 1:N_JUNCTIONS
        for incoming_row in 1:4
            P_swap = replace_turning_row(P_post, junction, incoming_row, true_mats[junction][incoming_row, :])
            y_swap = simulator_dataset(P_swap, dataset)
            swap_fit_norm = normalized_rmse_values(y_swap, observations.y_obs, observations.sigma_model)
            swap_truth_rmse = rmse_values(y_swap, observations.y_true)
            truth_improvement = post_truth_rmse - swap_truth_rmse
            fit_improvement = post_fit_norm - swap_fit_norm
            improves_truth = truth_improvement > 0.0
            improves_fit = fit_improvement > 0.0

            truth_improved += improves_truth
            fit_improved += improves_fit
            contradiction_count += improves_truth && !improves_fit
            max_truth_improvement = max(max_truth_improvement, truth_improvement)
            min_fit_improvement = min(min_fit_improvement, fit_improvement)

            if improves_truth && !improves_fit && (isnan(worst_truth_improvement) || truth_improvement > worst_truth_improvement)
                worst_label = @sprintf("J%d row %d", junction, incoming_row)
                worst_truth_improvement = truth_improvement
                worst_fit_improvement = fit_improvement
            end
        end
    end

    return (
        row_swap_truth_improved_count=truth_improved,
        row_swap_noisy_fit_improved_count=fit_improved,
        row_swap_contradiction_count=contradiction_count,
        row_swap_max_truth_improvement=max_truth_improvement,
        row_swap_min_fit_improvement=min_fit_improvement,
        row_swap_worst_contradiction_label=worst_label,
        row_swap_worst_contradiction_truth_improvement=worst_truth_improvement,
        row_swap_worst_contradiction_fit_improvement=worst_fit_improvement,
    )
end

function current_floor_noise_summary(dataset::MultiScenarioDataset, observations, floor_fraction::Real)
    label = floor_label(floor_fraction)
    rows = noise_weight_summary_rows(
        observations,
        dataset.setups[1].physical_noise_peak_sigma;
        specs=[(label=label, kind=:hard_floor, floor_fraction=Float64(floor_fraction))],
    )
    return first(rows)
end

function likelihood_floor_metric_row(dataset, observations, esmda, P_true, seed_info, floor_fraction)
    base = final_esmda_metric_row(
        "likelihood_floor",
        "sigma_floor",
        dataset,
        observations,
        esmda,
        LIKELIHOOD_FLOOR_BUDGET,
        P_true,
        seed_info,
    )
    swap = row_swap_summary_metrics(dataset, observations, esmda.P_post_mean, P_true)
    noise = current_floor_noise_summary(dataset, observations, floor_fraction)

    return merge(
        (
            sigma_floor_fraction=Float64(floor_fraction),
            sigma_floor_label=floor_label(floor_fraction),
            sigma_floor_value=noise.floor_value,
        ),
        base,
        (
            sigma_min=noise.sigma_min,
            sigma_median=noise.sigma_median,
            sigma_max=noise.sigma_max,
            at_floor_count=noise.at_floor_count,
            at_floor_fraction=noise.at_floor_fraction,
            max_to_min_weight=noise.max_to_min_weight,
            max_abs_true_standardized=noise.max_abs_true_standardized,
            true_loss_top1_fraction=noise.true_loss_top1_fraction,
            true_loss_top5_fraction=noise.true_loss_top5_fraction,
            true_loss_top10_fraction=noise.true_loss_top10_fraction,
        ),
        swap,
    )
end

function aggregate_likelihood_floor_rows(rows)
    groups = Dict{String, Vector{NamedTuple}}()
    for row in rows
        push!(get!(groups, row.sigma_floor_label, NamedTuple[]), row)
    end

    summary_rows = NamedTuple[]
    for label in sort(collect(keys(groups)); by=label -> parse(Float64, replace(split(label, "_")[end], "p" => ".")))
        group = groups[label]
        push!(
            summary_rows,
            merge(
                (
                    sigma_floor_label=label,
                    sigma_floor_fraction=first(group).sigma_floor_fraction,
                    seed_count=length(group),
                    seeds=join(string.(getproperty.(group, :run_seed)), ","),
                    observation_count=first(group).observation_count,
                    ensemble_size=first(group).ensemble_size,
                    esmda_iters=first(group).esmda_iters,
                    sigma_floor_value=first(group).sigma_floor_value,
                    max_to_min_weight=first(group).max_to_min_weight,
                ),
                range_tuple(:solve_seconds, getproperty.(group, :solve_seconds)),
                range_tuple(:turning_rmse, getproperty.(group, :turning_rmse)),
                range_tuple(:predictive_rmse, getproperty.(group, :predictive_rmse)),
                range_tuple(:final_state_rmse_all, getproperty.(group, :final_state_rmse_all)),
                range_tuple(:final_state_rmse_unobserved, getproperty.(group, :final_state_rmse_unobserved)),
                range_tuple(:normalized_fit_rmse_noisy, getproperty.(group, :normalized_fit_rmse_noisy)),
                range_tuple(:turning_ci90_coverage, getproperty.(group, :turning_ci90_coverage)),
                range_tuple(:turning_ci90_width_mean, getproperty.(group, :turning_ci90_width_mean)),
                range_tuple(:max_abs_true_standardized, getproperty.(group, :max_abs_true_standardized)),
                range_tuple(:true_loss_top1_fraction, getproperty.(group, :true_loss_top1_fraction)),
                range_tuple(:true_loss_top5_fraction, getproperty.(group, :true_loss_top5_fraction)),
                range_tuple(:row_swap_contradiction_count, getproperty.(group, :row_swap_contradiction_count)),
                range_tuple(:row_swap_truth_improved_count, getproperty.(group, :row_swap_truth_improved_count)),
                range_tuple(:row_swap_noisy_fit_improved_count, getproperty.(group, :row_swap_noisy_fit_improved_count)),
            ),
        )
    end

    return summary_rows
end

function plot_likelihood_floor_summary(summary_rows; output_path=nothing)
    ordered = sort(summary_rows; by=row -> row.sigma_floor_fraction)
    x = getproperty.(ordered, :sigma_floor_fraction)
    labels = [@sprintf("%.2f", row.sigma_floor_fraction) for row in ordered]

    specs = [
        (:turning_rmse, "(a) Turning RMSE", "RMSE"),
        (:final_state_rmse_all, "(b) Final-state RMSE", "RMSE"),
        (:true_loss_top1_fraction, "(c) Largest single-observation loss share", "fraction"),
    ]

    plt = plot(
        layout=(3, 1),
        size=(900, 1100),
        dpi=180,
        legend=false,
        left_margin=8Plots.mm,
        bottom_margin=9Plots.mm,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
    )

    for (subplot_id, (prefix, title_text, ylabel_text)) in enumerate(specs)
        mean_vals = Float64.(getproperty.(ordered, Symbol(prefix, :_mean)))
        min_vals = Float64.(getproperty.(ordered, Symbol(prefix, :_min)))
        max_vals = Float64.(getproperty.(ordered, Symbol(prefix, :_max)))
        plot!(
            plt,
            x,
            mean_vals;
            yerror=(mean_vals .- min_vals, max_vals .- mean_vals),
            marker=:circle,
            markersize=5,
            linewidth=2,
            color=:steelblue4,
            xlabel="likelihood-floor fraction",
            ylabel=ylabel_text,
            title=title_text,
            xticks=(x, labels),
            xrotation=0,
            subplot=subplot_id,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function write_likelihood_floor_config(output_path, saved_paths; seeds, floors)
    lines = [
        "Square multi-scenario likelihood floor comparison",
        "============================================",
        @sprintf("regime: %s", LIKELIHOOD_FLOOR_REGIME.label),
        @sprintf("scenario count: %d", LIKELIHOOD_FLOOR_REGIME.scenario_count),
        @sprintf("horizon factor: %d", LIKELIHOOD_FLOOR_REGIME.horizon_factor),
        @sprintf("ESMDA: %d x %d", LIKELIHOOD_FLOOR_BUDGET.ensemble_size, LIKELIHOOD_FLOOR_BUDGET.esmda_iters),
        @sprintf("seeds: %s", join(string.(seeds), ",")),
        @sprintf("floor fractions: %s", join(string.(floors), ",")),
        "",
        "The same observation seed is used for each floor on a given run seed.",
        "The floor changes only sigma_model used in the working likelihood.",
        "",
        "Outputs",
        "-------",
    ]
    append!(lines, string.(saved_paths))
    return write_config_file(output_path, lines)
end

function run_square_multi_scenario_likelihood_floor_comparison(;
    seeds=LIKELIHOOD_FLOOR_SEEDS,
    floors=LIKELIHOOD_FLOOR_FRACTIONS,
    output_dir=LIKELIHOOD_FLOOR_OUTPUT_DIR,
)
    mkpath(output_dir)
    P_true = true_turning_matrices()
    dataset = build_multi_scenario_dataset(LIKELIHOOD_FLOOR_REGIME)
    rows = NamedTuple[]

    println("Square multi-scenario likelihood floor comparison")
    println("--------------------------------------------")
    println(@sprintf("observations: %d (%.1fx)", dataset_observation_length(dataset), observation_multiplier(dataset)))
    println(@sprintf("ESMDA: %d x %d", LIKELIHOOD_FLOOR_BUDGET.ensemble_size, LIKELIHOOD_FLOOR_BUDGET.esmda_iters))
    println(@sprintf("seeds: %s", join(string.(seeds), ",")))
    println(@sprintf("floors: %s", join(string.(floors), ",")))
    println()
    flush(stdout)

    for seed in seeds
        seed_info = likelihood_floor_seed_tuple(seed)
        for floor_fraction in floors
            observations = generate_physical_dataset_observations(
                P_true,
                dataset;
                seed=seed_info.observation_seed,
                floor_fraction=floor_fraction,
            )

            esmda = run_esmda_multi_scenario(
                dataset,
                observations.y_obs,
                observations.sigma_model;
                seed=seed_info.esmda_seed,
                prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
                ensemble_size=LIKELIHOOD_FLOOR_BUDGET.ensemble_size,
                esmda_maxiters=LIKELIHOOD_FLOOR_BUDGET.esmda_iters,
                P_true=P_true,
            )

            row = likelihood_floor_metric_row(dataset, observations, esmda, P_true, seed_info, floor_fraction)
            push!(rows, row)

            println(
                @sprintf(
                    "seed %d floor %.2f | %.1fs | turn %.4f | state %.4f | contradictions %d | top1 %.3f",
                    seed,
                    floor_fraction,
                    row.solve_seconds,
                    row.turning_rmse,
                    row.final_state_rmse_all,
                    row.row_swap_contradiction_count,
                    row.true_loss_top1_fraction,
                ),
            )
            flush(stdout)
        end
    end

    summary_rows = aggregate_likelihood_floor_rows(rows)
    seed_path = write_namedtuple_table(rows, joinpath(output_dir, "likelihood_floor_seed_metrics.tsv"))
    summary_path = write_namedtuple_table(summary_rows, joinpath(output_dir, "likelihood_floor_summary.tsv"))
    figure_path = joinpath(LIKELIHOOD_FLOOR_FIGURE_DIR, "square_likelihood_floor_comparison.png")
    plot_likelihood_floor_summary(summary_rows; output_path=figure_path)
    config_path = write_likelihood_floor_config(
        joinpath(output_dir, "likelihood_floor_config.txt"),
        [seed_path, summary_path, figure_path];
        seeds=seeds,
        floors=floors,
    )

    println()
    println("Outputs")
    println("-------")
    for path in [seed_path, summary_path, figure_path, config_path]
        println(path)
    end

    return (
        rows=rows,
        summary_rows=summary_rows,
        seed_path=seed_path,
        summary_path=summary_path,
        figure_path=figure_path,
        config_path=config_path,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_LIKELIHOOD_FLOOR_COMPARISON", "0") != "1"
    run_square_multi_scenario_likelihood_floor_comparison()
end
