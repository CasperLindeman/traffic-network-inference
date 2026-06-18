if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using Statistics
using Plots
using LaTeXStrings

if !isdefined(@__MODULE__, :MultiScenarioDataset)
    include(joinpath(@__DIR__, "..", "..", "common", "multi_scenario", "common.jl"))
end

DATA_BUDGET_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_DATA_BUDGET_OUTPUT_DIR,
    joinpath(MULTI_SCENARIO_OUTPUT_DIR, "data_budget_experiment"),
)

const DATA_BUDGET_SEEDS = [1, 2, 3, 4, 5]
const DATA_BUDGET_OBSERVATION_SEED_OFFSET = 0
const DATA_BUDGET_ESMDA_SEED_OFFSET = 100_000

const DATA_BUDGET_REFERENCE_BUDGET = (
    label="reference_192x6",
    ensemble_size=192,
    esmda_iters=6,
)

const DATA_BUDGET_HORIZON_REGIMES = [
    (axis="horizon", regime=MultiScenarioDataRegime("horizon_1x", 1, 1)),
    (axis="horizon", regime=MultiScenarioDataRegime("horizon_2x", 2, 1)),
    (axis="horizon", regime=MultiScenarioDataRegime("horizon_4x", 4, 1)),
    (axis="horizon", regime=MultiScenarioDataRegime("horizon_8x", 8, 1)),
]

const DATA_BUDGET_SCENARIO_REGIMES = [
    (axis="scenarios", regime=MultiScenarioDataRegime("scenarios_1x", 1, 1)),
    (axis="scenarios", regime=MultiScenarioDataRegime("scenarios_3x", 1, 3)),
    (axis="scenarios", regime=MultiScenarioDataRegime("scenarios_6x", 1, 6)),
    (axis="scenarios", regime=MultiScenarioDataRegime("scenarios_12x", 1, 12)),
]

const DATA_BUDGET_COMPUTE_REGIMES = [
    (axis="compute", regime=MultiScenarioDataRegime("compute_1x", 1, 1)),
    (axis="compute", regime=MultiScenarioDataRegime("compute_12x_scenarios", 1, 12)),
]

const DATA_BUDGET_COMPUTE_BUDGETS = [
    (label="tiny_64x2", ensemble_size=64, esmda_iters=2),
    (label="small_128x4", ensemble_size=128, esmda_iters=4),
    DATA_BUDGET_REFERENCE_BUDGET,
    (label="more_iterations_192x12", ensemble_size=192, esmda_iters=12),
    (label="larger_ensemble_384x6", ensemble_size=384, esmda_iters=6),
]

final_work_units(budget) = Int(budget.ensemble_size * budget.esmda_iters)

function final_progress(args...)
    println(args...)
    flush(stdout)
end

function final_rmse(x::AbstractVector, y::AbstractVector)
    return sqrt(mean((Float64.(x) .- Float64.(y)) .^ 2))
end

function final_seed_tuple(seed::Int)
    return (
        run_seed=seed,
        observation_seed=DATA_BUDGET_OBSERVATION_SEED_OFFSET + seed,
        esmda_seed=DATA_BUDGET_ESMDA_SEED_OFFSET + seed,
    )
end

function final_observation_design(dataset::MultiScenarioDataset)
    setup = first(dataset.setups)
    return (
        observed_road_count=length(setup.observed_road_ids),
        sensors_per_observed_road=length(setup.sensor_fractions),
        saved_times_per_scenario=length(setup.control_times),
        total_saved_times=sum(length(setup.control_times) for setup in dataset.setups),
    )
end

function turning_interval_metrics(esmda)
    widths = esmda.entry_ci_95 .- esmda.entry_ci_05
    covered = (esmda.entry_ci_05 .<= esmda.entry_true) .& (esmda.entry_true .<= esmda.entry_ci_95)
    return (
        turning_ci90_coverage=mean(covered),
        turning_ci90_width_mean=mean(widths),
        turning_ci90_width_median=quantile(widths, 0.50),
    )
end

function final_esmda_metric_row(
    experiment_group::String,
    data_axis::String,
    dataset::MultiScenarioDataset,
    observations,
    esmda,
    budget,
    P_true,
    seed_info,
)
    base_row = multi_scenario_metric_row(
        dataset,
        :esmda,
        budget.label,
        final_work_units(budget),
        esmda.solve_seconds,
        budget.esmda_iters,
        esmda.P_post_mean,
        esmda.y_post_mean,
        P_true,
        observations.y_true,
        state_metrics=ensemble_mean_state_metrics(esmda, dataset, P_true),
    )

    normalized_fit = sqrt(mean(((esmda.y_post_mean .- observations.y_obs) ./ observations.sigma_model) .^ 2))

    return merge(
        (
            experiment_group=experiment_group,
            data_axis=data_axis,
            run_seed=seed_info.run_seed,
            observation_seed=seed_info.observation_seed,
            esmda_seed=seed_info.esmda_seed,
            compute_label=budget.label,
            ensemble_size=Int(budget.ensemble_size),
            esmda_iters=Int(budget.esmda_iters),
            ensemble_forward_runs=final_work_units(budget),
            mean_clip_fraction=observations.mean_clip_fraction,
            max_clip_fraction=observations.max_clip_fraction,
        ),
        final_observation_design(dataset),
        base_row,
        (
            fit_rmse_noisy=final_rmse(esmda.y_post_mean, observations.y_obs),
            normalized_fit_rmse_noisy=normalized_fit,
        ),
        turning_interval_metrics(esmda),
    )
end

function run_final_esmda_case(
    experiment_group::String,
    data_axis::String,
    regime::MultiScenarioDataRegime,
    budget;
    seeds=DATA_BUDGET_SEEDS,
    peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA,
    prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
)
    P_true = true_turning_matrices()
    dataset = build_multi_scenario_dataset(regime; peak_noise_sigma=peak_noise_sigma)
    rows = NamedTuple[]

    final_progress(
        @sprintf(
            "%-12s %-22s | obs %6d (%5.1fx) | scenarios=%2d horizon=x%d | budget=%s",
            data_axis,
            regime.label,
            dataset_observation_length(dataset),
            observation_multiplier(dataset),
            regime.scenario_count,
            regime.horizon_factor,
            budget.label,
        ),
    )

    for seed in seeds
        seed_info = final_seed_tuple(seed)
        observations = generate_physical_dataset_observations(P_true, dataset; seed=seed_info.observation_seed)

        esmda = run_esmda_multi_scenario(
            dataset,
            observations.y_obs,
            observations.sigma_model;
            seed=seed_info.esmda_seed,
            prior_scale=prior_scale,
            ensemble_size=budget.ensemble_size,
            esmda_maxiters=budget.esmda_iters,
            P_true=P_true,
        )

        row = final_esmda_metric_row(
            experiment_group,
            data_axis,
            dataset,
            observations,
            esmda,
            budget,
            P_true,
            seed_info,
        )
        push!(rows, row)

        final_progress(
            @sprintf(
                "  seed %d | %.1fs | turn %.4f | state %.4f | norm fit %.2f",
                seed,
                row.solve_seconds,
                row.turning_rmse,
                row.final_state_rmse_all,
                row.normalized_fit_rmse_noisy,
            ),
        )
    end

    return rows
end

function range_tuple(prefix::Symbol, values)
    vals = Float64.(values)
    return NamedTuple{
        (
            Symbol(prefix, :_mean),
            Symbol(prefix, :_min),
            Symbol(prefix, :_max),
        )
    }((mean(vals), minimum(vals), maximum(vals)))
end

function aggregate_final_rows(rows)
    groups = Dict{Tuple{String, String, String, String}, Vector{NamedTuple}}()

    for row in rows
        key = (row.experiment_group, row.data_axis, row.regime_label, row.compute_label)
        push!(get!(groups, key, NamedTuple[]), row)
    end

    summary_rows = NamedTuple[]
    for ((experiment_group, data_axis, regime_label, compute_label), group_rows) in sort(
        collect(groups);
        by=entry -> (
            first(entry[2]).experiment_group,
            first(entry[2]).data_axis,
            first(entry[2]).observation_multiplier,
            first(entry[2]).ensemble_forward_runs,
        ),
    )
        push!(
            summary_rows,
            merge(
                (
                    experiment_group=experiment_group,
                    data_axis=data_axis,
                    regime_label=regime_label,
                    compute_label=compute_label,
                    seed_count=length(group_rows),
                    seeds=join(string.(getproperty.(group_rows, :run_seed)), ","),
                    scenario_count=first(group_rows).scenario_count,
                    horizon_factor=first(group_rows).horizon_factor,
                    observation_count=first(group_rows).observation_count,
                    observation_multiplier=first(group_rows).observation_multiplier,
                    observed_road_count=first(group_rows).observed_road_count,
                    sensors_per_observed_road=first(group_rows).sensors_per_observed_road,
                    saved_times_per_scenario=first(group_rows).saved_times_per_scenario,
                    ensemble_size=first(group_rows).ensemble_size,
                    esmda_iters=first(group_rows).esmda_iters,
                    ensemble_forward_runs=first(group_rows).ensemble_forward_runs,
                    mean_clip_fraction_max=maximum(getproperty.(group_rows, :mean_clip_fraction)),
                    max_clip_fraction_max=maximum(getproperty.(group_rows, :max_clip_fraction)),
                ),
                range_tuple(:solve_seconds, getproperty.(group_rows, :solve_seconds)),
                range_tuple(:turning_rmse, getproperty.(group_rows, :turning_rmse)),
                range_tuple(:predictive_rmse, getproperty.(group_rows, :predictive_rmse)),
                range_tuple(:fit_rmse_noisy, getproperty.(group_rows, :fit_rmse_noisy)),
                range_tuple(:normalized_fit_rmse_noisy, getproperty.(group_rows, :normalized_fit_rmse_noisy)),
                range_tuple(:final_state_rmse_all, getproperty.(group_rows, :final_state_rmse_all)),
                range_tuple(:final_state_rmse_observed, getproperty.(group_rows, :final_state_rmse_observed)),
                range_tuple(:final_state_rmse_unobserved, getproperty.(group_rows, :final_state_rmse_unobserved)),
                range_tuple(:turning_ci90_coverage, getproperty.(group_rows, :turning_ci90_coverage)),
                range_tuple(:turning_ci90_width_mean, getproperty.(group_rows, :turning_ci90_width_mean)),
            ),
        )
    end

    return summary_rows
end

function plot_range_series!(plt, rows, x_key::Symbol, y_prefix::Symbol; subplot, label="", kwargs...)
    ordered = sort(rows; by=row -> getproperty(row, x_key))
    x = Float64.(getproperty.(ordered, x_key))
    y_mean = Float64.(getproperty.(ordered, Symbol(y_prefix, :_mean)))
    y_min = Float64.(getproperty.(ordered, Symbol(y_prefix, :_min)))
    y_max = Float64.(getproperty.(ordered, Symbol(y_prefix, :_max)))
    plot!(
        plt,
        x,
        y_mean;
        marker=:circle,
        markersize=7,
        linewidth=2.6,
        yerror=(y_mean .- y_min, y_max .- y_mean),
        markerstrokewidth=1.0,
        label=label,
        subplot=subplot,
        kwargs...,
    )
end

function plot_budget_range_series!(plt, rows, budget_order, y_prefix::Symbol; subplot, label="", x_offset=0.0, kwargs...)
    row_by_budget = Dict(row.compute_label => row for row in rows)
    ordered = [row_by_budget[budget] for budget in budget_order if haskey(row_by_budget, budget)]
    x_ticks = collect(1:length(ordered))
    x = x_ticks .+ x_offset
    labels = replace.(getproperty.(ordered, :compute_label), "_" => " ")
    labels = replace.(labels, "tiny " => "", "small " => "", "reference " => "",
        "more iterations " => "", "larger ensemble " => "")
    y_mean = Float64.(getproperty.(ordered, Symbol(y_prefix, :_mean)))
    y_min = Float64.(getproperty.(ordered, Symbol(y_prefix, :_min)))
    y_max = Float64.(getproperty.(ordered, Symbol(y_prefix, :_max)))
    plot!(
        plt,
        x,
        y_mean;
        marker=:circle,
        markersize=7,
        linewidth=2.6,
        yerror=(y_mean .- y_min, y_max .- y_mean),
        markerstrokewidth=1.0,
        label=label,
        subplot=subplot,
        xticks=(x_ticks, labels),
        kwargs...,
    )
end

function plot_final_data_amount_summary(summary_rows; output_path=nothing)
    data_rows = filter(row -> row.experiment_group == "data_amount", summary_rows)
    axes = ["horizon", "scenarios"]
    titles = Dict(
        "horizon" => "More time, one scenario",
        "scenarios" => "More IC/BC scenarios, short horizon",
    )

    plt = plot(
        layout=(3, 1),
        size=(1200, 1600),
        dpi=180,
        left_margin=12Plots.mm,
        bottom_margin=10Plots.mm,
        top_margin=5Plots.mm,
        legend=:topright,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
    )

    for axis in axes
        rows = filter(row -> row.data_axis == axis, data_rows)
        isempty(rows) && continue
        label = titles[axis]

        plot_range_series!(
            plt,
            rows,
            :observation_multiplier,
            :turning_rmse;
            subplot=1,
            label=label,
            xlabel="Observation multiplier",
            ylabel=L"\operatorname{RMSE}_{\theta}",
            title="(a) Turning recovery",
        )

        plot_range_series!(
            plt,
            rows,
            :observation_multiplier,
            :final_state_rmse_all;
            subplot=2,
            label="",
            xlabel="Observation multiplier",
            ylabel=L"\operatorname{RMSE}_{\mathrm{state}}",
            title="(b) Final-state recovery",
        )

        plot_range_series!(
            plt,
            rows,
            :observation_multiplier,
            :normalized_fit_rmse_noisy;
            subplot=3,
            label="",
            xlabel="Observation multiplier",
            ylabel=L"\operatorname{RMSE}_{\mathrm{fit},\sigma}",
            title="(c) Fit to noisy observations",
        )
    end

    hline!(plt, [1.0]; subplot=3, linestyle=:dash, color=:gray45, label="")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_final_compute_cost_summary(summary_rows; output_path=nothing)
    compute_rows = filter(row -> row.experiment_group == "compute_cost", summary_rows)
    regimes = unique(getproperty.(compute_rows, :regime_label))
    regime_labels = Dict(
        "compute_1x" => "1x data",
        "compute_12x_scenarios" => "12x scenarios",
    )
    budget_order = [
        "tiny_64x2",
        "small_128x4",
        "reference_192x6",
        "more_iterations_192x12",
        "larger_ensemble_384x6",
    ]

    plt = plot(
        layout=(3, 1),
        size=(1200, 1600),
        dpi=180,
        left_margin=12Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=5Plots.mm,
        legend=:topright,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
    )

    for regime_label in regimes
        rows = filter(row -> row.regime_label == regime_label, compute_rows)
        isempty(rows) && continue
        label = get(regime_labels, regime_label, regime_label)
        x_offset = regime_label == "compute_1x" ? -0.06 : 0.06

        plot_budget_range_series!(
            plt,
            rows,
            budget_order,
            :turning_rmse;
            subplot=1,
            label=label,
            xlabel="ESMDA budget",
            ylabel=L"\operatorname{RMSE}_{\theta}",
            title="(a) Turning recovery",
            x_offset=x_offset,
        )

        plot_budget_range_series!(
            plt,
            rows,
            budget_order,
            :final_state_rmse_all;
            subplot=2,
            label="",
            xlabel="ESMDA budget",
            ylabel=L"\operatorname{RMSE}_{\mathrm{state}}",
            title="(b) Final-state recovery",
            x_offset=x_offset,
        )

        plot_budget_range_series!(
            plt,
            rows,
            budget_order,
            :turning_ci90_width_mean;
            subplot=3,
            label="",
            xlabel="ESMDA budget",
            ylabel="Mean width",
            title="(c) Turning 90% interval width",
            x_offset=x_offset,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function write_data_budget_experiment_config(output_dir, paths; seeds=DATA_BUDGET_SEEDS)
    base_setup = square_single_scenario_setup()
    lines = [
        "Square network data amount and ESMDA budget experiment",
        "======================================================",
        "",
        "Purpose:",
        "1. Separate the effect of data amount into longer horizons and more independent IC/BC scenarios.",
        "2. Separate the effect of ESMDA compute into ensemble size and assimilation iterations.",
        "",
        "Reporting convention:",
        @sprintf("Seeds: %s", join(string.(seeds), ", ")),
        "Summary tables report mean, min, and max across seeds. Standard deviations are deliberately omitted.",
        "",
        "Fixed observation design:",
        @sprintf("Observed roads: %d", length(base_setup.observed_road_ids)),
        @sprintf("Sensors per observed road: %d", length(base_setup.sensor_fractions)),
        @sprintf("Base saved times per scenario: %d", length(base_setup.control_times)),
        @sprintf("Base observation count: %d", observation_length(base_setup)),
        "",
        "Fixed numerical setup:",
        @sprintf("Physical peak noise sigma: %.4f", MULTI_SCENARIO_PEAK_NOISE_SIGMA),
        @sprintf("Prior scale: %.2f", MULTI_SCENARIO_PRIOR_SCALE),
        @sprintf("Shared base length: %.3f km", base_setup.base_length_km),
        @sprintf("Uniform dx: %.1f m", uniform_physical_dx_meters(base_setup)),
        "",
        "Reference ESMDA budget:",
        @sprintf(
            "%s: ensemble=%d, iterations=%d, forward runs=%d",
            DATA_BUDGET_REFERENCE_BUDGET.label,
            DATA_BUDGET_REFERENCE_BUDGET.ensemble_size,
            DATA_BUDGET_REFERENCE_BUDGET.esmda_iters,
            final_work_units(DATA_BUDGET_REFERENCE_BUDGET),
        ),
        "",
        "Saved outputs:",
    ]
    append!(lines, paths)
    return write_config_file(joinpath(output_dir, "data_budget_experiment_config.txt"), lines)
end

function run_square_data_budget_experiment(;
    seeds=DATA_BUDGET_SEEDS,
    output_dir=DATA_BUDGET_OUTPUT_DIR,
    run_data_amount=true,
    run_compute_cost=true,
)
    mkpath(output_dir)

    data_seed_rows = NamedTuple[]
    compute_seed_rows = NamedTuple[]

    final_progress("Square network data amount and ESMDA budget experiment")
    final_progress("====================================")
    final_progress("Seeds: ", join(string.(seeds), ", "))
    final_progress()

    if run_data_amount
        final_progress("Data amount experiment")
        final_progress("----------------------")
        for spec in vcat(DATA_BUDGET_HORIZON_REGIMES, DATA_BUDGET_SCENARIO_REGIMES)
            append!(
                data_seed_rows,
                run_final_esmda_case(
                    "data_amount",
                    spec.axis,
                    spec.regime,
                    DATA_BUDGET_REFERENCE_BUDGET;
                    seeds=seeds,
                ),
            )
        end
        final_progress()
    end

    if run_compute_cost
        final_progress("Compute cost experiment")
        final_progress("-----------------------")
        for spec in DATA_BUDGET_COMPUTE_REGIMES
            for budget in DATA_BUDGET_COMPUTE_BUDGETS
                append!(
                    compute_seed_rows,
                    run_final_esmda_case(
                        "compute_cost",
                        spec.axis,
                        spec.regime,
                        budget;
                        seeds=seeds,
                    ),
                )
            end
        end
        final_progress()
    end

    seed_rows = vcat(data_seed_rows, compute_seed_rows)
    summary_rows = aggregate_final_rows(seed_rows)

    data_seed_path = write_namedtuple_table(data_seed_rows, joinpath(output_dir, "data_budget_data_amount_seed_metrics.tsv"))
    compute_seed_path = write_namedtuple_table(compute_seed_rows, joinpath(output_dir, "data_budget_compute_cost_seed_metrics.tsv"))
    all_seed_path = write_namedtuple_table(seed_rows, joinpath(output_dir, "data_budget_all_seed_metrics.tsv"))
    summary_path = write_namedtuple_table(summary_rows, joinpath(output_dir, "data_budget_summary.tsv"))

    data_plot_path = joinpath(output_dir, "data_budget_data_amount_summary.png")
    compute_plot_path = joinpath(output_dir, "data_budget_compute_cost_summary.png")

    run_data_amount && plot_final_data_amount_summary(summary_rows; output_path=data_plot_path)
    run_compute_cost && plot_final_compute_cost_summary(summary_rows; output_path=compute_plot_path)

    saved_paths = filter(
        path -> path !== nothing,
        [
            data_seed_path,
            compute_seed_path,
            all_seed_path,
            summary_path,
            run_data_amount ? data_plot_path : nothing,
            run_compute_cost ? compute_plot_path : nothing,
        ],
    )
    config_path = write_data_budget_experiment_config(output_dir, saved_paths; seeds=seeds)

    final_progress("Outputs")
    final_progress("-------")
    for path in vcat(saved_paths, [config_path])
        final_progress(path)
    end

    return (
        data_seed_rows=data_seed_rows,
        compute_seed_rows=compute_seed_rows,
        seed_rows=seed_rows,
        summary_rows=summary_rows,
        data_seed_path=data_seed_path,
        compute_seed_path=compute_seed_path,
        all_seed_path=all_seed_path,
        summary_path=summary_path,
        data_plot_path=data_plot_path,
        compute_plot_path=compute_plot_path,
        config_path=config_path,
    )
end

function data_budget_env_flag(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    return raw in ("1", "true", "yes", "on")
end

function data_budget_env_seeds(default_seeds=DATA_BUDGET_SEEDS)
    raw = strip(get(ENV, "TRAFFICNETWORKS_DATA_BUDGET_SEEDS", ""))
    isempty(raw) && return default_seeds
    return [parse(Int, strip(part)) for part in split(raw, ",") if !isempty(strip(part))]
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT", "0") != "1"
    data_budget_experiment_results = run_square_data_budget_experiment(
        seeds=data_budget_env_seeds(),
        run_data_amount=data_budget_env_flag("TRAFFICNETWORKS_DATA_BUDGET_RUN_DATA_AMOUNT", true),
        run_compute_cost=data_budget_env_flag("TRAFFICNETWORKS_DATA_BUDGET_RUN_COMPUTE_COST", true),
    )
end
