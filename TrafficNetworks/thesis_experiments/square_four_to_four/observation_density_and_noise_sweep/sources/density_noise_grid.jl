include(joinpath(@__DIR__, "density_noise_helpers.jl"))

using Optim

const DENSITY_NOISE_LBFGS_SECONDS = 300.0
const DENSITY_NOISE_LBFGS_OUTPUT_DIR = joinpath(
    @__DIR__,
    "..",
    "outputs",
)
const DENSITY_NOISE_LBFGS_FIGURE_DIR = joinpath(
    @__DIR__,
    "..",
    "figures",
)

function apply_density_noise_thesis_plot_style!()
    default(tickfontsize=13, guidefontsize=15, legendfontsize=12, titlefontsize=18)
end

function run_lbfgs_map_multi_scenario_timed(
    y_obs::AbstractVector,
    dataset::MultiScenarioDataset,
    sigma_model::AbstractVector;
    prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
    z0=zeros(N_PARAMS),
    time_limit_seconds=DENSITY_NOISE_LBFGS_SECONDS,
    maxiters=10_000,
)
    loss_fn = z -> map_loss_dataset_weighted_forwarddiff(z, y_obs, dataset, sigma_model; prior_scale=prior_scale)

    function fg!(F, G, z)
        if G !== nothing
            ForwardDiff.gradient!(G, loss_fn, z)
        end
        if F !== nothing
            return loss_fn(z)
        end
        return nothing
    end

    result = Optim.optimize(
        Optim.only_fg!(fg!),
        Float64.(collect(z0)),
        Optim.LBFGS(),
        Optim.Options(
            iterations=maxiters,
            store_trace=true,
            show_trace=false,
            time_limit=time_limit_seconds,
        ),
    )

    z_best = Optim.minimizer(result)
    return (
        z=copy(z_best),
        z_best=copy(z_best),
        P_est=turning_matrices(z_best),
        y_est=simulator_dataset(z_best, dataset),
        best_loss=Optim.minimum(result),
        final_loss=loss_fn(z_best),
        losses=[tr.value for tr in Optim.trace(result) if tr.value !== nothing],
        solve_seconds=Optim.time_run(result),
        iterations=Optim.iterations(result),
        converged=Optim.converged(result),
        result=result,
    )
end

function lbfgs_metric_row(density, noise_sigma, seed, dataset, observations, lbfgs, P_true)
    base = base_metric_row(density, noise_sigma, seed, "map_lbfgs", dataset, observations, lbfgs.P_est, lbfgs.y_est, P_true)

    return (
        base...,
        requested_budget=DENSITY_NOISE_LBFGS_SECONDS,
        solve_seconds=lbfgs.solve_seconds,
        iterations=lbfgs.iterations,
        ensemble_size=0,
        esmda_maxiters=0,
        adam_time_limit_seconds=NaN,
        adam_best_loss=lbfgs.best_loss,
        adam_final_loss=lbfgs.final_loss,
        adam_best_iter=lbfgs.iterations,
        adam_final_raw_grad_norm=NaN,
        adam_loss_tail_relspan=tail_relative_span(lbfgs.losses),
        adam_grad_tail_relspan=NaN,
        turning_interval_width_mean=NaN,
        turning_interval_width_max=NaN,
        turning_interval_coverage=NaN,
    )
end

function seed_esmda_rows_from_existing!(metric_rows, entry_rows, diagnostic_rows; output_dir=DENSITY_NOISE_LBFGS_OUTPUT_DIR)
    old_metrics_path = joinpath(DENSITY_NOISE_FULL_OUTPUT_DIR, "full_fit_metrics.tsv")
    old_entries_path = joinpath(DENSITY_NOISE_FULL_OUTPUT_DIR, "full_turning_entry_estimates.tsv")
    old_diagnostics_path = joinpath(DENSITY_NOISE_FULL_OUTPUT_DIR, "full_esmda_diagnostics.tsv")

    if isempty(metric_rows) && isfile(old_metrics_path)
        append!(metric_rows, [row for row in existing_rows(old_metrics_path) if parse_table_string(row.method) == "esmda"])
        write_namedtuple_table(metric_rows, joinpath(output_dir, "full_fit_metrics.tsv"))
    end

    if isempty(entry_rows) && isfile(old_entries_path)
        append!(entry_rows, [row for row in existing_rows(old_entries_path) if parse_table_string(row.method) == "esmda"])
        write_namedtuple_table(entry_rows, joinpath(output_dir, "full_turning_entry_estimates.tsv"))
    end

    if isempty(diagnostic_rows) && isfile(old_diagnostics_path)
        append!(diagnostic_rows, existing_rows(old_diagnostics_path))
        write_namedtuple_table(diagnostic_rows, joinpath(output_dir, "full_esmda_diagnostics.tsv"))
    end

    return metric_rows, entry_rows, diagnostic_rows
end

function write_lbfgs_summary(rows; output_dir=DENSITY_NOISE_LBFGS_OUTPUT_DIR, filename="full_summary.tsv")
    isempty(rows) && return NamedTuple[]
    summary_rows = NamedTuple[]

    for density in ["dense", "sparse", "minimal"]
        for noise_sigma in DENSITY_NOISE_FULL_LEVELS
            for method in ["esmda", "map_lbfgs"]
                group = [
                    row for row in rows
                    if parse_table_string(row.density) == density &&
                       parse_table_string(row.method) == method &&
                       isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
                ]
                isempty(group) && continue
                turning = parse_table_float_or_nan.(getproperty.(group, :turning_rmse))
                predictive = parse_table_float_or_nan.(getproperty.(group, :predictive_rmse))
                final_unobs = parse_table_float_or_nan.(getproperty.(group, :final_state_rmse_unobserved))
                solve_seconds = parse_table_float_or_nan.(getproperty.(group, :solve_seconds))
                push!(
                    summary_rows,
                    (
                        density=density,
                        noise_sigma=noise_sigma,
                        method=method,
                        seed_count=length(group),
                        turning_rmse_mean=mean(turning),
                        turning_rmse_min=minimum(turning),
                        turning_rmse_max=maximum(turning),
                        predictive_rmse_mean=mean(predictive),
                        predictive_rmse_min=minimum(predictive),
                        predictive_rmse_max=maximum(predictive),
                        final_state_rmse_unobserved_mean=mean(final_unobs),
                        final_state_rmse_unobserved_min=minimum(final_unobs),
                        final_state_rmse_unobserved_max=maximum(final_unobs),
                        solve_seconds_mean=mean(solve_seconds),
                    ),
                )
            end
        end
    end

    write_namedtuple_table(summary_rows, joinpath(output_dir, filename))
    return summary_rows
end

function add_lbfgs_mean_range_series!(
    plt,
    subplot_id::Int,
    x_positions::Vector{Float64},
    means::Vector{Float64},
    mins::Vector{Float64},
    maxs::Vector{Float64};
    label::String,
    color,
    linestyle=:solid,
)
    return add_mean_range_series!(
        plt,
        subplot_id,
        x_positions,
        means,
        mins,
        maxs;
        label=label,
        color=color,
        linestyle=linestyle,
    )
end

function common_metric_ylims(summary_rows, metric_prefix::String)
    mins = Float64[]
    maxs = Float64[]

    for row in summary_rows
        push!(mins, parse_table_float_or_nan(getproperty(row, Symbol(metric_prefix * "_min"))))
        push!(maxs, parse_table_float_or_nan(getproperty(row, Symbol(metric_prefix * "_max"))))
    end

    finite_mins = [value for value in mins if isfinite(value)]
    finite_maxs = [value for value in maxs if isfinite(value)]
    if isempty(finite_mins) || isempty(finite_maxs)
        return nothing
    end

    lower = minimum(finite_mins)
    upper = maximum(finite_maxs)
    span = upper > lower ? upper - lower : max(abs(upper), 1.0)
    pad = 0.08 * span
    return (max(0.0, lower - pad), upper + pad)
end

function write_full_metric_plot_lbfgs(
    summary_rows,
    metric_prefix::String,
    ylabel::String,
    output_path::String;
    plot_title::String,
)
    densities = ["dense", "sparse", "minimal"]
    methods = [
        (method="esmda", label="ESMDA", color=:steelblue, offset=-0.06, linestyle=:solid),
        (method="map_lbfgs", label="MAP (L-BFGS)", color=:darkorange, offset=0.06, linestyle=:dash),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]
    ylims = common_metric_ylims(summary_rows, metric_prefix)

    plt = plot(
        layout=(1, 3),
        size=(1200, 420),
        legend=:topleft,
        bottom_margin=9Plots.mm,
        left_margin=8Plots.mm,
    )

    for (subplot_id, density) in enumerate(densities)
        plot!(
            plt;
            subplot=subplot_id,
            title=uppercasefirst(density),
            xlabel="Peak noise scale",
            ylabel=subplot_id == 1 ? ylabel : "",
            xticks=(x_base, x_labels),
            xlims=(0.6, length(noise_levels) + 0.4),
            ylims=ylims,
            grid=true,
            framestyle=:box,
        )

        for method_spec in methods
            means = Float64[]
            mins = Float64[]
            maxs = Float64[]

            for noise_sigma in noise_levels
                group = [
                    row for row in summary_rows
                    if parse_table_string(row.density) == density &&
                       parse_table_string(row.method) == method_spec.method &&
                       isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
                ]
                isempty(group) && continue
                row = only(group)
                push!(means, parse_table_float_or_nan(getproperty(row, Symbol(metric_prefix * "_mean"))))
                push!(mins, parse_table_float_or_nan(getproperty(row, Symbol(metric_prefix * "_min"))))
                push!(maxs, parse_table_float_or_nan(getproperty(row, Symbol(metric_prefix * "_max"))))
            end

            x_positions = Float64.(x_base[1:length(means)]) .+ method_spec.offset
            add_lbfgs_mean_range_series!(
                plt,
                subplot_id,
                x_positions,
                means,
                mins,
                maxs;
                label=subplot_id == 1 ? method_spec.label : "",
                color=method_spec.color,
                linestyle=method_spec.linestyle,
            )
        end
    end

    if !isempty(plot_title)
        plot!(plt; plot_title=plot_title)
    end
    savefig(plt, output_path)
    return output_path
end

function write_full_turning_headline_plot_lbfgs(summary_rows, output_path::String)
    density_specs_plot = [
        (density="dense", label="dense", color=:steelblue),
        (density="sparse", label="sparse", color=:darkorange),
        (density="minimal", label="minimal", color=:seagreen),
    ]
    method_specs = [
        (method="esmda", label="ESMDA", linestyle=:solid, offset=-0.035),
        (method="map_lbfgs", label="MAP (L-BFGS)", linestyle=:dash, offset=0.035),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]

    plt = plot(
        size=(780, 460),
        xlabel="Peak noise scale",
        ylabel="Turning-fraction RMSE",
        xticks=(x_base, x_labels),
        xlims=(0.65, length(noise_levels) + 0.35),
        legend=:topleft,
        grid=true,
        framestyle=:box,
        left_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
    )

    for density_spec in density_specs_plot
        for method_spec in method_specs
            means = Float64[]
            mins = Float64[]
            maxs = Float64[]

            for noise_sigma in noise_levels
                group = [
                    row for row in summary_rows
                    if parse_table_string(row.density) == density_spec.density &&
                       parse_table_string(row.method) == method_spec.method &&
                       isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
                ]
                isempty(group) && continue
                row = only(group)
                push!(means, parse_table_float_or_nan(row.turning_rmse_mean))
                push!(mins, parse_table_float_or_nan(row.turning_rmse_min))
                push!(maxs, parse_table_float_or_nan(row.turning_rmse_max))
            end

            x_positions = Float64.(x_base[1:length(means)]) .+ method_spec.offset
            add_lbfgs_mean_range_series!(
                plt,
                1,
                x_positions,
                means,
                mins,
                maxs;
                label="$(density_spec.label), $(method_spec.label)",
                color=density_spec.color,
                linestyle=method_spec.linestyle,
            )
        end
    end

    savefig(plt, output_path)
    return output_path
end

function write_full_prediction_rmse_panels_lbfgs(summary_rows, output_path::String)
    densities = ["dense", "sparse", "minimal"]
    metric_specs = [
        (prefix="predictive_rmse", ylabel="Predictive RMSE"),
        (prefix="final_state_rmse_unobserved", ylabel="Unobserved final-state RMSE"),
    ]
    method_specs = [
        (method="esmda", label="ESMDA", color=:steelblue, offset=-0.06, linestyle=:solid),
        (method="map_lbfgs", label="MAP (L-BFGS)", color=:darkorange, offset=0.06, linestyle=:dash),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]
    metric_ylims = [common_metric_ylims(summary_rows, spec.prefix) for spec in metric_specs]

    plt = plot(
        layout=(2, 3),
        size=(1200, 720),
        legend=:topleft,
        bottom_margin=9Plots.mm,
        left_margin=9Plots.mm,
    )

    for (metric_id, metric_spec) in enumerate(metric_specs)
        for (density_id, density) in enumerate(densities)
            subplot_id = (metric_id - 1) * length(densities) + density_id
            plot!(
                plt;
                subplot=subplot_id,
                title=metric_id == 1 ? uppercasefirst(density) : "",
                xlabel=metric_id == length(metric_specs) ? "Peak noise scale" : "",
                ylabel=density_id == 1 ? metric_spec.ylabel : "",
                xticks=(x_base, x_labels),
                xlims=(0.6, length(noise_levels) + 0.4),
                ylims=metric_ylims[metric_id],
                grid=true,
                framestyle=:box,
            )

            for method_spec in method_specs
                means = Float64[]
                mins = Float64[]
                maxs = Float64[]

                for noise_sigma in noise_levels
                    group = [
                        row for row in summary_rows
                        if parse_table_string(row.density) == density &&
                           parse_table_string(row.method) == method_spec.method &&
                           isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
                    ]
                    isempty(group) && continue
                    row = only(group)
                    push!(means, parse_table_float_or_nan(getproperty(row, Symbol(metric_spec.prefix * "_mean"))))
                    push!(mins, parse_table_float_or_nan(getproperty(row, Symbol(metric_spec.prefix * "_min"))))
                    push!(maxs, parse_table_float_or_nan(getproperty(row, Symbol(metric_spec.prefix * "_max"))))
                end

                x_positions = Float64.(x_base[1:length(means)]) .+ method_spec.offset
                add_lbfgs_mean_range_series!(
                    plt,
                    subplot_id,
                    x_positions,
                    means,
                    mins,
                    maxs;
                    label=subplot_id == 1 ? method_spec.label : "",
                    color=method_spec.color,
                    linestyle=method_spec.linestyle,
                )
            end
        end
    end

    savefig(plt, output_path)
    return output_path
end

function write_full_rmse_diagnostic_panels_lbfgs(summary_rows, output_path::String)
    densities = ["dense", "sparse", "minimal"]
    metric_specs = [
        (prefix="turning_rmse", ylabel="Turning-fraction RMSE"),
        (prefix="predictive_rmse", ylabel="Predictive RMSE"),
        (prefix="final_state_rmse_unobserved", ylabel="Unobserved final-state RMSE"),
    ]
    method_specs = [
        (method="esmda", label="ESMDA", color=:steelblue, offset=-0.06, linestyle=:solid),
        (method="map_lbfgs", label="MAP (L-BFGS)", color=:darkorange, offset=0.06, linestyle=:dash),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]
    metric_ylims = [common_metric_ylims(summary_rows, spec.prefix) for spec in metric_specs]

    plt = plot(
        layout=(length(metric_specs), length(densities)),
        size=(1250, 1020),
        legend=:topleft,
        bottom_margin=9Plots.mm,
        left_margin=9Plots.mm,
        right_margin=3Plots.mm,
        top_margin=5Plots.mm,
    )

    for (metric_id, metric_spec) in enumerate(metric_specs)
        for (density_id, density) in enumerate(densities)
            subplot_id = (metric_id - 1) * length(densities) + density_id
            plot!(
                plt;
                subplot=subplot_id,
                title=metric_id == 1 ? uppercasefirst(density) : "",
                xlabel=metric_id == length(metric_specs) ? "Peak noise scale" : "",
                ylabel=density_id == 1 ? metric_spec.ylabel : "",
                xticks=(x_base, x_labels),
                xlims=(0.6, length(noise_levels) + 0.4),
                ylims=metric_ylims[metric_id],
                grid=true,
                framestyle=:box,
            )

            for method_spec in method_specs
                means = Float64[]
                mins = Float64[]
                maxs = Float64[]

                for noise_sigma in noise_levels
                    group = [
                        row for row in summary_rows
                        if parse_table_string(row.density) == density &&
                           parse_table_string(row.method) == method_spec.method &&
                           isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
                    ]
                    isempty(group) && continue
                    row = only(group)
                    push!(means, parse_table_float_or_nan(getproperty(row, Symbol(metric_spec.prefix * "_mean"))))
                    push!(mins, parse_table_float_or_nan(getproperty(row, Symbol(metric_spec.prefix * "_min"))))
                    push!(maxs, parse_table_float_or_nan(getproperty(row, Symbol(metric_spec.prefix * "_max"))))
                end

                x_positions = Float64.(x_base[1:length(means)]) .+ method_spec.offset
                add_lbfgs_mean_range_series!(
                    plt,
                    subplot_id,
                    x_positions,
                    means,
                    mins,
                    maxs;
                    label=subplot_id == 1 ? method_spec.label : "",
                    color=method_spec.color,
                    linestyle=method_spec.linestyle,
                )
            end
        end
    end

    savefig(plt, output_path)
    return output_path
end

function run_density_noise_full_lbfgs_plots(;
    output_dir=DENSITY_NOISE_LBFGS_OUTPUT_DIR,
    figure_dir=DENSITY_NOISE_LBFGS_FIGURE_DIR,
)
    apply_density_noise_thesis_plot_style!()
    summary_rows = read_namedtuple_table(joinpath(output_dir, "full_summary.tsv"))
    figure_path = joinpath(figure_dir, "square_density_noise_rmse_diagnostics.png")
    mkpath(dirname(figure_path))
    return [write_full_rmse_diagnostic_panels_lbfgs(summary_rows, figure_path)]
end

function run_density_noise_full_lbfgs(; output_dir=DENSITY_NOISE_LBFGS_OUTPUT_DIR)
    mkpath(output_dir)
    metrics_path = joinpath(output_dir, "full_fit_metrics.tsv")
    entries_path = joinpath(output_dir, "full_turning_entry_estimates.tsv")
    diagnostics_path = joinpath(output_dir, "full_esmda_diagnostics.tsv")
    metric_rows = existing_rows(metrics_path)
    entry_rows = existing_rows(entries_path)
    diagnostic_rows = existing_rows(diagnostics_path)

    density_sensor_layout_rows(; output_dir=output_dir, filename="full_sensor_layouts.tsv")
    write_config_file(
        joinpath(output_dir, "full_config.txt"),
        [
            "experiment = joint observation density x noise full grid with L-BFGS MAP rows",
            "source = sources/density_noise_grid.jl",
            "regime = $(DENSITY_NOISE_REGIME.label)",
            "scenario_count = $(DENSITY_NOISE_REGIME.scenario_count)",
            "noise_levels = $(join(DENSITY_NOISE_FULL_LEVELS, ","))",
            "noise_seeds = $(join(DENSITY_NOISE_FULL_SEEDS, ","))",
            "floor_fraction = $(DENSITY_NOISE_FLOOR_FRACTION)",
            "densities = dense,sparse,minimal",
            "dense = 9 roads, 4 sensors per road",
            "sparse = 9 roads, 1 midpoint sensor per road",
            "minimal = roads $(join(MINIMAL_INTERNAL_ROADS, ",")), 1 midpoint sensor per road",
            "esmda = recomputed when missing, reused from the current output table when available",
            "map_optimizer = Optim.jl L-BFGS",
            "map_time_limit_seconds = $(DENSITY_NOISE_LBFGS_SECONDS)",
            "prior_scale = $(MULTI_SCENARIO_PRIOR_SCALE)",
            "note = thesis grid contains ESMDA and MAP/L-BFGS rows only",
        ],
    )

    seed_esmda_rows_from_existing!(metric_rows, entry_rows, diagnostic_rows; output_dir=output_dir)

    P_true = true_turning_matrices()
    println("Density-noise full grid with L-BFGS MAP rows")
    println("---------------------------------------------")
    println("Warming up L-BFGS compilation")
    warmup_dataset = dataset_for_density(first(density_specs()), first(DENSITY_NOISE_FULL_LEVELS))
    warmup_observations = generate_physical_dataset_observations(P_true, warmup_dataset; seed=999, floor_fraction=DENSITY_NOISE_FLOOR_FRACTION)
    run_lbfgs_map_multi_scenario_timed(
        warmup_observations.y_obs,
        warmup_dataset,
        warmup_observations.sigma_model;
        time_limit_seconds=2.0,
    )

    total_cases = length(density_specs()) * length(DENSITY_NOISE_FULL_LEVELS) * length(DENSITY_NOISE_FULL_SEEDS)
    case_idx = 0

    for spec in density_specs()
        for noise_sigma in DENSITY_NOISE_FULL_LEVELS
            dataset = dataset_for_density(spec, noise_sigma)
            for noise_seed in DENSITY_NOISE_FULL_SEEDS
                case_idx += 1
                observations = generate_physical_dataset_observations(P_true, dataset; seed=noise_seed, floor_fraction=DENSITY_NOISE_FLOOR_FRACTION)
                @printf(
                    "\ncase %d/%d density=%s noise=%.3f seed=%d obs=%d\n",
                    case_idx,
                    total_cases,
                    spec.density,
                    noise_sigma,
                    noise_seed,
                    dataset_observation_length(dataset),
                )
                flush(stdout)

                esmda_metrics_done = fit_completed(metric_rows, spec.density, noise_sigma, noise_seed, "esmda")
                esmda_diag_done = esmda_diagnostic_completed(diagnostic_rows, spec.density, noise_sigma, noise_seed)
                if !(esmda_metrics_done && esmda_diag_done)
                    @printf("running ESMDA density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                    esmda = run_esmda_multi_scenario(
                        dataset,
                        observations.y_obs,
                        observations.sigma_model;
                        seed=1,
                        ensemble_size=DENSITY_NOISE_ENSEMBLE_SIZE,
                        esmda_maxiters=DENSITY_NOISE_ESMDA_MAXITERS,
                        P_true=P_true,
                    )

                    if !esmda_metrics_done
                        push!(metric_rows, esmda_metric_row(spec.density, noise_sigma, noise_seed, dataset, observations, esmda, P_true))
                        append!(entry_rows, turning_entry_rows_for_fit(spec.density, noise_sigma, noise_seed, "esmda", esmda.P_post_mean, P_true))
                        write_namedtuple_table(metric_rows, metrics_path)
                        write_namedtuple_table(entry_rows, entries_path)
                    end

                    if !esmda_diag_done
                        diagnostics_seconds = @elapsed row = esmda_diagnostic_row(
                            spec.density,
                            noise_sigma,
                            noise_seed,
                            dataset,
                            observations,
                            esmda,
                            P_true,
                        )
                        row = merge(row, (diagnostics_seconds=diagnostics_seconds,))
                        push!(diagnostic_rows, row)
                        write_namedtuple_table(diagnostic_rows, diagnostics_path)
                    end

                    metric = last([row for row in metric_rows if parse_table_string(row.density) == spec.density &&
                        isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0) &&
                        parse_table_int(row.noise_seed) == noise_seed &&
                        parse_table_string(row.method) == "esmda"])
                    diag = last([row for row in diagnostic_rows if parse_table_string(row.density) == spec.density &&
                        isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0) &&
                        parse_table_int(row.noise_seed) == noise_seed])
                    @printf(
                        "  ESMDA done %.1fs turning=%.4f pred=%.4f final_unobs=%.4f turn_cov=%.3f pred_cov=%.3f state_unobs_cov=%.3f\n",
                        esmda.solve_seconds,
                        metric.turning_rmse,
                        metric.predictive_rmse,
                        metric.final_state_rmse_unobserved,
                        diag.turning_interval_coverage,
                        diag.predictive_interval_coverage,
                        diag.final_state_interval_coverage_unobserved,
                    )
                    flush(stdout)
                else
                    @printf("skipping existing ESMDA density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                end

                if fit_completed(metric_rows, spec.density, noise_sigma, noise_seed, "map_lbfgs")
                    @printf("skipping existing MAP/L-BFGS density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                    continue
                end

                @printf("running MAP/L-BFGS density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                flush(stdout)
                lbfgs = run_lbfgs_map_multi_scenario_timed(
                    observations.y_obs,
                    dataset,
                    observations.sigma_model;
                    time_limit_seconds=DENSITY_NOISE_LBFGS_SECONDS,
                )
                push!(metric_rows, lbfgs_metric_row(spec.density, noise_sigma, noise_seed, dataset, observations, lbfgs, P_true))
                append!(entry_rows, turning_entry_rows_for_fit(spec.density, noise_sigma, noise_seed, "map_lbfgs", lbfgs.P_est, P_true))
                write_namedtuple_table(metric_rows, metrics_path)
                write_namedtuple_table(entry_rows, entries_path)
                write_lbfgs_summary(metric_rows; output_dir=output_dir)
                @printf(
                    "  MAP/L-BFGS done %.1fs iter=%d loss=%.3g turning=%.4f pred=%.4f final_unobs=%.4f\n",
                    lbfgs.solve_seconds,
                    lbfgs.iterations,
                    lbfgs.best_loss,
                    last(metric_rows).turning_rmse,
                    last(metric_rows).predictive_rmse,
                    last(metric_rows).final_state_rmse_unobserved,
                )
                flush(stdout)
            end
        end
    end

    summary_rows = write_lbfgs_summary(metric_rows; output_dir=output_dir)
    plot_paths = run_density_noise_full_lbfgs_plots(; output_dir=output_dir)

    println()
    println("Density-noise L-BFGS full outputs")
    println("---------------------------------")
    println(metrics_path)
    println(entries_path)
    println(diagnostics_path)
    println(joinpath(output_dir, "full_summary.tsv"))
    println(joinpath(output_dir, "full_sensor_layouts.tsv"))
    for path in plot_paths
        println(path)
    end
    println()

    for row in summary_rows
        @printf(
            "%s noise=%.3f %-9s n=%d turning=%.4f [%.4f, %.4f] pred=%.4f final_unobs=%.4f solve=%.1fs\n",
            row.density,
            row.noise_sigma,
            row.method,
            row.seed_count,
            row.turning_rmse_mean,
            row.turning_rmse_min,
            row.turning_rmse_max,
            row.predictive_rmse_mean,
            row.final_state_rmse_unobserved_mean,
            row.solve_seconds_mean,
        )
    end

    return summary_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "full"
    if mode == "full"
        run_density_noise_full_lbfgs()
    elseif mode == "plot_full"
        run_density_noise_full_lbfgs_plots()
    else
        error("Unknown mode $(mode). Use full or plot_full.")
    end
end
