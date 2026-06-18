include(joinpath(@__DIR__, "plots.jl"))

using Optim

const REPRESENTATIVE_LBFGS_SECONDS = 300.0
const REPRESENTATIVE_LBFGS_OUTPUT_DIR = joinpath(@__DIR__, "outputs")
const REPRESENTATIVE_LBFGS_FIGURE_DIR = joinpath(@__DIR__, "figures")

function run_lbfgs_map_multi_scenario_timed(
    y_obs::AbstractVector,
    dataset::MultiScenarioDataset,
    sigma_model::AbstractVector;
    prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
    z0=zeros(N_PARAMS),
    time_limit_seconds=REPRESENTATIVE_LBFGS_SECONDS,
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

function rename_map_road_rows(rows)
    return [
        (
            road_id=row.road_id,
            observed=row.observed,
            map_rmse=row.adam_rmse,
            esmda_rmse=row.esmda_rmse,
            display_score=row.display_score,
        )
        for row in rows
    ]
end

function run_square_multi_scenario_representative_lbfgs(;
    seed=1,
    output_dir=REPRESENTATIVE_LBFGS_OUTPUT_DIR,
    figure_dir=REPRESENTATIVE_LBFGS_FIGURE_DIR,
)
    mkpath(output_dir)
    mkpath(figure_dir)
    P_true = true_turning_matrices()
    dataset = build_representative_dataset(REPRESENTATIVE_HORIZON_MINUTES)
    setup = only(dataset.setups)
    observations = generate_physical_dataset_observations(P_true, dataset; seed=seed)

    write_config_file(
        joinpath(output_dir, "square_representative_lbfgs_config.txt"),
        [
            "experiment = square multi-scenario representative reconstruction with L-BFGS MAP",
            "source = sources/reconstruction_figures.jl",
            "horizon_minutes = $(REPRESENTATIVE_HORIZON_MINUTES)",
            "noise_seed = $(seed)",
            "peak_noise_sigma = $(setup.physical_noise_peak_sigma)",
            "floor_fraction = $(DEFAULT_SIGMA_FLOOR_FRACTION)",
            "prior_scale = $(MULTI_SCENARIO_PRIOR_SCALE)",
            "esmda = $(MULTI_SCENARIO_ESMDA_ENSEMBLE_SIZE)x$(MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS)",
            "map_optimizer = Optim.jl L-BFGS",
            "map_time_limit_seconds = $(REPRESENTATIVE_LBFGS_SECONDS)",
            "note = neighbor output only; thesis figures are not overwritten",
        ],
    )

    println("Square-network representative reconstruction with L-BFGS")
    println("--------------------------------------------------------")
    println(@sprintf("horizon: %.1f minutes", REPRESENTATIVE_HORIZON_MINUTES))
    println(@sprintf("observations: %d", dataset_observation_length(dataset)))
    println(@sprintf("peak noise scale: %.4f", setup.physical_noise_peak_sigma))
    println(@sprintf("L-BFGS budget: %.0fs", REPRESENTATIVE_LBFGS_SECONDS))

    println("Running ESMDA")
    flush(stdout)
    esmda = run_esmda_multi_scenario(
        dataset,
        observations.y_obs,
        observations.sigma_model;
        seed=seed,
        prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
        ensemble_size=MULTI_SCENARIO_ESMDA_ENSEMBLE_SIZE,
        esmda_maxiters=MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
        P_true=P_true,
    )

    println("Running MAP/L-BFGS")
    flush(stdout)
    lbfgs = run_lbfgs_map_multi_scenario_timed(
        observations.y_obs,
        dataset,
        observations.sigma_model;
        prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
        time_limit_seconds=REPRESENTATIVE_LBFGS_SECONDS,
    )

    y_true = observations.y_true
    metric_rows = [
        multi_scenario_metric_row(
            dataset,
            :map_lbfgs,
            "lbfgs_$(Int(round(REPRESENTATIVE_LBFGS_SECONDS)))s",
            REPRESENTATIVE_LBFGS_SECONDS,
            lbfgs.solve_seconds,
            lbfgs.iterations,
            lbfgs.P_est,
            lbfgs.y_est,
            P_true,
            y_true,
        ),
        multi_scenario_metric_row(
            dataset,
            :esmda,
            "esmda_$(MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS)_iters",
            MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
            esmda.solve_seconds,
            MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
            esmda.P_post_mean,
            esmda.y_post_mean,
            P_true,
            y_true,
            state_metrics=ensemble_mean_state_metrics(esmda, dataset, P_true),
        ),
    ]
    metrics_path = write_namedtuple_table(metric_rows, joinpath(output_dir, "square_representative_lbfgs_metrics.tsv"))
    optimizer_path = write_namedtuple_table(
        [
            (
                method="map_lbfgs",
                requested_seconds=REPRESENTATIVE_LBFGS_SECONDS,
                solve_seconds=lbfgs.solve_seconds,
                iterations=lbfgs.iterations,
                best_loss=lbfgs.best_loss,
                final_loss=lbfgs.final_loss,
                converged=lbfgs.converged,
            ),
        ],
        joinpath(output_dir, "square_representative_lbfgs_optimizer.tsv"),
    )

    true_state = final_state_snapshot(P_true, setup)
    map_state = final_state_snapshot(lbfgs.P_est, setup)
    esmda_summary = summarize_selected_final_states(esmda.param_samples, esmda.weights, setup, all_square_road_ids())
    representative_roads, road_error_rows, selected_road_rows = select_representative_roads(
        setup,
        true_state,
        map_state,
        esmda_summary,
    )

    final_state_path = joinpath(figure_dir, "square_representative_final_state.png")
    plot_representative_final_state(
        setup,
        true_state,
        map_state,
        esmda_summary;
        road_ids=representative_roads,
        output_path=final_state_path,
    )

    entry_map = turning_entries(lbfgs.P_est)
    selected_junction, junction_scores = select_representative_junction(esmda.entry_post_mean, esmda.entry_true)
    turning_path = joinpath(figure_dir, "square_representative_turning_junction.png")
    plot_representative_turning_boxplot(
        esmda,
        entry_map;
        junction=selected_junction,
        output_path=turning_path,
    )

    renamed_selected_rows = rename_map_road_rows(selected_road_rows)
    selection_path = joinpath(output_dir, "square_representative_lbfgs_selection.txt")
    open(selection_path, "w") do io
        println(io, @sprintf("horizon_minutes\t%.1f", REPRESENTATIVE_HORIZON_MINUTES))
        println(io, @sprintf("observation_count\t%d", dataset_observation_length(dataset)))
        println(io, @sprintf("selected_junction\t%d", selected_junction))
        println(io, @sprintf("selected_junction_label\t%s", JUNCTION_LABELS[selected_junction]))
        println(io, "mean_abs_errors\t", join([@sprintf("%.8f", value) for value in junction_scores], "\t"))
        println(io, "representative_roads\t", join(representative_roads, ","))
        println(io, "road_id\tobserved\tmap_rmse\tesmda_rmse\tdisplay_score")
        for row in renamed_selected_rows
            println(
                io,
                @sprintf(
                    "%d\t%s\t%.8f\t%.8f\t%.8f",
                    row.road_id,
                    row.observed ? "true" : "false",
                    row.map_rmse,
                    row.esmda_rmse,
                    row.display_score,
                ),
            )
        end
    end

    write_namedtuple_table(rename_map_road_rows(road_error_rows), joinpath(output_dir, "square_representative_lbfgs_road_errors.tsv"))

    println()
    println("Outputs")
    println("-------")
    println(metrics_path)
    println(optimizer_path)
    println(final_state_path)
    println(turning_path)
    println(selection_path)

    for row in metric_rows
        @printf(
            "%-9s solve=%.1fs iter=%d turning=%.4f pred=%.4f final_all=%.4f final_obs=%.4f final_unobs=%.4f\n",
            row.method,
            row.solve_seconds,
            row.iterations,
            row.turning_rmse,
            row.predictive_rmse,
            row.final_state_rmse_all,
            row.final_state_rmse_observed,
            row.final_state_rmse_unobserved,
        )
    end

    return (
        esmda=esmda,
        lbfgs=lbfgs,
        metric_rows=metric_rows,
        metrics_path=metrics_path,
        optimizer_path=optimizer_path,
        final_state_path=final_state_path,
        turning_path=turning_path,
        selection_path=selection_path,
        selected_junction=selected_junction,
        representative_roads=representative_roads,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_square_multi_scenario_representative_lbfgs()
end
