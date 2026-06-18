if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using Plots
using Statistics

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_FOLLOWUPS_DEMO"] = "1"
if !isdefined(@__MODULE__, :FOLLOWUP_OUTPUT_DIR)
    include(joinpath(@__DIR__, "followups.jl"))
end

RUNTIME_VERIFY_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFY_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "generated", "square_four_to_four", "square_four_to_four_runtime_verification"),
)
LONGRUN_COLORS[:hybrid_forwarddiff] = :purple4
LONGRUN_LABELS[:hybrid_forwarddiff] = "ADAM + L-BFGS"

function optimizer_metric_row(label::String, method::Symbol, solve_seconds, iterations, P_est, y_est, true_state, y_true, y_obs, setup; best_loss=NaN)
    return (
        label=label,
        method=String(method),
        solve_seconds=Float64(solve_seconds),
        iterations=Int(iterations),
        turning_rmse=overall_turning_rmse(P_est, true_turning_matrices()),
        predictive_rmse=predictive_rmse(y_est, y_true),
        fit_rmse=predictive_rmse(y_est, y_obs),
        final_state_rmse_all=final_state_rmse(reduce(hcat, [final_state_snapshot(P_est, setup)[:, i] for i in 1:size(final_state_snapshot(P_est, setup), 2)]), true_state),
        final_state_rmse_observed=observed_road_state_rmse(final_state_snapshot(P_est, setup), true_state, setup.observed_road_ids),
        final_state_rmse_unobserved=observed_road_state_rmse(final_state_snapshot(P_est, setup), true_state, unobserved_square_road_ids(setup)),
        best_loss=Float64(best_loss),
    )
end

function optimizer_metric_row_from_state(label::String, method::Symbol, solve_seconds, iterations, P_est, y_est, est_state, true_state, y_true, y_obs, setup; best_loss=NaN)
    return (
        label=label,
        method=String(method),
        solve_seconds=Float64(solve_seconds),
        iterations=Int(iterations),
        turning_rmse=overall_turning_rmse(P_est, true_turning_matrices()),
        predictive_rmse=predictive_rmse(y_est, y_true),
        fit_rmse=predictive_rmse(y_est, y_obs),
        final_state_rmse_all=final_state_rmse(est_state, true_state),
        final_state_rmse_observed=observed_road_state_rmse(est_state, true_state, setup.observed_road_ids),
        final_state_rmse_unobserved=observed_road_state_rmse(est_state, true_state, unobserved_square_road_ids(setup)),
        best_loss=Float64(best_loss),
    )
end

function plot_runtime_metric_grid(rows, metric_specs; output_path=nothing, title="")
    plt = plot(layout=(2, 2), size=(1360, 980), dpi=180, legend=:topright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
    methods = unique(getproperty.(rows, :method))
    method_symbols = Symbol.(methods)

    for (subplot_id, spec) in enumerate(metric_specs)
        for method in method_symbols
            method_rows = filter(row -> row.method == String(method), rows)
            xs = getproperty.(method_rows, :solve_seconds)
            ys = getproperty.(method_rows, spec.key)
            order = sortperm(xs)
            plot!(
                plt,
                xs[order],
                ys[order];
                color=LONGRUN_COLORS[method],
                linewidth=2.4,
                markershape=:circle,
                markersize=6,
                label=subplot_id == 1 ? LONGRUN_LABELS[method] : "",
                subplot=subplot_id,
            )
            scatter!(
                plt,
                xs[order],
                ys[order];
                color=LONGRUN_COLORS[method],
                markershape=:circle,
                markersize=6,
                label="",
                subplot=subplot_id,
            )
            for idx in eachindex(order)
                annotate!(
                    plt,
                    xs[order][idx],
                    ys[order][idx],
                    text(method_rows[order[idx]].label, 7, :black, :left),
                    subplot=subplot_id,
                )
            end
        end

        plot!(
            plt;
            xlabel="Solve time (s)",
            ylabel=spec.ylabel,
            title=spec.title,
            subplot=subplot_id,
        )
    end

    plot!(plt; plot_title=title)

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function run_optimizer_runtime_verification(;
    noise_sigma=0.001,
    seed=1,
    prior_scale=1.25,
    adam_budgets=(30.0, 120.0, 300.0),
    lbfgs_budgets=(30.0, 120.0, 300.0),
    hybrid_budgets=((30.0, 90.0), (30.0, 270.0)),
    output_dir=joinpath(RUNTIME_VERIFY_OUTPUT_DIR, "optimizer_low_noise"),
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    P_true, y_true, y_obs = generate_square_observations(setup, seed)
    true_state = final_state_snapshot(P_true, setup)

    println("Optimizer runtime verification")
    println("------------------------------")
    println(@sprintf("noise sigma: %.4f, seed: %d", noise_sigma, seed))
    println("Warming up ADAM and L-BFGS")
    run_adam_map_forwarddiff_timed(y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    run_lbfgs_map_forwarddiff_timed(y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    println()

    rows = NamedTuple[]

    for budget in adam_budgets
        println(@sprintf("Running ADAM ForwardDiff for %.0fs", budget))
        adam = run_adam_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        adam_state = final_state_snapshot(adam.z_best, setup)
        push!(
            rows,
            optimizer_metric_row_from_state(
                "ADAM $(Int(round(budget)))s",
                :adam_forwarddiff,
                adam.solve_seconds,
                adam.iterations,
                adam.P_est,
                adam.y_est,
                adam_state,
                true_state,
                y_true,
                y_obs,
                setup;
                best_loss=adam.best_loss,
            ),
        )
    end

    for budget in lbfgs_budgets
        println(@sprintf("Running L-BFGS ForwardDiff from zero for %.0fs", budget))
        lbfgs = run_lbfgs_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        lbfgs_state = final_state_snapshot(lbfgs.z_best, setup)
        push!(
            rows,
            optimizer_metric_row_from_state(
                "L-BFGS $(Int(round(budget)))s",
                :lbfgs_forwarddiff,
                lbfgs.solve_seconds,
                lbfgs.iterations,
                lbfgs.P_est,
                lbfgs.y_est,
                lbfgs_state,
                true_state,
                y_true,
                y_obs,
                setup;
                best_loss=lbfgs.best_loss,
            ),
        )
    end

    for (adam_budget, lbfgs_budget) in hybrid_budgets
        total_budget = adam_budget + lbfgs_budget
        println(@sprintf("Running hybrid ADAM %.0fs + L-BFGS %.0fs", adam_budget, lbfgs_budget))
        adam = run_adam_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=adam_budget,
        )
        lbfgs = run_lbfgs_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            z0=adam.z_best,
            time_limit_seconds=lbfgs_budget,
        )
        lbfgs_state = final_state_snapshot(lbfgs.z_best, setup)
        push!(
            rows,
            optimizer_metric_row_from_state(
                "Hybrid $(Int(round(total_budget)))s",
                :hybrid_forwarddiff,
                adam.solve_seconds + lbfgs.solve_seconds,
                adam.iterations + lbfgs.iterations,
                lbfgs.P_est,
                lbfgs.y_est,
                lbfgs_state,
                true_state,
                y_true,
                y_obs,
                setup;
                best_loss=lbfgs.best_loss,
            ),
        )
    end

    metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "optimizer_runtime_metrics.tsv"))

    println("Saved outputs:")
    println("  ", metrics_path)
    println()

    return (
        rows=rows,
        metrics_path=metrics_path,
    )
end

function run_esmda_runtime_verification(;
    noise_sigma=0.100,
    seed=1,
    prior_scale=1.25,
    configs=[
        (label="192x6", ensemble_size=192, esmda_maxiters=6),
        (label="192x12", ensemble_size=192, esmda_maxiters=12),
        (label="192x20", ensemble_size=192, esmda_maxiters=20),
        (label="256x12", ensemble_size=256, esmda_maxiters=12),
        (label="256x20", ensemble_size=256, esmda_maxiters=20),
    ],
    output_dir=joinpath(RUNTIME_VERIFY_OUTPUT_DIR, "esmda_high_noise"),
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    P_true = true_turning_matrices()
    true_state = final_state_snapshot(P_true, setup)

    println("ESMDA runtime verification")
    println("--------------------------")
    println(@sprintf("noise sigma: %.4f, seed: %d", noise_sigma, seed))

    rows = NamedTuple[]
    for cfg in configs
        println(@sprintf("Running %s", cfg.label))
        res = run_square_four_to_four_esmda(
            setup;
            rng=MersenneTwister(seed),
            prior_scale=prior_scale,
            ensemble_size=cfg.ensemble_size,
            esmda_maxiters=cfg.esmda_maxiters,
            P_true=P_true,
        )
        post = summarize_final_states(res.param_samples, setup, res.weights; level=0.90)
        post_state = post.mean
        push!(
            rows,
            (
                label=cfg.label,
                method="esmda",
                ensemble_size=cfg.ensemble_size,
                esmda_maxiters=cfg.esmda_maxiters,
                work_units=cfg.ensemble_size * cfg.esmda_maxiters,
                solve_seconds=res.solve_seconds,
                turning_rmse=overall_turning_rmse(res.P_post_mean, P_true),
                predictive_rmse=predictive_rmse(res.y_post_mean, res.y_true),
                fit_rmse=predictive_rmse(res.y_post_mean, res.y_obs),
                final_state_rmse_all=final_state_rmse(post_state, true_state),
                final_state_rmse_observed=observed_road_state_rmse(post_state, true_state, setup.observed_road_ids),
                final_state_rmse_unobserved=observed_road_state_rmse(post_state, true_state, unobserved_square_road_ids(setup)),
                turning_interval_width=turning_interval_width(res.entry_ci_05, res.entry_ci_95),
                turning_interval_coverage=turning_interval_coverage(res.entry_true, res.entry_ci_05, res.entry_ci_95),
            ),
        )
    end

    metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "esmda_runtime_metrics.tsv"))
    plot_path = joinpath(output_dir, "esmda_runtime_verification.png")
    metric_specs = [
        (key=:turning_rmse, ylabel="RMSE", title="Turning-fraction RMSE"),
        (key=:final_state_rmse_all, ylabel="RMSE", title="Final-state RMSE (all roads)"),
        (key=:final_state_rmse_unobserved, ylabel="RMSE", title="Final-state RMSE (unobserved roads)"),
        (key=:turning_interval_width, ylabel="Width", title="Turning 90% interval width"),
    ]
    runtime_plot = plot_runtime_metric_grid(rows, metric_specs; output_path=plot_path, title="High-noise ESMDA runtime verification")

    println("Saved outputs:")
    println("  ", metrics_path)
    println("  ", plot_path)
    println()

    return (
        rows=rows,
        metrics_path=metrics_path,
        plot_path=plot_path,
        runtime_plot=runtime_plot,
    )
end

function print_runtime_verification_summary(opt_rows, esmda_rows)
    println("Runtime verification summary")
    println("----------------------------")
    println("Optimizer side, low-noise square case:")
    for row in sort(opt_rows, by=r -> r.solve_seconds)
        println(
            @sprintf(
                "  %-18s time %.2fs | turning %.4f | full state %.4f | unobserved %.4f | fit %.4f",
                row.label,
                row.solve_seconds,
                row.turning_rmse,
                row.final_state_rmse_all,
                row.final_state_rmse_unobserved,
                row.fit_rmse,
            ),
        )
    end
    println()
    println("ESMDA side, high-noise square case:")
    for row in sort(esmda_rows, by=r -> r.solve_seconds)
        println(
            @sprintf(
                "  %-10s time %.2fs | turning %.4f | full state %.4f | unobserved %.4f | width %.4f",
                row.label,
                row.solve_seconds,
                row.turning_rmse,
                row.final_state_rmse_all,
                row.final_state_rmse_unobserved,
                row.turning_interval_width,
            ),
        )
    end
    println()
end

function run_square_four_to_four_runtime_verification()
    opt = run_optimizer_runtime_verification()
    esmda = run_esmda_runtime_verification()
    print_runtime_verification_summary(opt.rows, esmda.rows)
    return (
        optimizer=opt,
        esmda=esmda,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFICATION_DEMO", "0") != "1"
    runtime_verification_results = run_square_four_to_four_runtime_verification()
end
