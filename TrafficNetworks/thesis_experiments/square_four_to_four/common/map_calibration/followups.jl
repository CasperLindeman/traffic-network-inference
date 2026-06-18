if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using Statistics
using LinearAlgebra
using Plots

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_BUDGET_DEMO"] = "1"
if !isdefined(@__MODULE__, :BUDGET_OUTPUT_DIR)
    include(joinpath(@__DIR__, "budget_comparison.jl"))
end

FOLLOWUP_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_FOLLOWUP_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "generated", "square_four_to_four", "square_four_to_four_followups"),
)

function generate_square_observations(setup::SquareFourToFourSetup, seed::Int)
    rng = MersenneTwister(seed)
    P_true = true_turning_matrices()
    y_true = simulator(P_true, setup)
    y_obs = if setup.generated_noise_sigma > 0
        y_true .+ setup.generated_noise_sigma .* randn(rng, length(y_true))
    else
        copy(y_true)
    end
    return P_true, y_true, y_obs
end

function plot_optimizer_budget_comparison(rows; output_path=nothing)
    budgets = unique(getproperty.(rows, :budget_seconds))
    sort!(budgets)
    methods = [:adam_fd, :adam_forwarddiff, :lbfgs_forwarddiff]
    specs = [
        (key=:turning_rmse, title="Turning-fraction RMSE", ylabel="RMSE"),
        (key=:final_state_rmse_all, title="Final-state RMSE (all roads)", ylabel="RMSE"),
        (key=:final_state_rmse_unobserved, title="Final-state RMSE (unobserved roads)", ylabel="RMSE"),
        (key=:solve_seconds, title="Actual solve time", ylabel="Seconds"),
    ]

    plt = plot(layout=(2, 2), size=(1320, 980), dpi=180, legend=:top, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
    x = collect(1:length(budgets))
    xtick_labels = ["$(Int(round(b))) s" for b in budgets]

    for (subplot_id, spec) in enumerate(specs)
        for method in methods
            ys = [
                first(getproperty.(filter(row -> row.budget_seconds == b && row.method == String(method), rows), spec.key))
                for b in budgets
            ]
            plot!(
                plt,
                x,
                ys;
                color=LONGRUN_COLORS[method],
                linewidth=2.4,
                markershape=:circle,
                markersize=6,
                label=subplot_id == 1 ? LONGRUN_LABELS[method] : "",
                subplot=subplot_id,
            )
        end

        plot!(
            plt;
            xlabel="Budget",
            ylabel=spec.ylabel,
            title=spec.title,
            xticks=(x, xtick_labels),
            subplot=subplot_id,
        )
    end

    plot!(plt; plot_title="Low-noise optimizer budget comparison")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function run_low_noise_optimizer_followup(;
    noise_sigma=0.001,
    seed=1,
    budgets_seconds=(30.0, 120.0),
    prior_scale=1.25,
    output_dir=joinpath(FOLLOWUP_OUTPUT_DIR, "low_noise_optimizers"),
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    P_true, y_true, y_obs = generate_square_observations(setup, seed)
    true_state = final_state_snapshot(P_true, setup)

    println("Low-noise optimizer follow-up")
    println("-----------------------------")
    println(@sprintf("noise sigma: %.4f, seed: %d", noise_sigma, seed))
    println("Warming up optimizer paths")
    run_adam_map_timed(y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    run_adam_map_forwarddiff_timed(y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    run_lbfgs_map_forwarddiff_timed(y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    println()

    rows = NamedTuple[]
    for budget in budgets_seconds
        println(@sprintf("Running optimizer budgets at %.0f seconds", budget))

        adam_fd = run_adam_map_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        adam_fd_state = final_state_snapshot(adam_fd.z_best, setup)
        push!(
            rows,
            budget_metric_row(
                :adam_fd,
                budget,
                adam_fd.solve_seconds,
                adam_fd.iterations,
                adam_fd.P_est,
                adam_fd.y_est,
                adam_fd_state,
                true_state,
                y_true,
                y_obs,
                setup;
                best_loss=adam_fd.best_loss,
            ),
        )
        println(@sprintf("  ADAM finite diff: %d iterations, %.2fs", adam_fd.iterations, adam_fd.solve_seconds))

        adam_forwarddiff = run_adam_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        adam_forwarddiff_state = final_state_snapshot(adam_forwarddiff.z_best, setup)
        push!(
            rows,
            budget_metric_row(
                :adam_forwarddiff,
                budget,
                adam_forwarddiff.solve_seconds,
                adam_forwarddiff.iterations,
                adam_forwarddiff.P_est,
                adam_forwarddiff.y_est,
                adam_forwarddiff_state,
                true_state,
                y_true,
                y_obs,
                setup;
                best_loss=adam_forwarddiff.best_loss,
            ),
        )
        println(@sprintf("  ADAM ForwardDiff: %d iterations, %.2fs", adam_forwarddiff.iterations, adam_forwarddiff.solve_seconds))

        lbfgs_forwarddiff = run_lbfgs_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        lbfgs_forwarddiff_state = final_state_snapshot(lbfgs_forwarddiff.z_best, setup)
        push!(
            rows,
            budget_metric_row(
                :lbfgs_forwarddiff,
                budget,
                lbfgs_forwarddiff.solve_seconds,
                lbfgs_forwarddiff.iterations,
                lbfgs_forwarddiff.P_est,
                lbfgs_forwarddiff.y_est,
                lbfgs_forwarddiff_state,
                true_state,
                y_true,
                y_obs,
                setup;
                best_loss=lbfgs_forwarddiff.best_loss,
            ),
        )
        println(@sprintf("  L-BFGS ForwardDiff: %d iterations, %.2fs", lbfgs_forwarddiff.iterations, lbfgs_forwarddiff.solve_seconds))
        println()
    end

    metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "low_noise_optimizer_metrics.tsv"))
    plot_path = joinpath(output_dir, "low_noise_optimizer_budget_comparison.png")
    comparison_plot = plot_optimizer_budget_comparison(rows; output_path=plot_path)

    println("Low-noise optimizer summary")
    println("---------------------------")
    for budget in budgets_seconds
        println(@sprintf("Budget %.0fs", budget))
        for row in filter(r -> r.budget_seconds == budget, rows)
            println(
                @sprintf(
                    "  %-20s turning %.4f | full state %.4f | unobserved %.4f | fit %.4f",
                    LONGRUN_LABELS[Symbol(row.method)],
                    row.turning_rmse,
                    row.final_state_rmse_all,
                    row.final_state_rmse_unobserved,
                    row.fit_rmse,
                ),
            )
        end
        println()
    end

    println("Saved outputs:")
    println("  ", metrics_path)
    println("  ", plot_path)

    return (
        setup=setup,
        rows=rows,
        metrics_path=metrics_path,
        plot_path=plot_path,
        comparison_plot=comparison_plot,
    )
end

function plot_esmda_tuning(rows; output_path=nothing)
    labels = getproperty.(rows, :label)
    x = collect(1:length(rows))
    plt = plot(layout=(2, 2), size=(1320, 980), dpi=180, legend=false, left_margin=10Plots.mm, bottom_margin=10Plots.mm)

    bar!(plt, x, getproperty.(rows, :turning_rmse); color=:steelblue, xticks=(x, labels), xrotation=20, ylabel="RMSE", title="Turning-fraction RMSE", subplot=1)
    bar!(plt, x, getproperty.(rows, :final_state_rmse_all); color=:seagreen4, xticks=(x, labels), xrotation=20, ylabel="RMSE", title="Final-state RMSE (all roads)", subplot=2)
    bar!(plt, x, getproperty.(rows, :final_state_rmse_unobserved); color=:darkorange2, xticks=(x, labels), xrotation=20, ylabel="RMSE", title="Final-state RMSE (unobserved roads)", subplot=3)
    bar!(plt, x, getproperty.(rows, :solve_seconds); color=:firebrick, xticks=(x, labels), xrotation=20, ylabel="Seconds", title="Solve time", subplot=4)
    plot!(plt; plot_title="High-noise ESMDA tuning")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function run_high_noise_esmda_followup(;
    noise_sigma=0.100,
    seed=1,
    prior_scale=1.25,
    configs=[
        (label="192x6", ensemble_size=192, esmda_maxiters=6),
        (label="128x12", ensemble_size=128, esmda_maxiters=12),
        (label="100x16", ensemble_size=100, esmda_maxiters=16),
    ],
    output_dir=joinpath(FOLLOWUP_OUTPUT_DIR, "high_noise_esmda"),
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    P_true = true_turning_matrices()
    true_state = final_state_snapshot(P_true, setup)

    println("High-noise ESMDA follow-up")
    println("--------------------------")
    println(@sprintf("noise sigma: %.4f, seed: %d", noise_sigma, seed))

    rows = NamedTuple[]
    for cfg in configs
        println(@sprintf("Running ESMDA config %s", cfg.label))
        result = run_square_four_to_four_esmda(
            setup;
            rng=MersenneTwister(seed),
            prior_scale=prior_scale,
            ensemble_size=cfg.ensemble_size,
            esmda_maxiters=cfg.esmda_maxiters,
            P_true=P_true,
        )
        posterior_summary = summarize_final_states(result.param_samples, setup, result.weights; level=0.90)
        esmda_state = posterior_summary.mean
        push!(
            rows,
            (
                label=cfg.label,
                ensemble_size=cfg.ensemble_size,
                esmda_maxiters=cfg.esmda_maxiters,
                solve_seconds=result.solve_seconds,
                turning_rmse=overall_turning_rmse(result.P_post_mean, P_true),
                predictive_rmse=predictive_rmse(result.y_post_mean, result.y_true),
                fit_rmse=predictive_rmse(result.y_post_mean, result.y_obs),
                final_state_rmse_all=final_state_rmse(esmda_state, true_state),
                final_state_rmse_observed=observed_road_state_rmse(esmda_state, true_state, setup.observed_road_ids),
                final_state_rmse_unobserved=observed_road_state_rmse(esmda_state, true_state, unobserved_square_road_ids(setup)),
                turning_interval_width=turning_interval_width(result.entry_ci_05, result.entry_ci_95),
                turning_interval_coverage=turning_interval_coverage(result.entry_true, result.entry_ci_05, result.entry_ci_95),
            ),
        )
        println(
            @sprintf(
                "  turning %.4f | full state %.4f | unobserved %.4f | time %.2fs",
                last(rows).turning_rmse,
                last(rows).final_state_rmse_all,
                last(rows).final_state_rmse_unobserved,
                last(rows).solve_seconds,
            ),
        )
    end
    println()

    metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "high_noise_esmda_metrics.tsv"))
    plot_path = joinpath(output_dir, "high_noise_esmda_tuning.png")
    tuning_plot = plot_esmda_tuning(rows; output_path=plot_path)

    println("Saved outputs:")
    println("  ", metrics_path)
    println("  ", plot_path)

    return (
        setup=setup,
        rows=rows,
        metrics_path=metrics_path,
        plot_path=plot_path,
        tuning_plot=tuning_plot,
    )
end

function estimate_square_nuts_cost(;
    noise_sigma=0.001,
    seed=1,
    prior_scale=1.25,
    repeats=3,
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    _, _, y_obs = generate_square_observations(setup, seed)
    inference_prob = build_square_inference_problem_forwarddiff(setup, y_obs; prior_scale=prior_scale)
    b = SimulationBasedInference.bijector(inference_prob)
    q0 = b(rand(MersenneTwister(10_000 + seed), inference_prob.prior))
    wrapped = ForwardDiffGradientLogDensity(logdensity(inference_prob))

    logdensity_and_gradient(wrapped, q0)
    grad_times = [@elapsed logdensity_and_gradient(wrapped, q0) for _ in 1:repeats]
    avg_grad_seconds = mean(grad_times)

    estimates = (
        one_hundred_gradients=100 * avg_grad_seconds,
        one_thousand_gradients=1_000 * avg_grad_seconds,
        three_thousand_gradients=3_000 * avg_grad_seconds,
    )

    println("Square-system NUTS cost estimate")
    println("--------------------------------")
    println(@sprintf("noise sigma: %.4f", noise_sigma))
    println("parameter dimension: 49")
    println(@sprintf("average logdensity+gradient call: %.4fs", avg_grad_seconds))
    println(@sprintf("100 gradients   ~= %.2f s", estimates.one_hundred_gradients))
    println(@sprintf("1000 gradients  ~= %.2f s", estimates.one_thousand_gradients))
    println(@sprintf("3000 gradients  ~= %.2f s", estimates.three_thousand_gradients))
    println()

    return (
        avg_grad_seconds=avg_grad_seconds,
        estimates=estimates,
    )
end

function run_low_noise_hybrid_followup(;
    noise_sigma=0.001,
    seed=1,
    prior_scale=1.25,
    adam_budget_seconds=30.0,
    lbfgs_budget_seconds=90.0,
    output_dir=joinpath(FOLLOWUP_OUTPUT_DIR, "low_noise_hybrid"),
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    P_true, y_true, y_obs = generate_square_observations(setup, seed)
    true_state = final_state_snapshot(P_true, setup)

    println("Low-noise hybrid follow-up")
    println("--------------------------")
    println(@sprintf("noise sigma: %.4f, seed: %d", noise_sigma, seed))
    println(@sprintf("ADAM budget: %.0fs, L-BFGS budget: %.0fs", adam_budget_seconds, lbfgs_budget_seconds))
    println("Warming up AD and L-BFGS")
    run_adam_map_forwarddiff_timed(y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    run_lbfgs_map_forwarddiff_timed(y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    println()

    adam_forwarddiff = run_adam_map_forwarddiff_timed(
        y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        time_limit_seconds=adam_budget_seconds,
    )
    adam_state = final_state_snapshot(adam_forwarddiff.z_best, setup)

    lbfgs_zero = run_lbfgs_map_forwarddiff_timed(
        y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        time_limit_seconds=lbfgs_budget_seconds,
    )
    lbfgs_zero_state = final_state_snapshot(lbfgs_zero.z_best, setup)

    lbfgs_warm = run_lbfgs_map_forwarddiff_timed(
        y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        z0=adam_forwarddiff.z_best,
        time_limit_seconds=lbfgs_budget_seconds,
    )
    lbfgs_warm_state = final_state_snapshot(lbfgs_warm.z_best, setup)

    rows = [
        (
            label="ADAM $(Int(round(adam_budget_seconds)))s",
            solve_seconds=adam_forwarddiff.solve_seconds,
            iterations=adam_forwarddiff.iterations,
            turning_rmse=overall_turning_rmse(adam_forwarddiff.P_est, P_true),
            predictive_rmse=predictive_rmse(adam_forwarddiff.y_est, y_true),
            fit_rmse=predictive_rmse(adam_forwarddiff.y_est, y_obs),
            final_state_rmse_all=final_state_rmse(adam_state, true_state),
            final_state_rmse_unobserved=observed_road_state_rmse(adam_state, true_state, unobserved_square_road_ids(setup)),
            best_loss=adam_forwarddiff.best_loss,
        ),
        (
            label="L-BFGS $(Int(round(lbfgs_budget_seconds)))s from zero",
            solve_seconds=lbfgs_zero.solve_seconds,
            iterations=lbfgs_zero.iterations,
            turning_rmse=overall_turning_rmse(lbfgs_zero.P_est, P_true),
            predictive_rmse=predictive_rmse(lbfgs_zero.y_est, y_true),
            fit_rmse=predictive_rmse(lbfgs_zero.y_est, y_obs),
            final_state_rmse_all=final_state_rmse(lbfgs_zero_state, true_state),
            final_state_rmse_unobserved=observed_road_state_rmse(lbfgs_zero_state, true_state, unobserved_square_road_ids(setup)),
            best_loss=lbfgs_zero.best_loss,
        ),
        (
            label="ADAM $(Int(round(adam_budget_seconds)))s + L-BFGS $(Int(round(lbfgs_budget_seconds)))s",
            solve_seconds=adam_forwarddiff.solve_seconds + lbfgs_warm.solve_seconds,
            iterations=adam_forwarddiff.iterations + lbfgs_warm.iterations,
            turning_rmse=overall_turning_rmse(lbfgs_warm.P_est, P_true),
            predictive_rmse=predictive_rmse(lbfgs_warm.y_est, y_true),
            fit_rmse=predictive_rmse(lbfgs_warm.y_est, y_obs),
            final_state_rmse_all=final_state_rmse(lbfgs_warm_state, true_state),
            final_state_rmse_unobserved=observed_road_state_rmse(lbfgs_warm_state, true_state, unobserved_square_road_ids(setup)),
            best_loss=lbfgs_warm.best_loss,
        ),
    ]

    metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "low_noise_hybrid_metrics.tsv"))

    println("Hybrid summary")
    println("--------------")
    for row in rows
        println(
            @sprintf(
                "  %-24s turning %.4f | full state %.4f | unobserved %.4f | fit %.4f | time %.2fs",
                row.label,
                row.turning_rmse,
                row.final_state_rmse_all,
                row.final_state_rmse_unobserved,
                row.fit_rmse,
                row.solve_seconds,
            ),
        )
    end
    println()
    println("Saved outputs:")
    println("  ", metrics_path)

    return (
        setup=setup,
        rows=rows,
        metrics_path=metrics_path,
        adam=adam_forwarddiff,
        lbfgs_zero=lbfgs_zero,
        lbfgs_warm=lbfgs_warm,
    )
end

function run_square_four_to_four_followups()
    low_noise = run_low_noise_optimizer_followup()
    high_noise = run_high_noise_esmda_followup()
    nuts_cost = estimate_square_nuts_cost()
    return (
        low_noise=low_noise,
        high_noise=high_noise,
        nuts_cost=nuts_cost,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_FOLLOWUPS_DEMO", "0") != "1"
    followup_results = run_square_four_to_four_followups()
end
