import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

using Printf
using Statistics
using LinearAlgebra
using Optim

include(joinpath(@__DIR__, "single_scenario_helpers.jl"))

const OBJECTIVE_LBFGS_OUTPUT_DIR = SINGLE_SCENARIO_OUTPUT_DIR
const OBJECTIVE_LBFGS_RESTART_SEEDS = [0, 1, 2, 3, 4]
const OBJECTIVE_LBFGS_SECONDS = 300.0
const OBJECTIVE_LBFGS_FLOOR_FRACTION = 0.15
const OBJECTIVE_LBFGS_PRIOR_SCALE = DEFAULT_PRIOR_SCALE

function restart_label_objective(seed::Int)
    return seed == 0 ? "prior mean" : "seed $(seed)"
end

function restart_initial_point_objective(seed::Int; init_scale=0.8)
    return seed == 0 ? zeros(N_PARAMS) : init_scale .* randn(MersenneTwister(100 + seed), N_PARAMS)
end

function run_lbfgs_forwarddiff_objective(
    loss_fn::Function;
    z0=zeros(N_PARAMS),
    time_limit_seconds=OBJECTIVE_LBFGS_SECONDS,
    maxiters=10_000,
)
    z_start = Float64.(collect(z0))

    function fg!(F, G, z)
        if G !== nothing
            ForwardDiff.gradient!(G, loss_fn, z)
        end
        if F !== nothing
            return loss_fn(z)
        end
        return nothing
    end

    options = Optim.Options(
        iterations=maxiters,
        store_trace=true,
        show_trace=false,
        time_limit=Float64(time_limit_seconds),
    )
    result = optimize(Optim.only_fg!(fg!), z_start, LBFGS(), options)
    z_best = Optim.minimizer(result)
    final_loss = loss_fn(z_best)
    final_grad = ForwardDiff.gradient(loss_fn, z_best)
    losses = [tr.value for tr in Optim.trace(result) if tr.value !== nothing]

    return (
        z=copy(z_best),
        z_best=copy(z_best),
        best_loss=Optim.minimum(result),
        best_iter=Optim.iterations(result),
        final_loss=final_loss,
        final_raw_grad_norm=norm(final_grad),
        final_postclip_grad_norm=norm(final_grad),
        losses=losses,
        raw_grad_norms=Float64[],
        grad_norms=Float64[],
        learning_rates=Float64[],
        clipped_flags=Bool[],
        clip_count=0,
        clip_fraction=0.0,
        loss_tail_relspan=tail_relative_span(losses),
        grad_tail_relspan=NaN,
        terminated_early=false,
        solve_seconds=Optim.time_run(result),
        iterations=Optim.iterations(result),
        converged=Optim.converged(result),
        result=result,
    )
end

function objective_extra_metrics(z, problem, setup; prior_scale=OBJECTIVE_LBFGS_PRIOR_SCALE)
    map_value = map_loss_forwarddiff(z, problem.y_obs, setup, problem.sigma_model; prior_scale=prior_scale)
    mse_value = mse_loss_forwarddiff(z, problem.y_obs, setup)
    return (
        map_loss_at_solution=map_value,
        mse_loss_at_solution=mse_value,
    )
end

function objective_lbfgs_metric_row(loss_kind, restart_seed, result, setup, problem; prior_scale=OBJECTIVE_LBFGS_PRIOR_SCALE)
    base = result_metric_row(
        uppercase(string(loss_kind)),
        loss_kind,
        restart_label_objective(restart_seed),
        restart_seed,
        result,
        setup,
        problem.P_true,
        problem.y_true,
        problem.y_obs,
        problem.true_state,
    )
    extra = objective_extra_metrics(result.z, problem, setup; prior_scale=prior_scale)
    return (
        base...,
        optimizer="lbfgs",
        requested_seconds=OBJECTIVE_LBFGS_SECONDS,
        converged=result.converged,
        extra...,
    )
end

function objective_metric_summary(values)
    vals = Float64.(values)
    return (
        mean=mean(vals),
        median=median(vals),
        minimum=minimum(vals),
        maximum=maximum(vals),
        std=length(vals) > 1 ? std(vals) : 0.0,
    )
end

function summarize_objective_rows(rows)
    summary_rows = NamedTuple[]

    for loss_kind in unique(getproperty.(rows, :loss_kind))
        group = filter(row -> row.loss_kind == loss_kind, rows)
        turning = objective_metric_summary(getproperty.(group, :turning_rmse))
        state = objective_metric_summary(getproperty.(group, :final_state_rmse_all))
        pred = objective_metric_summary(getproperty.(group, :predictive_rmse))
        fit = objective_metric_summary(getproperty.(group, :fit_rmse))
        map_loss = objective_metric_summary(getproperty.(group, :map_loss_at_solution))
        mse_loss = objective_metric_summary(getproperty.(group, :mse_loss_at_solution))
        seconds = objective_metric_summary(getproperty.(group, :solve_seconds))
        iterations = objective_metric_summary(getproperty.(group, :iterations))

        push!(
            summary_rows,
            (
                loss_kind=loss_kind,
                restarts=length(group),
                turning_rmse_mean=turning.mean,
                turning_rmse_median=turning.median,
                turning_rmse_min=turning.minimum,
                turning_rmse_max=turning.maximum,
                final_state_rmse_all_mean=state.mean,
                final_state_rmse_all_median=state.median,
                final_state_rmse_all_min=state.minimum,
                final_state_rmse_all_max=state.maximum,
                predictive_rmse_mean=pred.mean,
                predictive_rmse_median=pred.median,
                predictive_rmse_min=pred.minimum,
                predictive_rmse_max=pred.maximum,
                fit_rmse_mean=fit.mean,
                fit_rmse_median=fit.median,
                fit_rmse_min=fit.minimum,
                fit_rmse_max=fit.maximum,
                map_loss_at_solution_mean=map_loss.mean,
                map_loss_at_solution_median=map_loss.median,
                mse_loss_at_solution_mean=mse_loss.mean,
                mse_loss_at_solution_median=mse_loss.median,
                solve_seconds_mean=seconds.mean,
                solve_seconds_max=seconds.maximum,
                iterations_mean=iterations.mean,
                iterations_max=iterations.maximum,
                converged_count=count(row -> row.converged, group),
            ),
        )
    end

    return summary_rows
end

function protect_output_dir(output_dir; allow_overwrite=false)
    if isdir(output_dir) && !isempty(readdir(output_dir)) && !allow_overwrite
        error("Refusing to overwrite non-empty output directory: $(output_dir)")
    end
    mkpath(output_dir)
    return output_dir
end

function run_square_single_scenario_objective_choice_lbfgs(;
    seed=1,
    peak_noise_sigma=DEFAULT_PEAK_NOISE_SIGMA,
    sigma_floor_fraction=OBJECTIVE_LBFGS_FLOOR_FRACTION,
    prior_scale=OBJECTIVE_LBFGS_PRIOR_SCALE,
    init_scale=0.8,
    time_limit_seconds=OBJECTIVE_LBFGS_SECONDS,
    output_dir=OBJECTIVE_LBFGS_OUTPUT_DIR,
    allow_overwrite=true,
)
    protect_output_dir(output_dir; allow_overwrite=allow_overwrite)

    setup = square_single_scenario_setup(peak_noise_sigma=peak_noise_sigma)
    problem = single_scenario_problem_data(setup; seed=seed, floor_fraction=sigma_floor_fraction)
    rows = NamedTuple[]

    println("Square single-scenario objective choice with L-BFGS")
    println("--------------------------------------------")
    println(@sprintf("seed: %d", seed))
    println(@sprintf("peak physical noise sigma: %.4f", peak_noise_sigma))
    println(@sprintf("sigma floor fraction: %.4f", sigma_floor_fraction))
    println(@sprintf("prior scale: %.4f", prior_scale))
    println(@sprintf("time limit per restart: %.1f s", time_limit_seconds))
    println(@sprintf("observation clip fraction: %.4f", problem.clip_fraction))
    println()

    for loss_kind in (:map, :mse)
        loss_fn = build_single_scenario_loss(
            loss_kind,
            problem.y_obs,
            setup;
            sigma_model=problem.sigma_model,
            prior_scale=prior_scale,
        )

        println("Loss: ", uppercase(string(loss_kind)))
        for restart_seed in OBJECTIVE_LBFGS_RESTART_SEEDS
            z0 = restart_initial_point_objective(restart_seed; init_scale=init_scale)
            result = run_lbfgs_forwarddiff_objective(
                loss_fn;
                z0=z0,
                time_limit_seconds=time_limit_seconds,
            )
            row = objective_lbfgs_metric_row(loss_kind, restart_seed, result, setup, problem; prior_scale=prior_scale)
            push!(rows, row)
            println(
                @sprintf(
                    "  %-10s %.1fs iter=%d conv=%s | turn %.4f | state %.4f | fit %.4f | MAP loss %.2f | MSE loss %.6f",
                    row.restart_label,
                    row.solve_seconds,
                    row.iterations,
                    string(row.converged),
                    row.turning_rmse,
                    row.final_state_rmse_all,
                    row.fit_rmse,
                    row.map_loss_at_solution,
                    row.mse_loss_at_solution,
                ),
            )
            flush(stdout)
        end
        println()
    end

    summary_rows = summarize_objective_rows(rows)
    metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "objective_lbfgs_restart_metrics.tsv"))
    summary_path = write_namedtuple_table(summary_rows, joinpath(output_dir, "objective_lbfgs_summary_metrics.tsv"))

    config_path = write_config_file(
        joinpath(output_dir, "objective_lbfgs_config.txt"),
        [
            "Square single-scenario MAP-vs-MSE objective comparison with L-BFGS",
            @sprintf("seed = %d", seed),
            @sprintf("peak_noise_sigma = %.4f", peak_noise_sigma),
            @sprintf("sigma_floor_fraction = %.4f", sigma_floor_fraction),
            @sprintf("prior_scale = %.4f", prior_scale),
            @sprintf("time_limit_seconds = %.1f", time_limit_seconds),
            "restart_seeds = $(join(OBJECTIVE_LBFGS_RESTART_SEEDS, ","))",
            @sprintf("init_scale = %.4f", init_scale),
            "optimizer = Optim.jl LBFGS",
            "",
            "Saved outputs:",
            metrics_path,
            summary_path,
        ],
    )

    println("Summary")
    println("-------")
    for row in summary_rows
        println(
            @sprintf(
                "%s | turn %.4f [%.4f, %.4f] | state %.4f | fit %.4f | MAP loss %.2f | MSE loss %.6f | conv %d/%d",
                uppercase(row.loss_kind),
                row.turning_rmse_mean,
                row.turning_rmse_min,
                row.turning_rmse_max,
                row.final_state_rmse_all_mean,
                row.fit_rmse_mean,
                row.map_loss_at_solution_mean,
                row.mse_loss_at_solution_mean,
                row.converged_count,
                row.restarts,
            ),
        )
    end
    println()
    println("Saved outputs:")
    println("  ", metrics_path)
    println("  ", summary_path)
    println("  ", config_path)

    return (
        rows=rows,
        summary_rows=summary_rows,
        metrics_path=metrics_path,
        summary_path=summary_path,
        config_path=config_path,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_SINGLE_SCENARIO_OBJECTIVE_LBFGS_DEMO", "0") != "1"
    run_square_single_scenario_objective_choice_lbfgs()
end
