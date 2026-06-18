import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

using Printf
using Statistics

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFICATION_DEMO"] = "1"
include(joinpath(@__DIR__, "runtime_calibration_helpers.jl"))

const OPTIMIZER_MULTISEED_OUTPUT_DIR = joinpath(
    @__DIR__,
    "..",
    "outputs",
)

const OPTIMIZER_MULTISEED_CASE_METRICS = joinpath(OPTIMIZER_MULTISEED_OUTPUT_DIR, "optimizer_multiseed_case_metrics.tsv")

function append_namedtuple_row(path::AbstractString, row)
    header = collect(keys(row))
    mkpath(dirname(path))
    needs_header = !isfile(path) || filesize(path) == 0

    open(path, "a") do io
        if needs_header
            println(io, join(string.(header), '\t'))
        end
        vals = [
            value isa AbstractFloat ? @sprintf("%.8f", value) : string(value)
            for value in (getproperty(row, key) for key in header)
        ]
        println(io, join(vals, '\t'))
    end

    return path
end

function completed_optimizer_cases(path::AbstractString)
    done = Set{Tuple{Int, String}}()
    if !isfile(path)
        return done
    end

    for (line_id, line) in enumerate(eachline(path))
        line_id == 1 && continue
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        length(parts) < 3 && continue
        push!(done, (parse(Int, parts[1]), String(parts[2])))
    end

    return done
end

parse_float_cell(value::AbstractString) = value == "NaN" ? NaN : parse(Float64, value)

function read_optimizer_case_metrics(path::AbstractString)
    if !isfile(path)
        return NamedTuple[]
    end

    lines = collect(eachline(path))
    length(lines) <= 1 && return NamedTuple[]
    header = split(lines[1], '\t')
    rows = NamedTuple[]

    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        length(parts) == length(header) || continue
        cells = Dict(header[i] => parts[i] for i in eachindex(header))
        push!(
            rows,
            (
                seed=parse(Int, cells["seed"]),
                method=String(cells["method"]),
                method_label=String(cells["method_label"]),
                budget_seconds=parse_float_cell(cells["budget_seconds"]),
                adam_warmup_seconds=parse_float_cell(cells["adam_warmup_seconds"]),
                lbfgs_seconds=parse_float_cell(cells["lbfgs_seconds"]),
                learning_rate=parse_float_cell(cells["learning_rate"]),
                grad_clip=parse_float_cell(cells["grad_clip"]),
                solve_seconds=parse_float_cell(cells["solve_seconds"]),
                iterations=parse(Int, cells["iterations"]),
                best_loss=parse_float_cell(cells["best_loss"]),
                fit_rmse=parse_float_cell(cells["fit_rmse"]),
                predictive_rmse=parse_float_cell(cells["predictive_rmse"]),
                turning_rmse=parse_float_cell(cells["turning_rmse"]),
                final_state_rmse_all=parse_float_cell(cells["final_state_rmse_all"]),
                final_state_rmse_observed=parse_float_cell(cells["final_state_rmse_observed"]),
                final_state_rmse_unobserved=parse_float_cell(cells["final_state_rmse_unobserved"]),
            ),
        )
    end

    return rows
end

function optimizer_case_row(
    seed::Int,
    method::String,
    method_label::String,
    budget_seconds::Real,
    adam_warmup_seconds::Real,
    lbfgs_seconds::Real,
    learning_rate::Real,
    grad_clip::Real,
    solve_seconds::Real,
    iterations::Int,
    P_est,
    y_est,
    z_best,
    best_loss::Real,
    setup,
    true_state,
    y_true,
    y_obs,
)
    est_state = final_state_snapshot(z_best, setup)
    metric = optimizer_metric_row_from_state(
        method_label,
        Symbol(method),
        solve_seconds,
        iterations,
        P_est,
        y_est,
        est_state,
        true_state,
        y_true,
        y_obs,
        setup;
        best_loss=best_loss,
    )

    return (
        seed=seed,
        method=method,
        method_label=method_label,
        budget_seconds=Float64(budget_seconds),
        adam_warmup_seconds=Float64(adam_warmup_seconds),
        lbfgs_seconds=Float64(lbfgs_seconds),
        learning_rate=Float64(learning_rate),
        grad_clip=Float64(grad_clip),
        solve_seconds=Float64(solve_seconds),
        iterations=Int(iterations),
        best_loss=Float64(metric.best_loss),
        fit_rmse=Float64(metric.fit_rmse),
        predictive_rmse=Float64(metric.predictive_rmse),
        turning_rmse=Float64(metric.turning_rmse),
        final_state_rmse_all=Float64(metric.final_state_rmse_all),
        final_state_rmse_observed=Float64(metric.final_state_rmse_observed),
        final_state_rmse_unobserved=Float64(metric.final_state_rmse_unobserved),
    )
end

function run_optimizer_case(y_obs, setup, seed::Int, config; prior_scale=1.25)
    P_true = true_turning_matrices()
    y_true = simulator(P_true, setup)
    true_state = final_state_snapshot(P_true, setup)

    if config.method == :adam_300s_tuned
        result = run_adam_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            learning_rate=config.learning_rate,
            grad_clip=config.grad_clip,
            time_limit_seconds=config.budget_seconds,
        )
        return optimizer_case_row(
            seed,
            String(config.method),
            config.method_label,
            config.budget_seconds,
            0.0,
            0.0,
            config.learning_rate,
            config.grad_clip,
            result.solve_seconds,
            result.iterations,
            result.P_est,
            result.y_est,
            result.z_best,
            result.best_loss,
            setup,
            true_state,
            y_true,
            y_obs,
        )
    elseif config.method == :lbfgs_300s
        result = run_lbfgs_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=config.budget_seconds,
        )
        return optimizer_case_row(
            seed,
            String(config.method),
            config.method_label,
            config.budget_seconds,
            0.0,
            config.budget_seconds,
            config.learning_rate,
            config.grad_clip,
            result.solve_seconds,
            result.iterations,
            result.P_est,
            result.y_est,
            result.z_best,
            result.best_loss,
            setup,
            true_state,
            y_true,
            y_obs,
        )
    elseif config.method == :hybrid_300s
        adam = run_adam_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            learning_rate=config.learning_rate,
            grad_clip=config.grad_clip,
            time_limit_seconds=config.adam_warmup_seconds,
        )
        lbfgs = run_lbfgs_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            z0=adam.z_best,
            time_limit_seconds=config.lbfgs_seconds,
        )
        return optimizer_case_row(
            seed,
            String(config.method),
            config.method_label,
            config.budget_seconds,
            config.adam_warmup_seconds,
            config.lbfgs_seconds,
            config.learning_rate,
            config.grad_clip,
            adam.solve_seconds + lbfgs.solve_seconds,
            adam.iterations + lbfgs.iterations,
            lbfgs.P_est,
            lbfgs.y_est,
            lbfgs.z_best,
            lbfgs.best_loss,
            setup,
            true_state,
            y_true,
            y_obs,
        )
    end

    error("Unknown optimizer method $(config.method)")
end

function optimizer_multiseed_summary(rows)
    isempty(rows) && return NamedTuple[]

    best_method_by_seed = Dict{Int, String}()
    for seed in unique(getproperty.(rows, :seed))
        seed_rows = filter(row -> row.seed == seed, rows)
        best_method_by_seed[seed] = seed_rows[argmin(getproperty.(seed_rows, :best_loss))].method
    end

    summary_rows = NamedTuple[]
    for method in unique(getproperty.(rows, :method))
        method_rows = filter(row -> row.method == method, rows)
        losses = getproperty.(method_rows, :best_loss)
        fit = getproperty.(method_rows, :fit_rmse)
        turning = getproperty.(method_rows, :turning_rmse)
        state = getproperty.(method_rows, :final_state_rmse_all)
        solve_seconds = getproperty.(method_rows, :solve_seconds)
        n = length(method_rows)
        push!(
            summary_rows,
            (
                method=method,
                method_label=first(method_rows).method_label,
                n=n,
                best_loss_count=count(==(method), values(best_method_by_seed)),
                mean_best_loss=mean(losses),
                median_best_loss=median(losses),
                std_best_loss=n > 1 ? std(losses) : 0.0,
                min_best_loss=minimum(losses),
                max_best_loss=maximum(losses),
                mean_fit_rmse=mean(fit),
                mean_turning_rmse=mean(turning),
                mean_final_state_rmse_all=mean(state),
                mean_solve_seconds=mean(solve_seconds),
            ),
        )
    end

    return sort(summary_rows, by=row -> row.mean_best_loss)
end

function write_optimizer_multiseed_config(path; seeds, noise_sigma, prior_scale, configs)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "Square four-to-four optimizer multiseed experiment")
        println(io, "==================================================")
        println(io, @sprintf("noise_sigma = %.6f", noise_sigma))
        println(io, @sprintf("prior_scale = %.6f", prior_scale))
        println(io, "seeds = $(collect(seeds))")
        println(io)
        println(io, "Configs:")
        for config in configs
            println(
                io,
                @sprintf(
                    "- %s: budget %.0fs, ADAM warmup %.0fs, L-BFGS %.0fs, lr %.4g, clip %.4g",
                    config.method_label,
                    config.budget_seconds,
                    config.adam_warmup_seconds,
                    config.lbfgs_seconds,
                    config.learning_rate,
                    config.grad_clip,
                ),
            )
        end
    end
    return path
end

function run_square_four_to_four_optimizer_multiseed(;
    seeds=1:5,
    noise_sigma=0.001,
    prior_scale=1.25,
    output_dir=OPTIMIZER_MULTISEED_OUTPUT_DIR,
)
    configs = (
        (
            method=:adam_300s_tuned,
            method_label="ADAM 300s tuned",
            budget_seconds=300.0,
            adam_warmup_seconds=0.0,
            lbfgs_seconds=0.0,
            learning_rate=0.08,
            grad_clip=300.0,
        ),
        (
            method=:lbfgs_300s,
            method_label="L-BFGS 300s",
            budget_seconds=300.0,
            adam_warmup_seconds=0.0,
            lbfgs_seconds=300.0,
            learning_rate=NaN,
            grad_clip=NaN,
        ),
        (
            method=:hybrid_300s,
            method_label="ADAM 30s + L-BFGS 270s",
            budget_seconds=300.0,
            adam_warmup_seconds=30.0,
            lbfgs_seconds=270.0,
            learning_rate=0.08,
            grad_clip=300.0,
        ),
    )

    case_metrics_path = joinpath(output_dir, "optimizer_multiseed_case_metrics.tsv")
    summary_path = joinpath(output_dir, "optimizer_multiseed_summary.tsv")
    config_path = write_optimizer_multiseed_config(
        joinpath(output_dir, "optimizer_multiseed_config.txt");
        seeds=seeds,
        noise_sigma=noise_sigma,
        prior_scale=prior_scale,
        configs=configs,
    )

    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    completed = completed_optimizer_cases(case_metrics_path)
    rows = NamedTuple[]

    println("Square four-to-four optimizer multiseed")
    println("---------------------------------------")
    println(@sprintf("noise sigma: %.4f, seeds: %s", noise_sigma, collect(seeds)))
    println("Warming up ADAM and L-BFGS")
    _, _, warm_y_obs = generate_square_observations(setup, first(seeds))
    run_adam_map_forwarddiff_timed(
        warm_y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        learning_rate=0.08,
        grad_clip=300.0,
        time_limit_seconds=2.0,
    )
    run_lbfgs_map_forwarddiff_timed(
        warm_y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        time_limit_seconds=2.0,
    )
    println()

    for seed in seeds
        _, _, y_obs = generate_square_observations(setup, seed)
        for config in configs
            method_name = String(config.method)
            if (seed, method_name) in completed
                println(@sprintf("Skipping seed %d, %s (already in case table)", seed, config.method_label))
                continue
            end

            println(@sprintf("Running seed %d, %s", seed, config.method_label))
            row = run_optimizer_case(y_obs, setup, seed, config; prior_scale=prior_scale)
            append_namedtuple_row(case_metrics_path, row)
            push!(rows, row)
            println(
                @sprintf(
                    "  loss %.3e | fit %.5f | A %.4f | state %.4f | iters %d | %.1fs",
                    row.best_loss,
                    row.fit_rmse,
                    row.turning_rmse,
                    row.final_state_rmse_all,
                    row.iterations,
                    row.solve_seconds,
                ),
            )
        end
    end

    all_rows = read_optimizer_case_metrics(case_metrics_path)
    if !isempty(all_rows)
        summary_rows = optimizer_multiseed_summary(all_rows)
        write_namedtuple_table(summary_rows, summary_path)

        println()
        println("Summary by mean MAP loss:")
        for row in summary_rows
            println(
                @sprintf(
                    "  %-24s mean loss %.3e | median %.3e | best seeds %d/%d | A %.4f | state %.4f",
                    row.method_label,
                    row.mean_best_loss,
                    row.median_best_loss,
                    row.best_loss_count,
                    row.n,
                    row.mean_turning_rmse,
                    row.mean_final_state_rmse_all,
                ),
            )
        end
    else
        summary_rows = NamedTuple[]
        println("No new rows were run.")
    end

    println()
    println("Saved outputs:")
    println("  ", case_metrics_path)
    println("  ", summary_path)
    println("  ", config_path)

    return (
        rows=rows,
        summary=summary_rows,
        case_metrics_path=case_metrics_path,
        summary_path=summary_path,
        config_path=config_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    optimizer_multiseed_results = run_square_four_to_four_optimizer_multiseed()
end
