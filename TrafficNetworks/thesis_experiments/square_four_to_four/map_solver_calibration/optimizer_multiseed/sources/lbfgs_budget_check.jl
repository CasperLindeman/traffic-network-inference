import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

using Printf
using Statistics

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_RUNTIME_VERIFICATION_DEMO"] = "1"
include(joinpath(@__DIR__, "runtime_calibration_helpers.jl"))

const LBFGS_600S_OUTPUT_DIR = joinpath(
    @__DIR__,
    "..",
    "outputs",
)
const LBFGS_600S_CASE_PATH = joinpath(LBFGS_600S_OUTPUT_DIR, "lbfgs_600s_case_metrics.tsv")
const LBFGS_600S_SUMMARY_PATH = joinpath(LBFGS_600S_OUTPUT_DIR, "lbfgs_600s_summary.tsv")

function append_lbfgs_600s_row(path::AbstractString, row)
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
end

function completed_lbfgs_600s_seeds(path::AbstractString)
    done = Set{Int}()
    isfile(path) || return done
    for (line_id, line) in enumerate(eachline(path))
        line_id == 1 && continue
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        isempty(parts) && continue
        push!(done, parse(Int, parts[1]))
    end
    return done
end

function parse_lbfgs_600s_rows(path::AbstractString)
    rows = NamedTuple[]
    isfile(path) || return rows
    lines = collect(eachline(path))
    length(lines) <= 1 && return rows

    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        length(parts) < 12 && continue
        push!(
            rows,
            (
                seed=parse(Int, parts[1]),
                method_label=String(parts[2]),
                budget_seconds=parse(Float64, parts[3]),
                solve_seconds=parse(Float64, parts[4]),
                iterations=parse(Int, parts[5]),
                best_loss=parse(Float64, parts[6]),
                fit_rmse=parse(Float64, parts[7]),
                predictive_rmse=parse(Float64, parts[8]),
                turning_rmse=parse(Float64, parts[9]),
                final_state_rmse_all=parse(Float64, parts[10]),
                final_state_rmse_observed=parse(Float64, parts[11]),
                final_state_rmse_unobserved=parse(Float64, parts[12]),
            ),
        )
    end

    return rows
end

function lbfgs_600s_summary(rows)
    losses = getproperty.(rows, :best_loss)
    return (
        method_label="L-BFGS 600s",
        n=length(rows),
        mean_best_loss=mean(losses),
        median_best_loss=median(losses),
        std_best_loss=length(rows) > 1 ? std(losses) : 0.0,
        min_best_loss=minimum(losses),
        max_best_loss=maximum(losses),
        mean_fit_rmse=mean(getproperty.(rows, :fit_rmse)),
        mean_turning_rmse=mean(getproperty.(rows, :turning_rmse)),
        mean_final_state_rmse_all=mean(getproperty.(rows, :final_state_rmse_all)),
        mean_solve_seconds=mean(getproperty.(rows, :solve_seconds)),
    )
end

function run_lbfgs_600s_check(; seeds=1:5, noise_sigma=0.001, prior_scale=1.25, budget_seconds=600.0)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    completed = completed_lbfgs_600s_seeds(LBFGS_600S_CASE_PATH)

    println("L-BFGS 600s budget check")
    println("------------------------")
    println(@sprintf("noise sigma: %.4f, seeds: %s", noise_sigma, collect(seeds)))
    println("Warming up L-BFGS")
    _, _, warm_y_obs = generate_square_observations(setup, first(seeds))
    run_lbfgs_map_forwarddiff_timed(
        warm_y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        time_limit_seconds=2.0,
    )
    println()

    for seed in seeds
        if seed in completed
            println(@sprintf("Skipping seed %d (already in case table)", seed))
            continue
        end

        println(@sprintf("Running seed %d, L-BFGS %.0fs", seed, budget_seconds))
        _, y_true, y_obs = generate_square_observations(setup, seed)
        true_state = final_state_snapshot(true_turning_matrices(), setup)
        result = run_lbfgs_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget_seconds,
        )
        state = final_state_snapshot(result.z_best, setup)
        metric = optimizer_metric_row_from_state(
            "L-BFGS 600s",
            :lbfgs_forwarddiff,
            result.solve_seconds,
            result.iterations,
            result.P_est,
            result.y_est,
            state,
            true_state,
            y_true,
            y_obs,
            setup;
            best_loss=result.best_loss,
        )
        row = (
            seed=seed,
            method_label="L-BFGS 600s",
            budget_seconds=Float64(budget_seconds),
            solve_seconds=Float64(metric.solve_seconds),
            iterations=Int(metric.iterations),
            best_loss=Float64(metric.best_loss),
            fit_rmse=Float64(metric.fit_rmse),
            predictive_rmse=Float64(metric.predictive_rmse),
            turning_rmse=Float64(metric.turning_rmse),
            final_state_rmse_all=Float64(metric.final_state_rmse_all),
            final_state_rmse_observed=Float64(metric.final_state_rmse_observed),
            final_state_rmse_unobserved=Float64(metric.final_state_rmse_unobserved),
        )
        append_lbfgs_600s_row(LBFGS_600S_CASE_PATH, row)
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

    rows = parse_lbfgs_600s_rows(LBFGS_600S_CASE_PATH)
    if !isempty(rows)
        summary = lbfgs_600s_summary(rows)
        write_namedtuple_table([summary], LBFGS_600S_SUMMARY_PATH)
        println()
        println(
            @sprintf(
                "Summary: mean loss %.3e | median %.3e | A %.4f | state %.4f",
                summary.mean_best_loss,
                summary.median_best_loss,
                summary.mean_turning_rmse,
                summary.mean_final_state_rmse_all,
            ),
        )
    end

    println()
    println("Saved outputs:")
    println("  ", LBFGS_600S_CASE_PATH)
    println("  ", LBFGS_600S_SUMMARY_PATH)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lbfgs_600s_check()
end
