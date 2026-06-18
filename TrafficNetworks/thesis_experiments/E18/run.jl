#!/usr/bin/env julia

include(joinpath(@__DIR__, "plots.jl"))

const REQUIRED_E18_ARTIFACTS = [
    joinpath(E18_DIR, "sources", "source_json", "E18_1.json"),
    joinpath(E18_DIR, "sources", "source_json", "E18_11.json"),
    joinpath(E18_DIR, "sources", "build_e18_large_graph.py"),
    joinpath(E18_DIR, "sources", "build_e18_large_pruned_simulation_network.py"),
    joinpath(E18_DIR, "sources", "run_e18_large_small_inference.jl"),
    joinpath(E18_DIR, "sources", "export_e18_large_final_state_error.jl"),
    joinpath(E18_DIR, "outputs", "graph", "e18_large_pruned_sim_network.toml"),
    joinpath(E18_DIR, "outputs", "graph", "e18_large_pruned_sim_summary.csv"),
    joinpath(E18_DIR, "outputs", "selection", "proposed_inference_targets.tsv"),
    joinpath(E18_DIR, "outputs", "selection", "proposed_sensor_locations.tsv"),
    joinpath(E18_DIR, "outputs", "inference_64x2_180s", "small_inference_summary.tsv"),
    joinpath(E18_DIR, "outputs", "inference_128x4_180s", "small_inference_summary.tsv"),
    joinpath(E18_DIR, "outputs", "inference_128x4_180s", "final_state_error", "final_state_error_summary.tsv"),
    joinpath(E18_DIR, "outputs", "lbfgs_prior_mean_backtracking_180s", "lbfgs_best_checkpoint.tsv"),
    joinpath(E18_DIR, "outputs", "lbfgs_checkpoint_backtracking_alpha0050_retry_180s", "small_inference_summary.tsv"),
    joinpath(E18_DIR, "outputs", "lbfgs_checkpoint_backtracking_alpha0050_retry_180s", "lbfgs_trace.tsv"),
    joinpath(E18_DIR, "outputs", "lbfgs_checkpoint_backtracking_alpha0050_retry_180s", "lbfgs_turning_fraction_map.tsv"),
    joinpath(E18_DIR, "outputs", "lbfgs_checkpoint_backtracking_alpha0050_retry_180s", "lbfgs_prediction_map.tsv"),
]

function read_key_value_table(path)
    rows = Dict{String, String}()
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split(line, '\t')
            length(fields) >= 2 || continue
            rows[String(fields[1])] = String(fields[2])
        end
    end
    return rows
end

function validate_e18_artifacts()
    missing = filter(!isfile, REQUIRED_E18_ARTIFACTS)
    if !isempty(missing)
        error("Missing E18 experiment artifacts:\n" * join(missing, "\n"))
    end
    validate_e18_figures()
    return true
end

function print_esmda_summary(label, relpath)
    summary = read_key_value_table(joinpath(E18_DIR, relpath, "small_inference_summary.tsv"))
    println(label)
    println("  runtime: ", summary["esmda_seconds"], " s")
    println("  turning RMSE: ", summary["esmda_turning_rmse"])
    println("  predictive RMSE: ", summary["esmda_predictive_rmse"])
    println("  turning coverage: ", summary["esmda_coverage"])
    println("  predictive coverage: ", summary["esmda_predictive_coverage"])
end

function print_lbfgs_summary(label, relpath)
    summary = read_key_value_table(joinpath(E18_DIR, relpath, "small_inference_summary.tsv"))
    println(label)
    println("  runtime: ", summary["lbfgs_optimize_seconds"], " s")
    println("  iterations: ", summary["lbfgs_iterations"])
    println("  turning RMSE: ", summary["lbfgs_turning_rmse"])
    println("  predictive RMSE: ", summary["lbfgs_predictive_rmse"])
end

function main()
    sync_e18_figures()
    validate_e18_artifacts()

    println("E18 corridor thesis experiment artifacts are complete.")
    print_esmda_summary("ESMDA 64x2, 180 s", joinpath("outputs", "inference_64x2_180s"))
    print_esmda_summary("ESMDA 128x4, 180 s", joinpath("outputs", "inference_128x4_180s"))
    print_lbfgs_summary("L-BFGS MAP, 180 s", joinpath("outputs", "lbfgs_checkpoint_backtracking_alpha0050_retry_180s"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
