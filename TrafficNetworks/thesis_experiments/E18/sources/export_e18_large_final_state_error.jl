#!/usr/bin/env julia

using Printf

include(joinpath(@__DIR__, "run_e18_large_small_inference.jl"))

const E18_FINAL_ERROR_ENSEMBLE_DIR = get(
    ENV,
    "E18_FINAL_ERROR_ENSEMBLE_DIR",
    joinpath(@__DIR__, "inference_outputs_128x4_180s"),
)
const E18_FINAL_ERROR_OUTPUT_DIR = get(
    ENV,
    "E18_FINAL_ERROR_OUTPUT_DIR",
    joinpath(E18_FINAL_ERROR_ENSEMBLE_DIR, "final_state_error"),
)

function read_latent_ensemble(path, setup::E18InferenceSetup)
    lines = readlines(path)
    isempty(lines) && error("Latent ensemble file is empty: $path")
    headers = split(lines[1], '\t')
    weight_col = findfirst(==("weight"), headers)
    weight_col === nothing && error("No weight column in $path")
    param_cols = [
        begin
            col = findfirst(==(String(name)), headers)
            col === nothing && error("Missing latent column $(String(name)) in $path")
            col
        end
        for name in setup.latent_names
    ]

    nsamples = length(lines) - 1
    params = Matrix{Float64}(undef, n_params(setup), nsamples)
    weights = Vector{Float64}(undef, nsamples)

    for (sample_idx, line) in enumerate(lines[2:end])
        fields = split(line, '\t')
        weights[sample_idx] = parse(Float64, fields[weight_col])
        for (param_idx, col) in enumerate(param_cols)
            params[param_idx, sample_idx] = parse(Float64, fields[col])
        end
    end

    weights ./= sum(weights)
    return params, weights
end

weighted_mean_parameters(params::AbstractMatrix, weights::AbstractVector) =
    vec(sum(params .* reshape(weights, 1, :), dims=2))

function write_final_state_error_csv(path, setup::E18InferenceSetup, truth_hist, mean_hist)
    mkpath(dirname(path))
    final_time_index = length(setup.control_times)
    road_rows = Dict(Int(row["id"]) => row for row in setup.roads_table)

    open(path, "w") do io
        println(io, "road_id,cell_id,n_cells,cell_fraction_center,truth_density,mean_density,abs_error")
        for road_id in sort(collect(keys(road_rows)))
            truth_values = truth_hist.road_histories[road_id][:, final_time_index]
            mean_values = mean_hist.road_histories[road_id][:, final_time_index]
            n_cells = length(truth_values)
            for cell_id in 1:n_cells
                truth_value = Float64(truth_values[cell_id])
                mean_value = Float64(mean_values[cell_id])
                fraction = (cell_id - 0.5) / n_cells
                @printf(
                    io,
                    "%d,%d,%d,%.10g,%.10g,%.10g,%.10g\n",
                    road_id,
                    cell_id,
                    n_cells,
                    fraction,
                    truth_value,
                    mean_value,
                    abs(mean_value - truth_value),
                )
            end
        end
    end
end

function write_final_state_error_summary(path, setup::E18InferenceSetup, output_csv, forward_truth_seconds, forward_mean_seconds)
    errors = Float64[]
    open(output_csv, "r") do io
        readline(io)
        for line in eachline(io)
            push!(errors, parse(Float64, split(line, ',')[end]))
        end
    end

    rows = [
        ("horizon_seconds", INFERENCE_HORIZON_SECONDS),
        ("final_time_seconds", last(OBSERVATION_TIMES_SECONDS)),
        ("roads", length(setup.roads_table)),
        ("cells", length(errors)),
        ("truth_forward_seconds", forward_truth_seconds),
        ("mean_forward_seconds", forward_mean_seconds),
        ("mean_abs_cell_error", mean(errors)),
        ("rmse_cell_error", sqrt(mean(abs2, errors))),
        ("max_abs_cell_error", maximum(errors)),
        ("q95_abs_cell_error", quantile(errors, 0.95)),
    ]
    write_summary(path, rows)
end

function main()
    mkpath(E18_FINAL_ERROR_OUTPUT_DIR)
    setup = parse_setup()
    params, weights = read_latent_ensemble(joinpath(E18_FINAL_ERROR_ENSEMBLE_DIR, "latent_ensemble.tsv"), setup)
    z_mean = weighted_mean_parameters(params, weights)
    z_true = truth_logits(setup)

    truth_seconds = @elapsed truth_hist = simulate_history(z_true, setup)
    mean_seconds = @elapsed mean_hist = simulate_history(z_mean, setup)

    output_csv = joinpath(E18_FINAL_ERROR_OUTPUT_DIR, "final_state_cell_abs_error.csv")
    write_final_state_error_csv(output_csv, setup, truth_hist, mean_hist)
    write_final_state_error_summary(
        joinpath(E18_FINAL_ERROR_OUTPUT_DIR, "final_state_error_summary.tsv"),
        setup,
        output_csv,
        truth_seconds,
        mean_seconds,
    )

    @printf("Wrote final-state error field to %s\n", output_csv)
    @printf("Truth simulation: %.2f s, mean simulation: %.2f s\n", truth_seconds, mean_seconds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
