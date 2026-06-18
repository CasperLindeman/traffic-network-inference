#!/usr/bin/env julia

using Printf
using Random
using Statistics
using LinearAlgebra
using Distributions
using ForwardDiff
using Optim
using SimulationBasedInference
using TOML
using TrafficNetworks

include(joinpath(@__DIR__, "simulate_e18_large_smoke.jl"))

const E18_INFERENCE_NETWORK_TOML = get(
    ENV,
    "E18_INFERENCE_NETWORK_TOML",
    joinpath(@__DIR__, "e18_large_pruned_sim_network.toml"),
)
const E18_INFERENCE_OUTPUT_DIR = get(
    ENV,
    "E18_INFERENCE_OUTPUT_DIR",
    joinpath(@__DIR__, "small_inference_outputs"),
)

const INFERENCE_HORIZON_SECONDS = parse(Float64, get(ENV, "E18_INFERENCE_HORIZON_SECONDS", "180.0"))
const OBSERVATION_TIMES_SECONDS = collect(30.0:30.0:INFERENCE_HORIZON_SECONDS)
const TARGET_NODE_LABELS = [
    "RB08", "RB10", "RB14", "J168",
    "RB18", "RB19", "RB20", "J245", "J246",
    "RB23", "RB24", "J411", "J416",
    "RB28", "RB29", "RB30", "J686", "J635", "J641", "J644",
]
const SENSOR_ROAD_IDS = [1047, 1052, 931, 1036, 986, 1002, 791, 798, 544, 393, 551, 137, 113, 235]
const PRIOR_SCALE = 1.25
const GENERATED_PEAK_NOISE_SIGMA = 0.08
const SIGMA_FLOOR_FRACTION = 0.15
const SIGMA_ABSOLUTE_FLOOR = 1e-3
const ESMDA_COVARIANCE_INFLATION = 1.02
const ESMDA_ENSEMBLE_SIZE = parse(Int, get(ENV, "E18_ESMDA_ENSEMBLE_SIZE", "10"))
const ESMDA_ITERS = parse(Int, get(ENV, "E18_ESMDA_ITERS", "2"))
const LBFGS_TIME_LIMIT_SECONDS = parse(Float64, get(ENV, "E18_LBFGS_TIME_LIMIT_SECONDS", "120.0"))
const LBFGS_MAXITERS = parse(Int, get(ENV, "E18_LBFGS_MAXITERS", "1"))
const LBFGS_INITIAL_ALPHA = parse(Float64, get(ENV, "E18_LBFGS_INITIAL_ALPHA", "0.005"))
const LBFGS_LINESEARCH = lowercase(get(ENV, "E18_LBFGS_LINESEARCH", "backtracking"))
const LBFGS_START_PATH = get(ENV, "E18_LBFGS_START_PATH", "")

env_flag(name, default) = lowercase(get(ENV, name, default ? "true" : "false")) in ("1", "true", "yes", "y", "on")
const RUN_ESMDA = env_flag("E18_RUN_ESMDA", true)
const RUN_LBFGS = env_flag("E18_RUN_LBFGS", false)

function lbfgs_linesearch()
    if LBFGS_LINESEARCH == "hagerzhang"
        return Optim.LineSearches.HagerZhang()
    elseif LBFGS_LINESEARCH == "backtracking"
        return Optim.LineSearches.BackTracking(order=2)
    else
        error("Unknown E18_LBFGS_LINESEARCH=$(LBFGS_LINESEARCH). Use backtracking or hagerzhang.")
    end
end

function lbfgs_initial_point(setup)
    if isempty(LBFGS_START_PATH)
        return zeros(n_params(setup))
    end

    rows = collect(eachline(LBFGS_START_PATH))
    length(rows) >= n_params(setup) + 1 || error("L-BFGS start file has too few rows: $(LBFGS_START_PATH)")
    z = zeros(n_params(setup))
    for line in rows[2:end]
        cols = split(line, '\t')
        length(cols) >= 3 || continue
        idx = parse(Int, cols[1])
        1 <= idx <= length(z) || continue
        z[idx] = parse(Float64, cols[3])
    end
    return z
end

Base.@kwdef struct LearnableRow
    junction_id::Int
    node_label::String
    node_type::String
    row::Int
    incoming_road::Int
    outgoing_roads::Vector{Int}
    cols::Vector{Int}
    param_range::UnitRange{Int}
end

Base.@kwdef struct E18InferenceSetup
    data::Dict{String, Any}
    roads_table::Vector{Any}
    junctions_table::Vector{Any}
    boundaries_table::Vector{Any}
    target_rows::Vector{LearnableRow}
    latent_names::Vector{Symbol}
    sensor_road_ids::Vector{Int}
    sensor_cell_ids::Vector{Int}
    control_times::Vector{Float64}
    contexts::Dict{Int, NamedTuple}
end

n_params(setup::E18InferenceSetup) = length(setup.latent_names)

function parse_setup()
    data = TOML.parsefile(E18_INFERENCE_NETWORK_TOML)
    roads_table = sort(data["roads"]; by=row -> Int(row["id"]))
    junctions_table = sort(data["junctions"]; by=row -> Int(row["id"]))
    boundaries_table = sort(data["boundaries"]; by=row -> Int(row["id"]))
    contexts = Dict(Int(row["id"]) => road_context(row) for row in roads_table)

    target_set = Set(TARGET_NODE_LABELS)
    target_rows = LearnableRow[]
    latent_names = Symbol[]
    next_param = 1

    for junction in junctions_table
        node_label = String(junction["node_label"])
        node_label in target_set || continue

        P = matrix_float(junction["turning_matrix"])
        incoming = Int.(junction["incoming"])
        outgoing = Int.(junction["outgoing"])

        for row in axes(P, 1)
            cols = findall(value -> value > 1e-12, vec(P[row, :]))
            length(cols) >= 2 || continue
            param_range = next_param:(next_param + length(cols) - 2)
            junction_id = Int(junction["id"])
            push!(
                target_rows,
                LearnableRow(
                    junction_id=junction_id,
                    node_label=node_label,
                    node_type=String(junction["node_type"]),
                    row=row,
                    incoming_road=incoming[row],
                    outgoing_roads=outgoing[cols],
                    cols=cols,
                    param_range=param_range,
                ),
            )
            for local_param in 1:(length(cols) - 1)
                push!(latent_names, Symbol("z_j$(junction_id)_r$(row)_k$(local_param)"))
            end
            next_param += length(cols) - 1
        end
    end

    road_by_id = Dict(Int(row["id"]) => row for row in roads_table)
    sensor_cell_ids = [
        clamp(round(Int, Int(road_by_id[road_id]["blocks"]) * 0.5), 1, Int(road_by_id[road_id]["blocks"]))
        for road_id in SENSOR_ROAD_IDS
    ]

    return E18InferenceSetup(
        data=data,
        roads_table=roads_table,
        junctions_table=junctions_table,
        boundaries_table=boundaries_table,
        target_rows=target_rows,
        latent_names=latent_names,
        sensor_road_ids=copy(SENSOR_ROAD_IDS),
        sensor_cell_ids=sensor_cell_ids,
        control_times=seconds_to_hours_local.(OBSERVATION_TIMES_SECONDS),
        contexts=contexts,
    )
end

function parameter_vector(p, setup::E18InferenceSetup)
    if p isa NamedTuple
        return [Float64(getfield(p, name)) for name in setup.latent_names]
    end
    return collect(view(p, 1:n_params(setup)))
end

function baseline_turning_matrices(setup::E18InferenceSetup)
    max_id = maximum(Int(row["id"]) for row in setup.junctions_table)
    matrices = Vector{Matrix{Float64}}(undef, max_id)
    for row in setup.junctions_table
        matrices[Int(row["id"])] = matrix_float(row["turning_matrix"])
    end
    return matrices
end

function row_logits_from_probabilities(row_values::AbstractVector)
    reference = max(Float64(row_values[end]), 1e-8)
    return [log(max(Float64(value), 1e-8) / reference) for value in row_values[1:end-1]]
end

function truth_logits(setup::E18InferenceSetup)
    matrices = baseline_turning_matrices(setup)
    z = zeros(n_params(setup))
    row_index = 0
    for row_spec in setup.target_rows
        row_index += 1
        base = matrices[row_spec.junction_id][row_spec.row, row_spec.cols]
        logits = row_logits_from_probabilities(base)
        for local_idx in eachindex(logits)
            shift = 0.70 * sin(0.47 * row_index + 0.29 * local_idx)
            z[row_spec.param_range[local_idx]] = logits[local_idx] + shift
        end
    end
    return z
end

function turning_matrices_from_target_logits(p, setup::E18InferenceSetup)
    z = parameter_vector(p, setup)
    T = promote_type(eltype(z), Float64)
    base_matrices = baseline_turning_matrices(setup)
    matrices = Vector{Matrix{T}}(undef, length(base_matrices))
    for idx in eachindex(base_matrices)
        matrices[idx] = Matrix{T}(base_matrices[idx])
    end
    for row_spec in setup.target_rows
        P = matrices[row_spec.junction_id]
        P[row_spec.row, :] .= zero(T)
        values = TrafficNetworks.stable_row_softmax(view(z, row_spec.param_range))
        P[row_spec.row, row_spec.cols] .= values
        matrices[row_spec.junction_id] = P
    end
    return matrices
end

function target_entries(Ps, setup::E18InferenceSetup)
    entries = Float64[]
    for row_spec in setup.target_rows
        append!(entries, vec(Ps[row_spec.junction_id][row_spec.row, row_spec.cols]))
    end
    return entries
end

function target_entry_samples(param_samples::AbstractMatrix, setup::E18InferenceSetup)
    n_entries = length(target_entries(turning_matrices_from_target_logits(view(param_samples, :, 1), setup), setup))
    samples = Matrix{Float64}(undef, n_entries, size(param_samples, 2))
    for sample_idx in axes(param_samples, 2)
        samples[:, sample_idx] = target_entries(
            turning_matrices_from_target_logits(view(param_samples, :, sample_idx), setup),
            setup,
        )
    end
    return samples
end

function build_network_with_turning_matrices(Ps, setup::E18InferenceSetup)
    basis_length_km = Float64(setup.data["discretization"]["basis_length_m"]) / 1000.0
    cells_per_block = Int(setup.data["discretization"]["cells_per_block"])
    state_eltype = promote_type(Float64, (eltype(P) for P in Ps)...)

    roads = Road[
        make_road(
            Int(row["id"]),
            Int(row["blocks"]),
            basis_length_km,
            cells_per_block,
            varied_initial_condition(row, setup.contexts[Int(row["id"])]),
            Int(row["speed_limit"]),
            Float64(row["lanes"]),
            state_eltype=state_eltype,
        )
        for row in setup.roads_table
    ]

    junctions = Junction[
        Junction(
            Int.(row["incoming"]),
            Int.(row["outgoing"]),
            TurningFractionRule(Ps[Int(row["id"])]),
        )
        for row in setup.junctions_table
    ]

    boundaries = Boundary[
        Boundary(Int(row["road_id"]), varied_boundary_signal(row, setup.contexts[Int(row["road_id"])]))
        for row in setup.boundaries_table
    ]

    return RoadNetwork(
        roads,
        junctions,
        boundaries,
        seconds_to_hours_local(INFERENCE_HORIZON_SECONDS),
        Float64(setup.data["simulation"]["cfl"]),
    )
end

function simulate_history(p, setup::E18InferenceSetup)
    Ps = p isa AbstractVector || p isa NamedTuple ? turning_matrices_from_target_logits(p, setup) : p
    net = build_network_with_turning_matrices(Ps, setup)
    return simulate!(net; times=setup.control_times)
end

function flatten_sensor_observations(hist, setup::E18InferenceSetup)
    chunks = [
        vec(hist.road_histories[road_id][cell_id, :])
        for (road_id, cell_id) in zip(setup.sensor_road_ids, setup.sensor_cell_ids)
    ]
    T = promote_type((eltype(chunk) for chunk in chunks)...)
    values = T[]
    for chunk in chunks
        append!(values, chunk)
    end
    return values
end

simulator(p, setup::E18InferenceSetup) =
    flatten_sensor_observations(simulate_history(p, setup), setup)

function simulate_ensemble(param_ensemble::AbstractMatrix, setup::E18InferenceSetup)
    n_obs = length(setup.sensor_road_ids) * length(setup.control_times)
    ensemble_size = size(param_ensemble, 2)
    predictions = Matrix{Float64}(undef, n_obs, ensemble_size)

    for member in 1:ensemble_size
        predictions[:, member] = simulator(view(param_ensemble, :, member), setup)
    end

    return predictions
end

function build_model_prior(setup::E18InferenceSetup)
    dists = ntuple(_ -> Normal(0.0, PRIOR_SCALE), n_params(setup))
    kwargs = NamedTuple{Tuple(setup.latent_names)}(dists)
    return prior(; kwargs...)
end

function posterior_predictive_mean(sol, weights)
    pred_ens = Array(get_observables(sol).y)
    return TrafficNetworks.prediction_ensemble_mean(pred_ens, weights)
end

function map_loss(z, y_obs, sigma_model, setup::E18InferenceSetup)
    residual = (simulator(z, setup) .- y_obs) ./ sigma_model
    return 0.5 * sum(abs2, residual) + 0.5 * sum(abs2, z ./ PRIOR_SCALE)
end

function ensemble_space_esmda_update(
    param_ensemble::AbstractMatrix,
    pred_ensemble::AbstractMatrix,
    y_obs::AbstractVector,
    sigma_model::AbstractVector,
    alpha::Real,
    rng::AbstractRNG,
)
    ensemble_size = size(param_ensemble, 2)
    param_mean = mean(param_ensemble; dims=2)
    pred_mean = mean(pred_ensemble; dims=2)

    param_anom = param_ensemble .- param_mean
    pred_anom = pred_ensemble .- pred_mean

    sqrt_inv_r = 1.0 ./ (sqrt(Float64(alpha)) .* sigma_model)
    pred_whitened = pred_anom .* sqrt_inv_r

    perturbed_obs = y_obs .+ sqrt(Float64(alpha)) .* sigma_model .* randn(rng, length(y_obs), ensemble_size)
    innovation_whitened = (perturbed_obs .- pred_ensemble) .* sqrt_inv_r

    system_matrix = pred_whitened' * pred_whitened + (ensemble_size - 1) * I
    weights = system_matrix \ (pred_whitened' * innovation_whitened)

    return param_ensemble + param_anom * weights
end

function esmda_alpha_schedule(maxiters::Integer)
    @assert maxiters >= 1 "ESMDA requires at least one assimilation iteration."
    schedule = fill(Float64(maxiters), Int(maxiters))
    @assert isapprox(sum(1.0 ./ schedule), 1.0; atol=1e-12, rtol=1e-12)
    return schedule
end

function inflate_ensemble_covariance(param_ensemble::AbstractMatrix, covariance_inflation::Real)
    inflation = Float64(covariance_inflation)
    @assert inflation >= 1.0 "Covariance inflation factor must be at least 1."
    inflation == 1.0 && return param_ensemble

    ensemble_mean = mean(param_ensemble; dims=2)
    return ensemble_mean .+ sqrt(inflation) .* (param_ensemble .- ensemble_mean)
end

function write_summary(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(["quantity", "value"], '\t'))
        for row in rows
            println(io, join((row[1], row[2]), '\t'))
        end
    end
end

function tsv_cell(value)
    if value isa AbstractFloat
        return @sprintf("%.10g", Float64(value))
    elseif value isa AbstractVector
        return join(value, " ")
    else
        return string(value)
    end
end

function write_tsv(path, headers, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(headers, '\t'))
        for row in rows
            println(io, join(tsv_cell.(row), '\t'))
        end
    end
end

function target_entry_metadata(setup::E18InferenceSetup, P_true)
    rows = Any[]
    entry_index = 0
    target_row_index = 0
    for row_spec in setup.target_rows
        target_row_index += 1
        for (local_col_index, matrix_col) in enumerate(row_spec.cols)
            entry_index += 1
            push!(
                rows,
                (
                    entry_index,
                    target_row_index,
                    row_spec.node_label,
                    row_spec.node_type,
                    row_spec.junction_id,
                    row_spec.row,
                    row_spec.incoming_road,
                    row_spec.outgoing_roads[local_col_index],
                    local_col_index,
                    matrix_col,
                    Float64(P_true[row_spec.junction_id][row_spec.row, matrix_col]),
                ),
            )
        end
    end
    return rows
end

function observation_metadata(setup::E18InferenceSetup, observations)
    rows = Any[]
    obs_index = 0
    for (sensor_index, (road_id, cell_id)) in enumerate(zip(setup.sensor_road_ids, setup.sensor_cell_ids))
        for (time_index, seconds) in enumerate(OBSERVATION_TIMES_SECONDS)
            obs_index += 1
            push!(
                rows,
                (
                    obs_index,
                    sensor_index,
                    road_id,
                    cell_id,
                    time_index,
                    seconds,
                    observations.y_true[obs_index],
                    observations.y_obs[obs_index],
                    observations.sigma_model[obs_index],
                ),
            )
        end
    end
    return rows
end

function write_static_diagnostics(output_dir, setup::E18InferenceSetup, observations, P_true, z_true)
    write_tsv(
        joinpath(output_dir, "target_rows.tsv"),
        [
            "target_row_index", "node_label", "node_type", "junction_id", "row",
            "incoming_road", "outgoing_roads", "active_matrix_columns",
            "latent_parameter_indices", "latent_parameter_names", "true_turning_fractions",
        ],
        [
            (
                row_index,
                row_spec.node_label,
                row_spec.node_type,
                row_spec.junction_id,
                row_spec.row,
                row_spec.incoming_road,
                row_spec.outgoing_roads,
                row_spec.cols,
                collect(row_spec.param_range),
                String.(setup.latent_names[row_spec.param_range]),
                Float64.(P_true[row_spec.junction_id][row_spec.row, row_spec.cols]),
            )
            for (row_index, row_spec) in enumerate(setup.target_rows)
        ],
    )

    write_tsv(
        joinpath(output_dir, "target_entries_truth.tsv"),
        [
            "entry_index", "target_row_index", "node_label", "node_type", "junction_id",
            "row", "incoming_road", "outgoing_road", "local_outgoing_index",
            "matrix_column", "true_turning_fraction",
        ],
        target_entry_metadata(setup, P_true),
    )

    write_tsv(
        joinpath(output_dir, "latent_truth.tsv"),
        ["parameter_index", "parameter_name", "true_logit"],
        [(idx, String(name), z_true[idx]) for (idx, name) in enumerate(setup.latent_names)],
    )

    write_tsv(
        joinpath(output_dir, "observations.tsv"),
        [
            "observation_index", "sensor_index", "road_id", "cell_id", "time_index",
            "time_seconds", "true_density", "observed_density", "sigma_model",
        ],
        observation_metadata(setup, observations),
    )
end

function write_esmda_artifacts(output_dir, setup::E18InferenceSetup, observations, P_true, z_true, esmda)
    write_static_diagnostics(output_dir, setup, observations, P_true, z_true)

    param_samples = esmda.param_samples
    weights = esmda.weights
    fraction_samples = esmda.fraction_samples
    pred_samples = esmda.pred_samples
    entry_meta = target_entry_metadata(setup, P_true)
    obs_meta = observation_metadata(setup, observations)

    write_tsv(
        joinpath(output_dir, "latent_ensemble.tsv"),
        vcat(["sample", "weight"], String.(setup.latent_names)),
        [
            vcat(Any[sample_idx, weights[sample_idx]], Any[param_samples[param_idx, sample_idx] for param_idx in axes(param_samples, 1)])
            for sample_idx in axes(param_samples, 2)
        ],
    )

    write_tsv(
        joinpath(output_dir, "turning_fraction_ensemble.tsv"),
        [
            "sample", "weight", "entry_index", "target_row_index", "node_label",
            "node_type", "junction_id", "row", "incoming_road", "outgoing_road",
            "local_outgoing_index", "matrix_column", "true_turning_fraction",
            "sample_turning_fraction",
        ],
        [
            (
                sample_idx,
                weights[sample_idx],
                meta[1],
                meta[2],
                meta[3],
                meta[4],
                meta[5],
                meta[6],
                meta[7],
                meta[8],
                meta[9],
                meta[10],
                meta[11],
                fraction_samples[entry_idx, sample_idx],
            )
            for sample_idx in axes(fraction_samples, 2)
            for (entry_idx, meta) in enumerate(entry_meta)
        ],
    )

    write_tsv(
        joinpath(output_dir, "turning_fraction_summary.tsv"),
        [
            "entry_index", "target_row_index", "node_label", "node_type", "junction_id",
            "row", "incoming_road", "outgoing_road", "local_outgoing_index",
            "matrix_column", "true_turning_fraction", "mean", "median", "q05", "q25",
            "q75", "q95", "mean_abs_error", "width90", "covered90",
        ],
        [
            (
                meta[1],
                meta[2],
                meta[3],
                meta[4],
                meta[5],
                meta[6],
                meta[7],
                meta[8],
                meta[9],
                meta[10],
                meta[11],
                esmda.entry_mean[entry_idx],
                esmda.entry_median[entry_idx],
                esmda.entry_q05[entry_idx],
                esmda.entry_q25[entry_idx],
                esmda.entry_q75[entry_idx],
                esmda.entry_q95[entry_idx],
                abs(esmda.entry_mean[entry_idx] - meta[11]),
                esmda.entry_q95[entry_idx] - esmda.entry_q05[entry_idx],
                esmda.entry_q05[entry_idx] <= meta[11] <= esmda.entry_q95[entry_idx],
            )
            for (entry_idx, meta) in enumerate(entry_meta)
        ],
    )

    write_tsv(
        joinpath(output_dir, "prediction_ensemble.tsv"),
        [
            "sample", "weight", "observation_index", "sensor_index", "road_id",
            "cell_id", "time_index", "time_seconds", "true_density",
            "observed_density", "sigma_model", "sample_prediction",
        ],
        [
            (
                sample_idx,
                weights[sample_idx],
                meta[1],
                meta[2],
                meta[3],
                meta[4],
                meta[5],
                meta[6],
                meta[7],
                meta[8],
                meta[9],
                pred_samples[obs_idx, sample_idx],
            )
            for sample_idx in axes(pred_samples, 2)
            for (obs_idx, meta) in enumerate(obs_meta)
        ],
    )

    write_tsv(
        joinpath(output_dir, "prediction_summary.tsv"),
        [
            "observation_index", "sensor_index", "road_id", "cell_id", "time_index",
            "time_seconds", "true_density", "observed_density", "sigma_model",
            "mean", "median", "q05", "q25", "q75", "q95",
            "mean_error_vs_true", "mean_error_vs_observed", "width90", "covered_true90",
        ],
        [
            (
                meta[1],
                meta[2],
                meta[3],
                meta[4],
                meta[5],
                meta[6],
                meta[7],
                meta[8],
                meta[9],
                esmda.pred_mean[obs_idx],
                esmda.pred_median[obs_idx],
                esmda.pred_q05[obs_idx],
                esmda.pred_q25[obs_idx],
                esmda.pred_q75[obs_idx],
                esmda.pred_q95[obs_idx],
                esmda.pred_mean[obs_idx] - meta[7],
                esmda.pred_mean[obs_idx] - meta[8],
                esmda.pred_q95[obs_idx] - esmda.pred_q05[obs_idx],
                esmda.pred_q05[obs_idx] <= meta[7] <= esmda.pred_q95[obs_idx],
            )
            for (obs_idx, meta) in enumerate(obs_meta)
        ],
    )
end

function run_esmda_inference(setup, y_obs, sigma_model, y_true, P_true; seed=20260615)
    rng = MersenneTwister(seed)
    param_samples = PRIOR_SCALE .* randn(rng, n_params(setup), ESMDA_ENSEMBLE_SIZE)
    alphas = esmda_alpha_schedule(ESMDA_ITERS)

    solve_seconds = @elapsed begin
        for iter in 1:ESMDA_ITERS
            pred_ensemble = simulate_ensemble(param_samples, setup)
            param_samples = ensemble_space_esmda_update(param_samples, pred_ensemble, y_obs, sigma_model, alphas[iter], rng)
            if iter < ESMDA_ITERS
                param_samples = inflate_ensemble_covariance(param_samples, ESMDA_COVARIANCE_INFLATION)
            end
        end
    end

    weights = fill(1.0 / size(param_samples, 2), size(param_samples, 2))
    fraction_samples = target_entry_samples(param_samples, setup)
    entry_true = target_entries(P_true, setup)
    entry_mean, entry_median, entry_ci05, entry_ci95 = TrafficNetworks.sample_summary(fraction_samples, weights)
    entry_q25 = [TrafficNetworks.weighted_quantile(vec(fraction_samples[i, :]), weights, 0.25) for i in axes(fraction_samples, 1)]
    entry_q75 = [TrafficNetworks.weighted_quantile(vec(fraction_samples[i, :]), weights, 0.75) for i in axes(fraction_samples, 1)]
    pred_samples = simulate_ensemble(param_samples, setup)
    pred_mean, pred_median, pred_ci05, pred_ci95 = TrafficNetworks.sample_summary(pred_samples, weights)
    pred_q25 = [TrafficNetworks.weighted_quantile(vec(pred_samples[i, :]), weights, 0.25) for i in axes(pred_samples, 1)]
    pred_q75 = [TrafficNetworks.weighted_quantile(vec(pred_samples[i, :]), weights, 0.75) for i in axes(pred_samples, 1)]
    turning_summary = TrafficNetworks.recovery_summary(entry_mean, entry_true; lower=entry_ci05, upper=entry_ci95)

    return (
        solve_seconds=solve_seconds,
        param_samples=param_samples,
        weights=weights,
        fraction_samples=fraction_samples,
        pred_samples=pred_samples,
        entry_mean=entry_mean,
        entry_median=entry_median,
        entry_q05=entry_ci05,
        entry_q25=entry_q25,
        entry_q75=entry_q75,
        entry_q95=entry_ci95,
        pred_mean=pred_mean,
        pred_median=pred_median,
        pred_q05=pred_ci05,
        pred_q25=pred_q25,
        pred_q75=pred_q75,
        pred_q95=pred_ci95,
        turning_rmse=turning_summary.rmse,
        turning_mae=turning_summary.mean_abs,
        mean_width=turning_summary.mean_interval_width,
        coverage=turning_summary.interval_coverage_count,
        n_entries=length(entry_true),
        fit_rmse=TrafficNetworks.rmse(pred_mean, y_obs),
        predictive_rmse=TrafficNetworks.rmse(pred_mean, y_true),
        predictive_coverage=sum((pred_ci05 .<= y_true) .& (y_true .<= pred_ci95)),
        n_observations=length(y_true),
    )
end

function write_lbfgs_checkpoint(output_dir, setup::E18InferenceSetup, z, loss, elapsed_seconds, stage)
    write_tsv(
        joinpath(output_dir, "lbfgs_best_checkpoint.tsv"),
        ["parameter_index", "parameter_name", "z", "best_loss", "elapsed_seconds", "stage"],
        [
            (
                idx,
                String(name),
                Float64(z[idx]),
                Float64(loss),
                Float64(elapsed_seconds),
                stage,
            )
            for (idx, name) in enumerate(setup.latent_names)
        ],
    )
end

function write_lbfgs_artifacts(output_dir, setup::E18InferenceSetup, observations, P_true, z_true, lbfgs)
    write_static_diagnostics(output_dir, setup, observations, P_true, z_true)

    z_best = lbfgs.z_best
    P_est = turning_matrices_from_target_logits(z_best, setup)
    y_est = lbfgs.prediction
    entry_true = target_entries(P_true, setup)
    entry_est = target_entries(P_est, setup)
    entry_meta = target_entry_metadata(setup, P_true)
    obs_meta = observation_metadata(setup, observations)

    write_tsv(
        joinpath(output_dir, "lbfgs_latent_map.tsv"),
        ["parameter_index", "parameter_name", "true_logit", "map_logit", "error"],
        [
            (
                idx,
                String(name),
                z_true[idx],
                z_best[idx],
                z_best[idx] - z_true[idx],
            )
            for (idx, name) in enumerate(setup.latent_names)
        ],
    )

    write_tsv(
        joinpath(output_dir, "lbfgs_turning_fraction_map.tsv"),
        [
            "entry_index", "target_row_index", "node_label", "node_type",
            "junction_id", "row", "incoming_road", "outgoing_road",
            "local_outgoing_index", "matrix_column", "true_turning_fraction",
            "map_turning_fraction", "abs_error",
        ],
        [
            (
                meta[1],
                meta[2],
                meta[3],
                meta[4],
                meta[5],
                meta[6],
                meta[7],
                meta[8],
                meta[9],
                meta[10],
                meta[11],
                entry_est[entry_idx],
                abs(entry_est[entry_idx] - entry_true[entry_idx]),
            )
            for (entry_idx, meta) in enumerate(entry_meta)
        ],
    )

    write_tsv(
        joinpath(output_dir, "lbfgs_prediction_map.tsv"),
        [
            "observation_index", "sensor_index", "road_id", "cell_id",
            "time_index", "time_seconds", "true_density", "observed_density",
            "sigma_model", "map_prediction", "error_vs_true",
            "error_vs_observed", "normalized_residual",
        ],
        [
            (
                meta[1],
                meta[2],
                meta[3],
                meta[4],
                meta[5],
                meta[6],
                meta[7],
                meta[8],
                meta[9],
                y_est[obs_idx],
                y_est[obs_idx] - meta[7],
                y_est[obs_idx] - meta[8],
                (y_est[obs_idx] - meta[8]) / meta[9],
            )
            for (obs_idx, meta) in enumerate(obs_meta)
        ],
    )

    write_tsv(
        joinpath(output_dir, "lbfgs_trace.tsv"),
        ["trace_index", "objective_value"],
        [(idx, value) for (idx, value) in enumerate(lbfgs.losses)],
    )
end

function run_lbfgs_probe(setup, y_obs, sigma_model, y_true, P_true; output_dir=E18_INFERENCE_OUTPUT_DIR)
    z0 = lbfgs_initial_point(setup)
    loss_fn = z -> map_loss(z, y_obs, sigma_model, setup)
    objective_seconds = @elapsed initial_loss = loss_fn(z0)
    gradient = zeros(n_params(setup))
    gradient_seconds = @elapsed ForwardDiff.gradient!(gradient, loss_fn, z0)

    @printf("L-BFGS initial loss %.6g, objective %.2f s, gradient %.2f s\n",
        initial_loss, objective_seconds, gradient_seconds)
    flush(stdout)

    best_loss = Ref(Float64(initial_loss))
    best_z = Ref(copy(z0))
    optimize_started_at = Ref(time())
    objective_evals = Ref(1)
    gradient_evals = Ref(1)
    write_lbfgs_checkpoint(output_dir, setup, z0, best_loss[], 0.0, "initial")

    function fg!(F, G, z)
        if G !== nothing
            seconds = @elapsed ForwardDiff.gradient!(G, loss_fn, z)
            gradient_evals[] += 1
            @printf("[lbfgs] gradient eval %d finished in %.2f s, elapsed %.2f s\n",
                gradient_evals[], seconds, time() - optimize_started_at[])
            flush(stdout)
        end
        if F !== nothing
            seconds = @elapsed value = loss_fn(z)
            objective_evals[] += 1
            elapsed = time() - optimize_started_at[]
            @printf("[lbfgs] objective eval %d: %.6g in %.2f s, elapsed %.2f s\n",
                objective_evals[], value, seconds, elapsed)
            flush(stdout)
            if isfinite(value) && value < best_loss[]
                best_loss[] = Float64(value)
                best_z[] = copy(Float64.(z))
                write_lbfgs_checkpoint(output_dir, setup, best_z[], best_loss[], elapsed, "objective_eval")
            end
            return value
        end
        return nothing
    end

    result = nothing
    optimize_started_at[] = time()
    optimize_seconds = @elapsed result = Optim.optimize(
        Optim.only_fg!(fg!),
        z0,
        Optim.LBFGS(
            alphaguess=Optim.LineSearches.InitialStatic(alpha=LBFGS_INITIAL_ALPHA),
            linesearch=lbfgs_linesearch(),
        ),
        Optim.Options(
            iterations=LBFGS_MAXITERS,
            store_trace=true,
            show_trace=true,
            time_limit=LBFGS_TIME_LIMIT_SECONDS,
        ),
    )

    z_best = Optim.minimum(result) <= best_loss[] ? Optim.minimizer(result) : best_z[]
    P_est = turning_matrices_from_target_logits(z_best, setup)
    y_est = simulator(z_best, setup)
    entry_true = target_entries(P_true, setup)
    entry_est = target_entries(P_est, setup)
    final_grad = ForwardDiff.gradient(loss_fn, z_best)
    losses = [tr.value for tr in Optim.trace(result) if tr.value !== nothing]

    return (
        z_best=copy(z_best),
        prediction=Float64.(y_est),
        losses=Float64.(losses),
        objective_seconds=objective_seconds,
        gradient_seconds=gradient_seconds,
        optimize_seconds=optimize_seconds,
        initial_loss=initial_loss,
        final_loss=loss_fn(z_best),
        optimizer_minimum=Optim.minimum(result),
        best_checkpoint_loss=best_loss[],
        iterations=Optim.iterations(result),
        converged=Optim.converged(result),
        turning_rmse=TrafficNetworks.rmse(entry_est, entry_true),
        turning_mae=mean(abs.(entry_est .- entry_true)),
        fit_rmse=TrafficNetworks.rmse(y_est, y_obs),
        predictive_rmse=TrafficNetworks.rmse(y_est, y_true),
        initial_gradient_norm=sqrt(sum(abs2, gradient)),
        final_gradient_norm=sqrt(sum(abs2, final_grad)),
        objective_evals=objective_evals[],
        gradient_evals=gradient_evals[],
    )
end

function run_small_inference()
    mkpath(E18_INFERENCE_OUTPUT_DIR)
    setup = parse_setup()
    rng = MersenneTwister(20260615)

    z_true = truth_logits(setup)
    P_true = turning_matrices_from_target_logits(z_true, setup)

    forward_seconds = @elapsed y_true = simulator(z_true, setup)
    observations = TrafficNetworks.generate_physical_observations(
        y_true,
        GENERATED_PEAK_NOISE_SIGMA,
        rng;
        floor_fraction=SIGMA_FLOOR_FRACTION,
        absolute_floor=SIGMA_ABSOLUTE_FLOOR,
    )

    @printf("E18 large small inference\n")
    @printf("horizon: %.0f s\n", INFERENCE_HORIZON_SECONDS)
    @printf("targets: %d nodes, %d learned rows, %d logits\n", length(TARGET_NODE_LABELS), length(setup.target_rows), n_params(setup))
    @printf("sensors: %d roads x %d times = %d observations\n", length(setup.sensor_road_ids), length(setup.control_times), length(y_true))
    @printf("one forward pass: %.2f s\n", forward_seconds)
    flush(stdout)

    summary_rows = Any[
        ("horizon_seconds", INFERENCE_HORIZON_SECONDS),
        ("observation_times_seconds", join(OBSERVATION_TIMES_SECONDS, " ")),
        ("target_nodes", length(TARGET_NODE_LABELS)),
        ("learned_rows", length(setup.target_rows)),
        ("latent_parameters", n_params(setup)),
        ("sensor_roads", length(setup.sensor_road_ids)),
        ("observations", length(y_true)),
        ("forward_seconds", forward_seconds),
        ("prior_scale", PRIOR_SCALE),
        ("generated_peak_noise_sigma", GENERATED_PEAK_NOISE_SIGMA),
        ("sigma_floor_fraction", SIGMA_FLOOR_FRACTION),
        ("sigma_absolute_floor", SIGMA_ABSOLUTE_FLOOR),
        ("esmda_covariance_inflation", ESMDA_COVARIANCE_INFLATION),
    ]

    esmda = nothing
    if RUN_ESMDA
        esmda = run_esmda_inference(setup, observations.y_obs, observations.sigma_model, observations.y_true, P_true)
        @printf("ESMDA %dx%d: %.2f s, turning RMSE %.4f, pred RMSE %.5f\n",
            ESMDA_ENSEMBLE_SIZE, ESMDA_ITERS, esmda.solve_seconds, esmda.turning_rmse, esmda.predictive_rmse)
        append!(
            summary_rows,
            [
                ("esmda_ensemble", ESMDA_ENSEMBLE_SIZE),
                ("esmda_steps", ESMDA_ITERS),
                ("esmda_seconds", esmda.solve_seconds),
                ("esmda_turning_rmse", esmda.turning_rmse),
                ("esmda_turning_mae", esmda.turning_mae),
                ("esmda_mean_width", esmda.mean_width),
                ("esmda_coverage", "$(esmda.coverage)/$(esmda.n_entries)"),
                ("esmda_fit_rmse", esmda.fit_rmse),
                ("esmda_predictive_rmse", esmda.predictive_rmse),
                ("esmda_predictive_coverage", "$(esmda.predictive_coverage)/$(esmda.n_observations)"),
            ],
        )
        write_esmda_artifacts(E18_INFERENCE_OUTPUT_DIR, setup, observations, P_true, z_true, esmda)
    else
        @printf("ESMDA skipped by E18_RUN_ESMDA=false\n")
        push!(summary_rows, ("esmda_status", "skipped"))
    end
    flush(stdout)

    lbfgs = nothing
    if RUN_LBFGS
        lbfgs = run_lbfgs_probe(setup, observations.y_obs, observations.sigma_model, observations.y_true, P_true)
        @printf("L-BFGS probe: obj %.2f s, grad %.2f s, optimize %.2f s, iter %d, turning RMSE %.4f\n",
            lbfgs.objective_seconds, lbfgs.gradient_seconds, lbfgs.optimize_seconds, lbfgs.iterations, lbfgs.turning_rmse)
        append!(
            summary_rows,
            [
                ("lbfgs_maxiters", LBFGS_MAXITERS),
                ("lbfgs_time_limit_seconds", LBFGS_TIME_LIMIT_SECONDS),
                ("lbfgs_initial_alpha", LBFGS_INITIAL_ALPHA),
                ("lbfgs_linesearch", LBFGS_LINESEARCH),
                ("lbfgs_start_path", isempty(LBFGS_START_PATH) ? "prior_mean" : LBFGS_START_PATH),
                ("lbfgs_objective_seconds", lbfgs.objective_seconds),
                ("lbfgs_gradient_seconds", lbfgs.gradient_seconds),
                ("lbfgs_optimize_seconds", lbfgs.optimize_seconds),
                ("lbfgs_iterations", lbfgs.iterations),
                ("lbfgs_converged", lbfgs.converged),
                ("lbfgs_initial_loss", lbfgs.initial_loss),
                ("lbfgs_final_loss", lbfgs.final_loss),
                ("lbfgs_optimizer_minimum", lbfgs.optimizer_minimum),
                ("lbfgs_best_checkpoint_loss", lbfgs.best_checkpoint_loss),
                ("lbfgs_initial_gradient_norm", lbfgs.initial_gradient_norm),
                ("lbfgs_final_gradient_norm", lbfgs.final_gradient_norm),
                ("lbfgs_objective_evals", lbfgs.objective_evals),
                ("lbfgs_gradient_evals", lbfgs.gradient_evals),
                ("lbfgs_turning_rmse", lbfgs.turning_rmse),
                ("lbfgs_turning_mae", lbfgs.turning_mae),
                ("lbfgs_fit_rmse", lbfgs.fit_rmse),
                ("lbfgs_predictive_rmse", lbfgs.predictive_rmse),
            ],
        )
        write_lbfgs_artifacts(E18_INFERENCE_OUTPUT_DIR, setup, observations, P_true, z_true, lbfgs)
    else
        @printf("L-BFGS skipped by E18_RUN_LBFGS=false\n")
        push!(summary_rows, ("lbfgs_status", "skipped"))
    end

    write_summary(
        joinpath(E18_INFERENCE_OUTPUT_DIR, "small_inference_summary.tsv"),
        summary_rows,
    )

    return (setup=setup, observations=observations, esmda=esmda, lbfgs=lbfgs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_small_inference()
end
