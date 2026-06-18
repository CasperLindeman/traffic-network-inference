ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_STRUCTURE"] = "1"

include(joinpath(@__DIR__, "sensor_layout_helpers.jl"))

using Printf
using Statistics
using LinearAlgebra
using ForwardDiff
using Plots

const DENSITY_NOISE_OUTPUT_DIR = joinpath(
    @__DIR__,
    "..",
    "outputs",
    "pilot",
)

const DENSITY_NOISE_FULL_OUTPUT_DIR = joinpath(
    @__DIR__,
    "..",
    "outputs",
)

const DENSITY_NOISE_REGIME = MultiScenarioDataRegime("multi_scenario_12x", 1, 12)
const DENSITY_NOISE_FLOOR_FRACTION = 0.15
const DENSITY_NOISE_ENSEMBLE_SIZE = 192
const DENSITY_NOISE_ESMDA_MAXITERS = 6
const DENSITY_NOISE_ADAM_SECONDS = 180.0
const DENSITY_NOISE_ADAM_PRIOR_SCALE = MULTI_SCENARIO_PRIOR_SCALE
const DENSITY_NOISE_LEVELS = [0.01, 0.40]
const DENSITY_NOISE_SEEDS = [1]
const DENSITY_NOISE_FULL_LEVELS = [0.01, 0.08, 0.20, 0.40]
const DENSITY_NOISE_FULL_SEEDS = [1, 2, 3]
const MINIMAL_INTERNAL_ROADS = [17, 24]

function density_specs()
    c5 = only([spec for spec in step1a_config_specs() if spec.config == "C5"])
    return [
        (density="dense", description="9 roads, 4 sensors per road, original baseline", observed_roads=copy(DEFAULT_OBSERVED_ROADS), sparse_config=nothing),
        (density="sparse", description="9 roads, 1 midpoint sensor per road", observed_roads=copy(DEFAULT_OBSERVED_ROADS), sparse_config=c5),
        (density="minimal", description="2 internal connector roads, 1 midpoint sensor per road", observed_roads=copy(MINIMAL_INTERNAL_ROADS), sparse_config=:minimal_midpoint),
    ]
end

function minimal_sensor_cells(; peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA)
    base_setup = square_single_scenario_setup(peak_noise_sigma=peak_noise_sigma)
    return Dict(
        road_id => [middle_cell(road_cell_count(base_setup, road_id))]
        for road_id in MINIMAL_INTERNAL_ROADS
    )
end

function dataset_for_density(spec, noise_sigma::Real)
    if spec.density == "dense"
        return build_multi_scenario_dataset(DENSITY_NOISE_REGIME; peak_noise_sigma=noise_sigma)
    elseif spec.density == "sparse"
        cells = sensor_cells_for_config(spec.sparse_config; observed_roads=spec.observed_roads, peak_noise_sigma=noise_sigma)
        return build_sparse_multi_scenario_dataset(
            DENSITY_NOISE_REGIME,
            cells;
            observed_roads=spec.observed_roads,
            peak_noise_sigma=noise_sigma,
        )
    elseif spec.density == "minimal"
        cells = minimal_sensor_cells(; peak_noise_sigma=noise_sigma)
        return build_sparse_multi_scenario_dataset(
            DENSITY_NOISE_REGIME,
            cells;
            observed_roads=spec.observed_roads,
            peak_noise_sigma=noise_sigma,
        )
    end

    error("Unknown density $(spec.density).")
end

function run_adam_map_multi_scenario_timed(
    y_obs::AbstractVector,
    dataset::MultiScenarioDataset,
    sigma_model::AbstractVector;
    prior_scale=DENSITY_NOISE_ADAM_PRIOR_SCALE,
    z0=zeros(N_PARAMS),
    learning_rate=MULTI_SCENARIO_ADAM_LEARNING_RATE,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    grad_clip=MULTI_SCENARIO_ADAM_GRAD_CLIP,
    decay_start=MULTI_SCENARIO_ADAM_DECAY_START,
    final_lr_scale=MULTI_SCENARIO_ADAM_FINAL_LR_SCALE,
    time_limit_seconds=DENSITY_NOISE_ADAM_SECONDS,
)
    loss_fn = z -> map_loss_dataset_weighted_forwarddiff(z, y_obs, dataset, sigma_model; prior_scale=prior_scale)
    z = Float64.(collect(z0))
    best_z = copy(z)
    best_loss = Inf
    best_iter = 0

    m = zeros(length(z))
    v = zeros(length(z))
    grad = zeros(length(z))
    losses = Float64[]
    raw_grad_norms = Float64[]
    grad_norms = Float64[]
    learning_rates = Float64[]
    clipped_flags = Bool[]

    cfg = ForwardDiff.GradientConfig(loss_fn, z)
    iter = 0
    terminated_early = false
    t0 = time()

    while true
        iter += 1
        loss = loss_fn(z)
        ForwardDiff.gradient!(grad, loss_fn, z, cfg)

        if !isfinite(loss) || !all(isfinite, grad)
            terminated_early = true
            break
        end

        raw_grad_norm = norm(grad)
        grad_norm = raw_grad_norm
        clipped = isfinite(grad_clip) && grad_norm > grad_clip

        if clipped
            grad .*= grad_clip / grad_norm
            grad_norm = grad_clip
        end

        push!(losses, loss)
        push!(raw_grad_norms, raw_grad_norm)
        push!(grad_norms, grad_norm)
        push!(clipped_flags, clipped)

        if loss < best_loss
            best_loss = loss
            best_z .= z
            best_iter = iter
        end

        lr = cosine_decay_learning_rate(iter, max(iter + 1, 360), learning_rate; decay_start=decay_start, final_lr_scale=final_lr_scale)
        push!(learning_rates, lr)

        m .= beta1 .* m .+ (1.0 - beta1) .* grad
        v .= beta2 .* v .+ (1.0 - beta2) .* (grad .^ 2)

        m_hat = m ./ (1.0 - beta1^iter)
        v_hat = v ./ (1.0 - beta2^iter)
        z .-= lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon)

        if time() - t0 >= time_limit_seconds
            break
        end
    end

    final_loss = loss_fn(z)
    final_grad = ForwardDiff.gradient(loss_fn, z)
    final_raw_grad_norm = all(isfinite, final_grad) && isfinite(final_loss) ? norm(final_grad) : Inf
    final_postclip_grad_norm = isfinite(grad_clip) ? min(final_raw_grad_norm, grad_clip) : final_raw_grad_norm

    if final_loss < best_loss
        best_loss = final_loss
        best_z .= z
        best_iter = iter
    end

    return (
        z=copy(z),
        z_best=copy(best_z),
        P_est=turning_matrices(best_z),
        y_est=simulator_dataset(best_z, dataset),
        best_loss=best_loss,
        best_iter=best_iter,
        final_loss=final_loss,
        final_raw_grad_norm=final_raw_grad_norm,
        final_postclip_grad_norm=final_postclip_grad_norm,
        losses=losses,
        raw_grad_norms=raw_grad_norms,
        grad_norms=grad_norms,
        learning_rates=learning_rates,
        clipped_flags=clipped_flags,
        clip_count=count(identity, clipped_flags),
        clip_fraction=isempty(clipped_flags) ? 0.0 : mean(clipped_flags),
        loss_tail_relspan=tail_relative_span(losses),
        grad_tail_relspan=tail_relative_span(raw_grad_norms),
        terminated_early=terminated_early,
        solve_seconds=time() - t0,
        iterations=length(losses),
    )
end

parse_table_string(value) = strip(string(value))

function parse_table_float_or_nan(value)
    text = strip(string(value))
    isempty(text) && return NaN
    text == "NaN" && return NaN
    return parse(Float64, text)
end

function existing_rows(path)
    return isfile(path) ? read_namedtuple_table(path) : NamedTuple[]
end

function fit_completed(rows, density::String, noise_sigma::Real, seed::Int, method::String)
    return any(rows) do row
        parse_table_string(row.density) == density &&
            isapprox(parse_table_float_or_nan(row.noise_sigma), Float64(noise_sigma); atol=1e-12, rtol=0.0) &&
            parse_table_int(row.noise_seed) == seed &&
            parse_table_string(row.method) == method
    end
end

function density_sensor_layout_rows(; output_dir=DENSITY_NOISE_OUTPUT_DIR, filename="pilot_sensor_layouts.tsv")
    rows = NamedTuple[]

    for spec in density_specs()
        dataset = dataset_for_density(spec, 0.08)
        setup = first(dataset.setups)
        for road_id in setup.observed_road_ids
            n_cells = road_cell_count(setup, road_id)
            for sensor_cell in road_sensor_cell_ids(setup, road_id)
                push!(
                    rows,
                    (
                        density=spec.density,
                        description=spec.description,
                        road_id=road_id,
                        road_label=road_label(road_id),
                        road_role=String(road_role_symbol(road_id)),
                        road_cells=n_cells,
                        sensor_cell=sensor_cell,
                        sensor_center_fraction=(sensor_cell - 0.5) / n_cells,
                    ),
                )
            end
        end
    end

    write_namedtuple_table(rows, joinpath(output_dir, filename))
    return rows
end

function turning_metric_values(P_est, P_true)
    errors = turning_entries(P_est) .- turning_entries(P_true)
    return (
        turning_rmse=sqrt(mean(errors .^ 2)),
        turning_mae=mean(abs.(errors)),
        turning_max_abs_error=maximum(abs.(errors)),
    )
end

function base_metric_row(density, noise_sigma, seed, method, dataset, observations, P_est, y_est, P_true; state_metrics=nothing)
    turning_metrics = turning_metric_values(P_est, P_true)
    junction_rmses = junction_turning_rmses(P_est, P_true)
    resolved_state_metrics = state_metrics === nothing ? average_state_metrics(P_est, dataset.setups, P_true) : state_metrics
    normalized_residual = (y_est .- observations.y_obs) ./ observations.sigma_model

    return (
        density=density,
        noise_sigma=Float64(noise_sigma),
        noise_seed=seed,
        method=method,
        scenario_count=dataset.regime.scenario_count,
        observation_count=dataset_observation_length(dataset),
        observation_multiplier=observation_multiplier(dataset),
        floor_fraction=DENSITY_NOISE_FLOOR_FRACTION,
        mean_clip_fraction=observations.mean_clip_fraction,
        max_clip_fraction=observations.max_clip_fraction,
        turning_rmse=turning_metrics.turning_rmse,
        turning_mae=turning_metrics.turning_mae,
        turning_max_abs_error=turning_metrics.turning_max_abs_error,
        turning_rmse_j1=junction_rmses[1],
        turning_rmse_j2=junction_rmses[2],
        turning_rmse_j3=junction_rmses[3],
        turning_rmse_j4=junction_rmses[4],
        predictive_rmse=predictive_rmse(y_est, observations.y_true),
        fit_rmse=predictive_rmse(y_est, observations.y_obs),
        normalized_fit_rmse=sqrt(mean(normalized_residual .^ 2)),
        weighted_loss=0.5 * sum(normalized_residual .^ 2),
        final_state_rmse_all=resolved_state_metrics.final_state_rmse_all,
        final_state_rmse_observed=resolved_state_metrics.final_state_rmse_observed,
        final_state_rmse_unobserved=resolved_state_metrics.final_state_rmse_unobserved,
    )
end

function esmda_metric_row(density, noise_sigma, seed, dataset, observations, esmda, P_true)
    state_metrics = ensemble_mean_state_metrics(esmda, dataset, P_true)
    base = base_metric_row(
        density,
        noise_sigma,
        seed,
        "esmda",
        dataset,
        observations,
        esmda.P_post_mean,
        esmda.y_post_mean,
        P_true;
        state_metrics=state_metrics,
    )
    interval_widths = esmda.entry_ci_95 .- esmda.entry_ci_05
    coverage = (esmda.entry_true .>= esmda.entry_ci_05) .& (esmda.entry_true .<= esmda.entry_ci_95)

    return (
        base...,
        requested_budget=DENSITY_NOISE_ENSEMBLE_SIZE * DENSITY_NOISE_ESMDA_MAXITERS,
        solve_seconds=esmda.solve_seconds,
        iterations=esmda.esmda_maxiters,
        ensemble_size=esmda.ensemble_size,
        esmda_maxiters=esmda.esmda_maxiters,
        adam_time_limit_seconds=NaN,
        adam_best_loss=NaN,
        adam_final_loss=NaN,
        adam_best_iter=0,
        adam_final_raw_grad_norm=NaN,
        adam_loss_tail_relspan=NaN,
        adam_grad_tail_relspan=NaN,
        turning_interval_width_mean=mean(interval_widths),
        turning_interval_width_max=maximum(interval_widths),
        turning_interval_coverage=mean(coverage),
    )
end

function adam_metric_row(density, noise_sigma, seed, dataset, observations, adam, P_true)
    base = base_metric_row(density, noise_sigma, seed, "map_adam", dataset, observations, adam.P_est, adam.y_est, P_true)

    return (
        base...,
        requested_budget=DENSITY_NOISE_ADAM_SECONDS,
        solve_seconds=adam.solve_seconds,
        iterations=adam.iterations,
        ensemble_size=0,
        esmda_maxiters=0,
        adam_time_limit_seconds=DENSITY_NOISE_ADAM_SECONDS,
        adam_best_loss=adam.best_loss,
        adam_final_loss=adam.final_loss,
        adam_best_iter=adam.best_iter,
        adam_final_raw_grad_norm=adam.final_raw_grad_norm,
        adam_loss_tail_relspan=adam.loss_tail_relspan,
        adam_grad_tail_relspan=adam.grad_tail_relspan,
        turning_interval_width_mean=NaN,
        turning_interval_width_max=NaN,
        turning_interval_coverage=NaN,
    )
end

function turning_entry_rows_for_fit(density, noise_sigma, seed, method, P_est, P_true)
    rows = NamedTuple[]
    P_est_mats = turning_matrices(P_est)
    P_true_mats = turning_matrices(P_true)

    for junction in 1:N_JUNCTIONS
        for incoming_row in 1:4
            for outgoing_col in 1:4
                truth = P_true_mats[junction][incoming_row, outgoing_col]
                estimate = P_est_mats[junction][incoming_row, outgoing_col]
                push!(
                    rows,
                    (
                        density=density,
                        noise_sigma=Float64(noise_sigma),
                        noise_seed=seed,
                        method=method,
                        junction=junction,
                        incoming_row=incoming_row,
                        outgoing_col=outgoing_col,
                        global_entry=turning_entry_global_index(junction, incoming_row, outgoing_col),
                        truth=truth,
                        estimate=estimate,
                        abs_error=abs(estimate - truth),
                    ),
                )
            end
        end
    end

    return rows
end

function write_pilot_summary(rows; output_dir=DENSITY_NOISE_OUTPUT_DIR, noise_levels=DENSITY_NOISE_LEVELS, filename="pilot_summary.tsv")
    isempty(rows) && return NamedTuple[]
    summary_rows = NamedTuple[]

    for density in ["dense", "sparse", "minimal"]
        for noise_sigma in noise_levels
            for method in ["esmda", "map_adam"]
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

function write_metric_plot(rows, metric::Symbol, ylabel::String, output_path::String)
    isempty(rows) && return nothing
    plt = plot(
        xlabel="Peak noise scale",
        ylabel=ylabel,
        xscale=:log10,
        legend=:outertopright,
        size=(900, 520),
        title="Density-noise pilot",
    )
    colors = Dict("dense" => :steelblue, "sparse" => :darkorange, "minimal" => :seagreen)
    styles = Dict("esmda" => :solid, "map_adam" => :dash)

    for density in ["dense", "sparse", "minimal"]
        for method in ["esmda", "map_adam"]
            group = [
                row for row in rows
                if parse_table_string(row.density) == density &&
                   parse_table_string(row.method) == method
            ]
            isempty(group) && continue
            ordered = sort(group; by=row -> parse_table_float_or_nan(row.noise_sigma))
            x = parse_table_float_or_nan.(getproperty.(ordered, :noise_sigma))
            y = parse_table_float_or_nan.(getproperty.(ordered, metric))
            plot!(plt, x, y; marker=:circle, linewidth=2, color=colors[density], linestyle=styles[method], label="$(density) $(method)")
        end
    end

    savefig(plt, output_path)
    return output_path
end

function write_pilot_plots(rows; output_dir=DENSITY_NOISE_OUTPUT_DIR)
    paths = String[]
    for (metric, ylabel, filename) in [
        (:turning_rmse, "Turning-fraction RMSE", "pilot_turning_rmse.png"),
        (:predictive_rmse, "Predictive RMSE", "pilot_predictive_rmse.png"),
        (:final_state_rmse_unobserved, "Unobserved final-state RMSE", "pilot_final_state_unobserved_rmse.png"),
    ]
        path = joinpath(output_dir, filename)
        result = write_metric_plot(rows, metric, ylabel, path)
        result === nothing || push!(paths, result)
    end
    return paths
end

function interval_metrics_from_samples(samples::AbstractMatrix, truth::AbstractVector)
    n_outputs = size(samples, 1)
    lower = Vector{Float64}(undef, n_outputs)
    upper = Vector{Float64}(undef, n_outputs)

    for idx in 1:n_outputs
        vals = collect(view(samples, idx, :))
        lower[idx] = quantile(vals, 0.05)
        upper[idx] = quantile(vals, 0.95)
    end

    point_mean = vec(mean(samples; dims=2))
    widths = upper .- lower
    covered = (truth .>= lower) .& (truth .<= upper)

    return (
        mean_rmse=sqrt(mean((point_mean .- truth) .^ 2)),
        interval_coverage=mean(covered),
        interval_width_mean=mean(widths),
        interval_width_max=maximum(widths),
    )
end

function flatten_final_state(snapshot, road_ids)
    values = Float64[]
    for road_id in road_ids
        append!(values, Float64.(snapshot[road_id]))
    end
    return values
end

function final_state_ensemble_interval_metrics(param_samples::AbstractMatrix, dataset::MultiScenarioDataset, P_true)
    keys = (:all, :observed, :unobserved)
    truth = Dict(key => Float64[] for key in keys)
    road_sets_by_setup = Vector{Dict{Symbol, Vector{Int}}}(undef, length(dataset.setups))
    block_lengths_by_setup = Vector{Dict{Symbol, Int}}(undef, length(dataset.setups))

    for (setup_idx, setup) in enumerate(dataset.setups)
        road_sets = Dict(
            :all => all_square_road_ids(),
            :observed => copy(setup.observed_road_ids),
            :unobserved => unobserved_square_road_ids(setup),
        )
        block_lengths = Dict{Symbol, Int}()
        true_snapshot = final_state_snapshot(P_true, setup)

        for key in keys
            values = flatten_final_state(true_snapshot, road_sets[key])
            append!(truth[key], values)
            block_lengths[key] = length(values)
        end

        road_sets_by_setup[setup_idx] = road_sets
        block_lengths_by_setup[setup_idx] = block_lengths
    end

    ensemble_size = size(param_samples, 2)
    samples = Dict(key => Matrix{Float64}(undef, length(truth[key]), ensemble_size) for key in keys)
    offsets = Dict(key => 0 for key in keys)

    for (setup_idx, setup) in enumerate(dataset.setups)
        road_sets = road_sets_by_setup[setup_idx]
        block_lengths = block_lengths_by_setup[setup_idx]

        for member in 1:ensemble_size
            snapshot = final_state_snapshot(view(param_samples, :, member), setup)
            for key in keys
                values = flatten_final_state(snapshot, road_sets[key])
                offset = offsets[key]
                samples[key][(offset + 1):(offset + block_lengths[key]), member] = values
            end
        end

        for key in keys
            offsets[key] += block_lengths[key]
        end
    end

    all_metrics = interval_metrics_from_samples(samples[:all], truth[:all])
    observed_metrics = interval_metrics_from_samples(samples[:observed], truth[:observed])
    unobserved_metrics = interval_metrics_from_samples(samples[:unobserved], truth[:unobserved])

    return (
        final_state_rmse_ensemble_mean_all=all_metrics.mean_rmse,
        final_state_rmse_ensemble_mean_observed=observed_metrics.mean_rmse,
        final_state_rmse_ensemble_mean_unobserved=unobserved_metrics.mean_rmse,
        final_state_interval_coverage_all=all_metrics.interval_coverage,
        final_state_interval_coverage_observed=observed_metrics.interval_coverage,
        final_state_interval_coverage_unobserved=unobserved_metrics.interval_coverage,
        final_state_interval_width_mean_all=all_metrics.interval_width_mean,
        final_state_interval_width_mean_observed=observed_metrics.interval_width_mean,
        final_state_interval_width_mean_unobserved=unobserved_metrics.interval_width_mean,
        final_state_interval_width_max_all=all_metrics.interval_width_max,
        final_state_interval_width_max_observed=observed_metrics.interval_width_max,
        final_state_interval_width_max_unobserved=unobserved_metrics.interval_width_max,
    )
end

function esmda_diagnostic_completed(rows, density::String, noise_sigma::Real, seed::Int)
    return any(rows) do row
        parse_table_string(row.density) == density &&
            isapprox(parse_table_float_or_nan(row.noise_sigma), Float64(noise_sigma); atol=1e-12, rtol=0.0) &&
            parse_table_int(row.noise_seed) == seed
    end
end

function esmda_diagnostic_row(density, noise_sigma, seed, dataset, observations, esmda, P_true; diagnostics_seconds=NaN)
    turning_widths = esmda.entry_ci_95 .- esmda.entry_ci_05
    turning_coverage = (esmda.entry_true .>= esmda.entry_ci_05) .& (esmda.entry_true .<= esmda.entry_ci_95)

    prediction_samples = simulate_ensemble_dataset(esmda.param_samples, dataset)
    prediction_metrics = interval_metrics_from_samples(prediction_samples, observations.y_true)
    state_metrics = final_state_ensemble_interval_metrics(esmda.param_samples, dataset, P_true)

    mean_parameter_state = average_state_metrics(esmda.P_post_mean, dataset.setups, P_true)
    mean_parameter_y = esmda.y_mean_parameter
    ensemble_mean_y = vec(mean(prediction_samples; dims=2))

    return (
        density=density,
        noise_sigma=Float64(noise_sigma),
        noise_seed=seed,
        scenario_count=dataset.regime.scenario_count,
        observation_count=dataset_observation_length(dataset),
        observation_multiplier=observation_multiplier(dataset),
        floor_fraction=DENSITY_NOISE_FLOOR_FRACTION,
        ensemble_size=esmda.ensemble_size,
        esmda_maxiters=esmda.esmda_maxiters,
        esmda_solve_seconds=esmda.solve_seconds,
        diagnostics_seconds=diagnostics_seconds,
        turning_rmse=overall_turning_rmse(esmda.P_post_mean, P_true),
        turning_interval_coverage=mean(turning_coverage),
        turning_interval_width_mean=mean(turning_widths),
        turning_interval_width_max=maximum(turning_widths),
        predictive_rmse_mean_parameter=predictive_rmse(mean_parameter_y, observations.y_true),
        predictive_rmse_ensemble_mean=predictive_rmse(ensemble_mean_y, observations.y_true),
        predictive_interval_coverage=prediction_metrics.interval_coverage,
        predictive_interval_width_mean=prediction_metrics.interval_width_mean,
        predictive_interval_width_max=prediction_metrics.interval_width_max,
        final_state_rmse_mean_parameter_all=mean_parameter_state.final_state_rmse_all,
        final_state_rmse_mean_parameter_observed=mean_parameter_state.final_state_rmse_observed,
        final_state_rmse_mean_parameter_unobserved=mean_parameter_state.final_state_rmse_unobserved,
        state_metrics...,
    )
end

function run_density_noise_esmda_diagnostics(; output_dir=DENSITY_NOISE_OUTPUT_DIR)
    mkpath(output_dir)
    diagnostics_path = joinpath(output_dir, "pilot_esmda_diagnostics.tsv")
    diagnostic_rows = existing_rows(diagnostics_path)
    P_true = true_turning_matrices()

    println("running ESMDA-only ensemble diagnostics")
    flush(stdout)

    for spec in density_specs()
        for noise_sigma in DENSITY_NOISE_LEVELS
            dataset = dataset_for_density(spec, noise_sigma)
            for noise_seed in DENSITY_NOISE_SEEDS
                if esmda_diagnostic_completed(diagnostic_rows, spec.density, noise_sigma, noise_seed)
                    @printf("skipping existing ESMDA diagnostics density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                    continue
                end

                observations = generate_physical_dataset_observations(P_true, dataset; seed=noise_seed, floor_fraction=DENSITY_NOISE_FLOOR_FRACTION)
                @printf(
                    "running ESMDA diagnostics density=%s noise=%.3f seed=%d obs=%d\n",
                    spec.density,
                    noise_sigma,
                    noise_seed,
                    dataset_observation_length(dataset),
                )
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

                @printf(
                    "  diagnostics done esmda=%.1fs diag=%.1fs turn_cov=%.3f pred_cov=%.3f state_unobs_cov=%.3f pred_rmse_ens=%.4f\n",
                    row.esmda_solve_seconds,
                    row.diagnostics_seconds,
                    row.turning_interval_coverage,
                    row.predictive_interval_coverage,
                    row.final_state_interval_coverage_unobserved,
                    row.predictive_rmse_ensemble_mean,
                )
                flush(stdout)
            end
        end
    end

    println()
    println("ESMDA diagnostics output")
    println("------------------------")
    println(diagnostics_path)
    for row in diagnostic_rows
        @printf(
            "%s noise=%.3f turn_cov=%.3f pred_cov=%.3f state_unobs_cov=%.3f pred_rmse_ens=%.4f\n",
            row.density,
            row.noise_sigma,
            row.turning_interval_coverage,
            row.predictive_interval_coverage,
            row.final_state_interval_coverage_unobserved,
            row.predictive_rmse_ensemble_mean,
        )
    end

    return diagnostic_rows
end

function run_warmup()
    P_true = true_turning_matrices()
    dataset = dataset_for_density(first(density_specs()), first(DENSITY_NOISE_LEVELS))
    observations = generate_physical_dataset_observations(P_true, dataset; seed=999, floor_fraction=DENSITY_NOISE_FLOOR_FRACTION)
    run_esmda_multi_scenario(dataset, observations.y_obs, observations.sigma_model; seed=999, ensemble_size=8, esmda_maxiters=1, P_true=P_true)
    run_adam_map_multi_scenario(observations.y_obs, dataset, observations.sigma_model; maxiters=1)
    return nothing
end

function run_density_noise_pilot(; output_dir=DENSITY_NOISE_OUTPUT_DIR)
    mkpath(output_dir)
    metrics_path = joinpath(output_dir, "pilot_fit_metrics.tsv")
    entries_path = joinpath(output_dir, "pilot_turning_entry_estimates.tsv")
    metric_rows = existing_rows(metrics_path)
    entry_rows = existing_rows(entries_path)

    density_sensor_layout_rows(; output_dir=output_dir)
    write_config_file(
        joinpath(output_dir, "pilot_config.txt"),
        [
            "experiment = joint observation density x noise pilot",
            "regime = $(DENSITY_NOISE_REGIME.label)",
            "scenario_count = $(DENSITY_NOISE_REGIME.scenario_count)",
            "noise_levels = $(join(DENSITY_NOISE_LEVELS, ","))",
            "noise_seeds = $(join(DENSITY_NOISE_SEEDS, ","))",
            "floor_fraction = $(DENSITY_NOISE_FLOOR_FRACTION)",
            "densities = dense,sparse,minimal",
            "dense = 9 roads, 4 sensors per road",
            "sparse = 9 roads, 1 midpoint sensor per road",
            "minimal = roads $(join(MINIMAL_INTERNAL_ROADS, ",")), 1 midpoint sensor per road",
            "esmda = $(DENSITY_NOISE_ENSEMBLE_SIZE)x$(DENSITY_NOISE_ESMDA_MAXITERS)",
            "map_adam = timed ADAM, $(DENSITY_NOISE_ADAM_SECONDS) seconds, prior_scale $(DENSITY_NOISE_ADAM_PRIOR_SCALE)",
        ],
    )

    P_true = true_turning_matrices()
    println("warming up ESMDA and ADAM compilation")
    flush(stdout)
    run_warmup()

    for spec in density_specs()
        for noise_sigma in DENSITY_NOISE_LEVELS
            dataset = dataset_for_density(spec, noise_sigma)
            for noise_seed in DENSITY_NOISE_SEEDS
                observations = generate_physical_dataset_observations(P_true, dataset; seed=noise_seed, floor_fraction=DENSITY_NOISE_FLOOR_FRACTION)

                if !fit_completed(metric_rows, spec.density, noise_sigma, noise_seed, "esmda")
                    @printf(
                        "running ESMDA density=%s noise=%.3f seed=%d obs=%d\n",
                        spec.density,
                        noise_sigma,
                        noise_seed,
                        dataset_observation_length(dataset),
                    )
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
                    push!(metric_rows, esmda_metric_row(spec.density, noise_sigma, noise_seed, dataset, observations, esmda, P_true))
                    append!(entry_rows, turning_entry_rows_for_fit(spec.density, noise_sigma, noise_seed, "esmda", esmda.P_post_mean, P_true))
                    write_namedtuple_table(metric_rows, metrics_path)
                    write_namedtuple_table(entry_rows, entries_path)
                    @printf("  ESMDA done %.1fs turning=%.4f predictive=%.4f final_unobs=%.4f\n", last(metric_rows).solve_seconds, last(metric_rows).turning_rmse, last(metric_rows).predictive_rmse, last(metric_rows).final_state_rmse_unobserved)
                    flush(stdout)
                else
                    @printf("skipping existing ESMDA density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                end

                if !fit_completed(metric_rows, spec.density, noise_sigma, noise_seed, "map_adam")
                    @printf(
                        "running MAP/ADAM density=%s noise=%.3f seed=%d obs=%d\n",
                        spec.density,
                        noise_sigma,
                        noise_seed,
                        dataset_observation_length(dataset),
                    )
                    flush(stdout)
                    adam = run_adam_map_multi_scenario_timed(
                        observations.y_obs,
                        dataset,
                        observations.sigma_model;
                        time_limit_seconds=DENSITY_NOISE_ADAM_SECONDS,
                    )
                    push!(metric_rows, adam_metric_row(spec.density, noise_sigma, noise_seed, dataset, observations, adam, P_true))
                    append!(entry_rows, turning_entry_rows_for_fit(spec.density, noise_sigma, noise_seed, "map_adam", adam.P_est, P_true))
                    write_namedtuple_table(metric_rows, metrics_path)
                    write_namedtuple_table(entry_rows, entries_path)
                    @printf("  MAP/ADAM done %.1fs iter=%d turning=%.4f predictive=%.4f final_unobs=%.4f\n", last(metric_rows).solve_seconds, last(metric_rows).iterations, last(metric_rows).turning_rmse, last(metric_rows).predictive_rmse, last(metric_rows).final_state_rmse_unobserved)
                    flush(stdout)
                else
                    @printf("skipping existing MAP/ADAM density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                end
            end
        end
    end

    summary_rows = write_pilot_summary(metric_rows; output_dir=output_dir)
    plot_paths = write_pilot_plots(metric_rows; output_dir=output_dir)

    println()
    println("Density-noise pilot outputs")
    println("---------------------------")
    println(metrics_path)
    println(entries_path)
    println(joinpath(output_dir, "pilot_summary.tsv"))
    println(joinpath(output_dir, "pilot_sensor_layouts.tsv"))
    for path in plot_paths
        println(path)
    end
    println()

    for row in summary_rows
        @printf(
            "%s noise=%.3f %-8s turning=%.4f predictive=%.4f final_unobs=%.4f solve=%.1fs\n",
            row.density,
            row.noise_sigma,
            row.method,
            row.turning_rmse_mean,
            row.predictive_rmse_mean,
            row.final_state_rmse_unobserved_mean,
            row.solve_seconds_mean,
        )
    end

    return summary_rows
end

function run_density_noise_full(; output_dir=DENSITY_NOISE_FULL_OUTPUT_DIR)
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
            "experiment = joint observation density x noise full grid",
            "regime = $(DENSITY_NOISE_REGIME.label)",
            "scenario_count = $(DENSITY_NOISE_REGIME.scenario_count)",
            "noise_levels = $(join(DENSITY_NOISE_FULL_LEVELS, ","))",
            "noise_seeds = $(join(DENSITY_NOISE_FULL_SEEDS, ","))",
            "floor_fraction = $(DENSITY_NOISE_FLOOR_FRACTION)",
            "densities = dense,sparse,minimal",
            "dense = 9 roads, 4 sensors per road",
            "sparse = 9 roads, 1 midpoint sensor per road",
            "minimal = roads $(join(MINIMAL_INTERNAL_ROADS, ",")), 1 midpoint sensor per road",
            "esmda = $(DENSITY_NOISE_ENSEMBLE_SIZE)x$(DENSITY_NOISE_ESMDA_MAXITERS)",
            "esmda_predictive_metric = mean simulated ensemble, not simulation of mean parameter",
            "map_adam = timed ADAM, $(DENSITY_NOISE_ADAM_SECONDS) seconds, prior_scale $(DENSITY_NOISE_ADAM_PRIOR_SCALE)",
            "persistence = scalar metrics, per-entry turning estimates, sensor layout, ESMDA coverage summaries; no full ensemble artifacts",
        ],
    )

    P_true = true_turning_matrices()
    println("warming up ESMDA and ADAM compilation")
    flush(stdout)
    run_warmup()

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

                    metric = last(metric_rows)
                    diag = last(diagnostic_rows)
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

                if !fit_completed(metric_rows, spec.density, noise_sigma, noise_seed, "map_adam")
                    @printf("running MAP/ADAM density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                    adam = run_adam_map_multi_scenario_timed(
                        observations.y_obs,
                        dataset,
                        observations.sigma_model;
                        time_limit_seconds=DENSITY_NOISE_ADAM_SECONDS,
                    )
                    push!(metric_rows, adam_metric_row(spec.density, noise_sigma, noise_seed, dataset, observations, adam, P_true))
                    append!(entry_rows, turning_entry_rows_for_fit(spec.density, noise_sigma, noise_seed, "map_adam", adam.P_est, P_true))
                    write_namedtuple_table(metric_rows, metrics_path)
                    write_namedtuple_table(entry_rows, entries_path)
                    @printf(
                        "  MAP/ADAM done %.1fs iter=%d turning=%.4f pred=%.4f final_unobs=%.4f\n",
                        adam.solve_seconds,
                        adam.iterations,
                        last(metric_rows).turning_rmse,
                        last(metric_rows).predictive_rmse,
                        last(metric_rows).final_state_rmse_unobserved,
                    )
                    flush(stdout)
                else
                    @printf("skipping existing MAP/ADAM density=%s noise=%.3f seed=%d\n", spec.density, noise_sigma, noise_seed)
                    flush(stdout)
                end
            end
        end
    end

    summary_rows = write_pilot_summary(
        metric_rows;
        output_dir=output_dir,
        noise_levels=DENSITY_NOISE_FULL_LEVELS,
        filename="full_summary.tsv",
    )

    println()
    println("Density-noise full outputs")
    println("--------------------------")
    println(metrics_path)
    println(entries_path)
    println(diagnostics_path)
    println(joinpath(output_dir, "full_summary.tsv"))
    println(joinpath(output_dir, "full_sensor_layouts.tsv"))
    println()

    for row in summary_rows
        @printf(
            "%s noise=%.3f %-8s n=%d turning=%.4f [%.4f, %.4f] pred=%.4f final_unobs=%.4f\n",
            row.density,
            row.noise_sigma,
            row.method,
            row.seed_count,
            row.turning_rmse_mean,
            row.turning_rmse_min,
            row.turning_rmse_max,
            row.predictive_rmse_mean,
            row.final_state_rmse_unobserved_mean,
        )
    end

    return summary_rows
end

function full_ordered_summary_rows(output_dir=DENSITY_NOISE_FULL_OUTPUT_DIR)
    path = joinpath(output_dir, "full_summary.tsv")
    isfile(path) || error("Missing full summary file: $(path)")
    rows = read_namedtuple_table(path)
    density_rank = Dict("dense" => 1, "sparse" => 2, "minimal" => 3)
    method_rank = Dict("esmda" => 1, "map_adam" => 2)
    return sort(
        rows;
        by=row -> (
            get(density_rank, parse_table_string(row.density), 99),
            parse_table_float_or_nan(row.noise_sigma),
            get(method_rank, parse_table_string(row.method), 99),
        ),
    )
end

function full_diagnostic_rows(output_dir=DENSITY_NOISE_FULL_OUTPUT_DIR)
    path = joinpath(output_dir, "full_esmda_diagnostics.tsv")
    isfile(path) || error("Missing ESMDA diagnostics file: $(path)")
    return read_namedtuple_table(path)
end

function add_mean_range_series!(
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
    plot!(
        plt,
        x_positions,
        means;
        subplot=subplot_id,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=2.6,
        marker=:circle,
        markersize=6,
        markerstrokewidth=1.0,
    )

    cap = 0.055
    for (x, y_min, y_max) in zip(x_positions, mins, maxs)
        plot!(plt, [x, x], [y_min, y_max]; subplot=subplot_id, color=color, alpha=0.78, linewidth=2.0, label="")
        plot!(plt, [x - cap, x + cap], [y_min, y_min]; subplot=subplot_id, color=color, alpha=0.78, linewidth=2.0, label="")
        plot!(plt, [x - cap, x + cap], [y_max, y_max]; subplot=subplot_id, color=color, alpha=0.78, linewidth=2.0, label="")
    end

    return plt
end

function write_full_metric_plot(
    summary_rows,
    metric_prefix::String,
    ylabel::String,
    output_path::String;
    plot_title::String,
)
    densities = ["dense", "sparse", "minimal"]
    methods = [
        (method="esmda", label="ESMDA", color=:steelblue, offset=-0.06, linestyle=:solid),
        (method="map_adam", label="MAP", color=:darkorange, offset=0.06, linestyle=:dash),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]

    plt = plot(
        layout=(1, 3),
        size=(1120, 360),
        legend=:top,
        bottom_margin=7Plots.mm,
        left_margin=6Plots.mm,
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
            add_mean_range_series!(
                plt,
                subplot_id,
                x_positions,
                means,
                mins,
                maxs;
                label=method_spec.label,
                color=method_spec.color,
                linestyle=method_spec.linestyle,
            )
        end
    end

    savefig(plt, output_path)
    return output_path
end

function coverage_summary_rows(diagnostic_rows; output_dir=DENSITY_NOISE_FULL_OUTPUT_DIR)
    rows = NamedTuple[]

    for density in ["dense", "sparse", "minimal"]
        for noise_sigma in DENSITY_NOISE_FULL_LEVELS
            group = [
                row for row in diagnostic_rows
                if parse_table_string(row.density) == density &&
                   isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
            ]
            isempty(group) && continue

            turning = parse_table_float_or_nan.(getproperty.(group, :turning_interval_coverage))
            predictive = parse_table_float_or_nan.(getproperty.(group, :predictive_interval_coverage))
            state_unobserved = parse_table_float_or_nan.(getproperty.(group, :final_state_interval_coverage_unobserved))

            push!(
                rows,
                (
                    density=density,
                    noise_sigma=noise_sigma,
                    seed_count=length(group),
                    turning_coverage_mean=mean(turning),
                    turning_coverage_min=minimum(turning),
                    turning_coverage_max=maximum(turning),
                    predictive_coverage_mean=mean(predictive),
                    predictive_coverage_min=minimum(predictive),
                    predictive_coverage_max=maximum(predictive),
                    final_state_unobserved_coverage_mean=mean(state_unobserved),
                    final_state_unobserved_coverage_min=minimum(state_unobserved),
                    final_state_unobserved_coverage_max=maximum(state_unobserved),
                ),
            )
        end
    end

    write_namedtuple_table(rows, joinpath(output_dir, "full_esmda_coverage_summary.tsv"))
    return rows
end

function write_full_esmda_coverage_plot(coverage_rows, output_path::String)
    densities = ["dense", "sparse", "minimal"]
    coverage_specs = [
        (prefix="turning_coverage", label="Turning fractions", color=:steelblue, offset=-0.08, linestyle=:solid),
        (prefix="predictive_coverage", label="Observed trajectories", color=:darkorange, offset=0.0, linestyle=:dash),
        (prefix="final_state_unobserved_coverage", label="Unobserved final state", color=:seagreen, offset=0.08, linestyle=:dot),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]

    plt = plot(
        layout=(1, 3),
        size=(1200, 420),
        legend=:bottomleft,
        bottom_margin=9Plots.mm,
        left_margin=8Plots.mm,
    )

    for (subplot_id, density) in enumerate(densities)
        plot!(
            plt;
            subplot=subplot_id,
            title=uppercasefirst(density),
            xlabel="Peak noise scale",
            ylabel=subplot_id == 1 ? "Coverage" : "",
            xticks=(x_base, x_labels),
            xlims=(0.6, length(noise_levels) + 0.4),
            ylims=(0.0, 1.05),
            grid=true,
            framestyle=:box,
        )
        hline!(plt, [0.90]; subplot=subplot_id, color=:gray45, linestyle=:dash, linewidth=1.2, label=subplot_id == 1 ? "Nominal 0.90" : "")

        for spec in coverage_specs
            means = Float64[]
            mins = Float64[]
            maxs = Float64[]

            for noise_sigma in noise_levels
                group = [
                    row for row in coverage_rows
                    if parse_table_string(row.density) == density &&
                       isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
                ]
                isempty(group) && continue
                row = only(group)
                push!(means, parse_table_float_or_nan(getproperty(row, Symbol(spec.prefix * "_mean"))))
                push!(mins, parse_table_float_or_nan(getproperty(row, Symbol(spec.prefix * "_min"))))
                push!(maxs, parse_table_float_or_nan(getproperty(row, Symbol(spec.prefix * "_max"))))
            end

            x_positions = Float64.(x_base[1:length(means)]) .+ spec.offset
            add_mean_range_series!(
                plt,
                subplot_id,
                x_positions,
                means,
                mins,
                maxs;
                label=subplot_id == 1 ? spec.label : "",
                color=spec.color,
                linestyle=spec.linestyle,
            )
        end
    end

    savefig(plt, output_path)
    return output_path
end

function write_full_turning_headline_plot(summary_rows, output_path::String)
    density_specs_plot = [
        (density="dense", label="dense", color=:steelblue),
        (density="sparse", label="sparse", color=:darkorange),
        (density="minimal", label="minimal", color=:seagreen),
    ]
    method_specs = [
        (method="esmda", label="ESMDA", linestyle=:solid, offset=-0.035),
        (method="map_adam", label="MAP", linestyle=:dash, offset=0.035),
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
            add_mean_range_series!(
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

function write_full_prediction_rmse_panels(summary_rows, output_path::String)
    densities = ["dense", "sparse", "minimal"]
    metric_specs = [
        (prefix="predictive_rmse", ylabel="Predictive RMSE"),
        (prefix="final_state_rmse_unobserved", ylabel="Unobserved final-state RMSE"),
    ]
    method_specs = [
        (method="esmda", label="ESMDA", color=:steelblue, offset=-0.06, linestyle=:solid),
        (method="map_adam", label="MAP", color=:darkorange, offset=0.06, linestyle=:dash),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]

    plt = plot(
        layout=(2, 3),
        size=(1120, 650),
        legend=:top,
        bottom_margin=7Plots.mm,
        left_margin=7Plots.mm,
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
                add_mean_range_series!(
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

function write_full_coverage_by_quantity_plot(coverage_rows, output_path::String)
    quantity_specs = [
        (prefix="turning_coverage", title="Turning fractions"),
        (prefix="predictive_coverage", title="Observed trajectories"),
        (prefix="final_state_unobserved_coverage", title="Unobserved final state"),
    ]
    density_specs_plot = [
        (density="dense", label="dense", color=:steelblue, offset=-0.06),
        (density="sparse", label="sparse", color=:darkorange, offset=0.0),
        (density="minimal", label="minimal", color=:seagreen, offset=0.06),
    ]
    noise_levels = DENSITY_NOISE_FULL_LEVELS
    x_base = collect(1:length(noise_levels))
    x_labels = [@sprintf("%.2g", sigma) for sigma in noise_levels]

    plt = plot(
        layout=(1, 3),
        size=(1200, 420),
        legend=:bottomleft,
        bottom_margin=9Plots.mm,
        left_margin=8Plots.mm,
    )

    for (subplot_id, quantity_spec) in enumerate(quantity_specs)
        plot!(
            plt;
            subplot=subplot_id,
            title=quantity_spec.title,
            xlabel="Peak noise scale",
            ylabel=subplot_id == 1 ? "Coverage" : "",
            xticks=(x_base, x_labels),
            xlims=(0.6, length(noise_levels) + 0.4),
            ylims=(0.0, 1.05),
            grid=true,
            framestyle=:box,
        )
        hline!(plt, [0.90]; subplot=subplot_id, color=:gray45, linestyle=:dash, linewidth=1.2, label=subplot_id == 1 ? "Nominal 0.90" : "")

        for density_spec in density_specs_plot
            means = Float64[]
            mins = Float64[]
            maxs = Float64[]

            for noise_sigma in noise_levels
                group = [
                    row for row in coverage_rows
                    if parse_table_string(row.density) == density_spec.density &&
                       isapprox(parse_table_float_or_nan(row.noise_sigma), noise_sigma; atol=1e-12, rtol=0.0)
                ]
                isempty(group) && continue
                row = only(group)
                push!(means, parse_table_float_or_nan(getproperty(row, Symbol(quantity_spec.prefix * "_mean"))))
                push!(mins, parse_table_float_or_nan(getproperty(row, Symbol(quantity_spec.prefix * "_min"))))
                push!(maxs, parse_table_float_or_nan(getproperty(row, Symbol(quantity_spec.prefix * "_max"))))
            end

            x_positions = Float64.(x_base[1:length(means)]) .+ density_spec.offset
            add_mean_range_series!(
                plt,
                subplot_id,
                x_positions,
                means,
                mins,
                maxs;
                label=subplot_id == 1 ? density_spec.label : "",
                color=density_spec.color,
                linestyle=:solid,
            )
        end
    end

    savefig(plt, output_path)
    return output_path
end

function run_density_noise_full_plots(; output_dir=DENSITY_NOISE_FULL_OUTPUT_DIR)
    summary_rows = full_ordered_summary_rows(output_dir)
    diagnostic_rows = full_diagnostic_rows(output_dir)
    coverage_rows = coverage_summary_rows(diagnostic_rows; output_dir=output_dir)

    paths = String[]
    push!(
        paths,
        write_full_metric_plot(
            summary_rows,
            "turning_rmse",
            "Turning-fraction RMSE",
            joinpath(output_dir, "full_turning_rmse_by_density.png");
            plot_title="Turning-fraction RMSE by observation density",
        ),
    )
    push!(
        paths,
        write_full_metric_plot(
            summary_rows,
            "predictive_rmse",
            "Predictive RMSE",
            joinpath(output_dir, "full_predictive_rmse_by_density.png");
            plot_title="Predictive RMSE by observation density",
        ),
    )
    push!(
        paths,
        write_full_metric_plot(
            summary_rows,
            "final_state_rmse_unobserved",
            "Unobserved final-state RMSE",
            joinpath(output_dir, "full_final_state_unobserved_rmse_by_density.png");
            plot_title="Unobserved final-state RMSE by observation density",
        ),
    )
    push!(
        paths,
        write_full_esmda_coverage_plot(
            coverage_rows,
            joinpath(output_dir, "full_esmda_coverage_by_density.png"),
        ),
    )
    push!(
        paths,
        write_full_turning_headline_plot(
            summary_rows,
            joinpath(output_dir, "full_turning_rmse_headline.png"),
        ),
    )
    push!(
        paths,
        write_full_coverage_by_quantity_plot(
            coverage_rows,
            joinpath(output_dir, "full_esmda_coverage_by_quantity.png"),
        ),
    )
    push!(
        paths,
        write_full_prediction_rmse_panels(
            summary_rows,
            joinpath(output_dir, "full_prediction_rmse_panels.png"),
        ),
    )

    println("Full-result visualizations")
    println("--------------------------")
    println(joinpath(output_dir, "full_esmda_coverage_summary.tsv"))
    for path in paths
        println(path)
    end

    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "pilot"
    if mode == "pilot"
        run_density_noise_pilot()
    elseif mode == "esmda_diagnostics"
        run_density_noise_esmda_diagnostics()
    elseif mode == "full"
        run_density_noise_full()
    elseif mode == "plot_full"
        run_density_noise_full_plots()
    else
        error("Unknown mode $(mode). Use pilot, esmda_diagnostics, full, or plot_full.")
    end
end
