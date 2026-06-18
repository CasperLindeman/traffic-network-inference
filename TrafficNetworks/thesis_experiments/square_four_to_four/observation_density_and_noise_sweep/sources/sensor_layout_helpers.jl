ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_STRUCTURE"] = "1"

include(joinpath(@__DIR__, "turning_recovery_helpers.jl"))

using Printf
using Statistics
using Plots

const SPARSE_SENSOR_OUTPUT_DIR = joinpath(
    @__DIR__,
    "..",
    "outputs",
    "sparse_sensor_experiments",
)

const SPARSE_CLASS_ORDER = ["both", "source", "target", "neither"]

middle_cell(n_cells::Int) = clamp(round(Int, (n_cells + 1) / 2), 1, n_cells)

function sparse_sensor_cell(base_setup::SquareSingleScenarioSetup, road_id::Int, position_label::String)
    n_cells = road_cell_count(base_setup, road_id)
    role = road_role_symbol(road_id)

    if role == :incoming
        if position_label == "near-junction"
            return n_cells
        elseif position_label == "near-boundary"
            return 1
        elseif position_label in ("midpoint", "middle")
            return middle_cell(n_cells)
        end
    elseif role == :outgoing
        if position_label == "near-junction"
            return 1
        elseif position_label == "near-boundary"
            return n_cells
        elseif position_label in ("midpoint", "middle")
            return middle_cell(n_cells)
        end
    else
        if position_label == "near-upstream"
            return 1
        elseif position_label == "near-downstream"
            return n_cells
        elseif position_label in ("midpoint", "middle")
            return middle_cell(n_cells)
        end
    end

    error("Unsupported sensor position $(position_label) for road $(road_id) ($(role)).")
end

function step1a_config_specs()
    return [
        (config="C1", label="C1_near_junction_middle", external_inflow="near-junction", external_outflow="near-junction", internal_connector="middle"),
        (config="C2", label="C2_near_boundary_middle", external_inflow="near-boundary", external_outflow="near-boundary", internal_connector="middle"),
        (config="C3", label="C3_near_junction_upstream", external_inflow="near-junction", external_outflow="near-junction", internal_connector="near-upstream"),
        (config="C4", label="C4_near_junction_downstream", external_inflow="near-junction", external_outflow="near-junction", internal_connector="near-downstream"),
        (config="C5", label="C5_midpoint", external_inflow="midpoint", external_outflow="midpoint", internal_connector="middle"),
        (config="C6", label="C6_inflow_boundary_outflow_junction", external_inflow="near-boundary", external_outflow="near-junction", internal_connector="middle"),
    ]
end

function sensor_cells_for_config(spec; observed_roads=copy(DEFAULT_OBSERVED_ROADS), peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA)
    base_setup = square_single_scenario_setup(peak_noise_sigma=peak_noise_sigma)
    cells = Dict{Int, Vector{Int}}()

    for road_id in observed_roads
        role = road_role_symbol(road_id)
        position = role == :incoming ? spec.external_inflow :
                   role == :outgoing ? spec.external_outflow :
                   spec.internal_connector
        cells[road_id] = [sparse_sensor_cell(base_setup, road_id, position)]
    end

    return cells
end

function square_single_scenario_setup_with_sensors(;
    observed_roads=copy(DEFAULT_OBSERVED_ROADS),
    sensor_cells_by_road,
    peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA,
)
    base = square_single_scenario_setup(peak_noise_sigma=peak_noise_sigma)
    return SquareSingleScenarioSetup(
        base.T,
        base.CFL,
        base.base_length_km,
        base.cells_per_base_length,
        copy(base.control_times),
        Int.(collect(observed_roads)),
        copy_sensor_spec(sensor_cells_by_road),
        copy(base.boundary_road_ids),
        copy(base.inflows),
        copy(base.road_profiles),
        copy(base.road_length_multipliers),
        copy(base.speed_limits),
        base.physical_noise_peak_sigma,
    )
end

function build_sparse_multi_scenario_dataset(
    regime::MultiScenarioDataRegime,
    sensor_cells_by_road;
    observed_roads=copy(DEFAULT_OBSERVED_ROADS),
    peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA,
)
    base_setup = square_single_scenario_setup_with_sensors(
        observed_roads=observed_roads,
        sensor_cells_by_road=sensor_cells_by_road,
        peak_noise_sigma=peak_noise_sigma,
    )
    specs = multi_scenario_scenario_specs(regime.scenario_count)
    labels = [spec.label for spec in specs]
    setups = [build_multi_scenario_setup(base_setup, spec, regime) for spec in specs]
    return MultiScenarioDataset(regime, labels, setups)
end

function sparse_sensor_layout_rows(config_specs; output_dir=SPARSE_SENSOR_OUTPUT_DIR)
    base_setup = square_single_scenario_setup()
    rows = NamedTuple[]

    for spec in config_specs
        cells = sensor_cells_for_config(spec)
        for road_id in DEFAULT_OBSERVED_ROADS
            n_cells = road_cell_count(base_setup, road_id)
            sensor_cell = only(cells[road_id])
            role = String(road_role_symbol(road_id))
            position = role == "incoming" ? spec.external_inflow :
                       role == "outgoing" ? spec.external_outflow :
                       spec.internal_connector
            push!(
                rows,
                (
                    config=spec.config,
                    config_label=spec.label,
                    road_id=road_id,
                    road_label=road_label(road_id),
                    road_role=role,
                    position=position,
                    road_cells=n_cells,
                    sensor_cell=sensor_cell,
                    sensor_center_fraction=(sensor_cell - 0.5) / n_cells,
                ),
            )
        end
    end

    write_namedtuple_table(rows, joinpath(output_dir, "sparse_sensor_layouts.tsv"))
    return rows
end

function parse_float_value(value)
    value isa Real && return Float64(value)
    return parse(Float64, strip(string(value)))
end

function parse_int_value(value)
    value isa Integer && return Int(value)
    return parse(Int, strip(string(value)))
end

function parse_string_value(value)
    return strip(string(value))
end

function sparse_fit_completed(rows, config_label::String, noise_seed::Int)
    return any(row -> parse_string_value(row.config_label) == config_label && parse_int_value(row.noise_seed) == noise_seed, rows)
end

function sparse_metric_rows(output_dir)
    path = joinpath(output_dir, "fit_metrics.tsv")
    return isfile(path) ? read_namedtuple_table(path) : NamedTuple[]
end

function entry_rows_for_fit(config_label, noise_seed, esmda, P_true)
    rows = NamedTuple[]
    P_est_mats = turning_matrices(esmda.P_post_mean)
    P_true_mats = turning_matrices(P_true)

    for junction in 1:N_JUNCTIONS
        for incoming_row in 1:4
            for outgoing_col in 1:4
                truth = P_true_mats[junction][incoming_row, outgoing_col]
                estimate = P_est_mats[junction][incoming_row, outgoing_col]
                push!(
                    rows,
                    (
                        config_label=config_label,
                        noise_seed=noise_seed,
                        junction=junction,
                        incoming_row=incoming_row,
                        outgoing_col=outgoing_col,
                        global_entry=turning_entry_global_index(junction, incoming_row, outgoing_col),
                        truth=truth,
                        posterior_mean=estimate,
                        abs_error=abs(estimate - truth),
                    ),
                )
            end
        end
    end

    return rows
end

function sparse_metric_row(stage, spec, noise_seed, dataset, observations, esmda, P_true)
    state_metrics = average_state_metrics(esmda.P_post_mean, dataset.setups, P_true)
    junction_rmses = junction_turning_rmses(esmda.P_post_mean, P_true)
    return (
        stage=stage,
        config=spec.config,
        config_label=spec.label,
        noise_seed=noise_seed,
        esmda_seed=1,
        scenario_count=dataset.regime.scenario_count,
        observation_count=dataset_observation_length(dataset),
        observation_multiplier=observation_multiplier(dataset),
        ensemble_size=esmda.ensemble_size,
        esmda_maxiters=esmda.esmda_maxiters,
        solve_seconds=esmda.solve_seconds,
        mean_clip_fraction=observations.mean_clip_fraction,
        max_clip_fraction=observations.max_clip_fraction,
        turning_rmse=overall_turning_rmse(esmda.P_post_mean, P_true),
        turning_rmse_j1=junction_rmses[1],
        turning_rmse_j2=junction_rmses[2],
        turning_rmse_j3=junction_rmses[3],
        turning_rmse_j4=junction_rmses[4],
        predictive_rmse=predictive_rmse(esmda.y_mean_parameter, observations.y_true),
        predictive_rmse_ensemble_mean=predictive_rmse(esmda.y_post_mean, observations.y_true),
        fit_rmse=predictive_rmse(esmda.y_mean_parameter, observations.y_obs),
        final_state_rmse_all=state_metrics.final_state_rmse_all,
        final_state_rmse_observed=state_metrics.final_state_rmse_observed,
        final_state_rmse_unobserved=state_metrics.final_state_rmse_unobserved,
    )
end

function run_sparse_fit!(
    metric_rows::Vector,
    entry_rows::Vector,
    stage::String,
    spec;
    noise_seed::Int,
    output_dir=SPARSE_SENSOR_OUTPUT_DIR,
    regime=MultiScenarioDataRegime("multi_scenario_12x", 1, 12),
    ensemble_size=192,
    esmda_maxiters=6,
    floor_fraction=0.15,
    peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA,
)
    if sparse_fit_completed(metric_rows, spec.label, noise_seed)
        println("$(spec.label) seed=$(noise_seed) already complete; skipping")
        return metric_rows, entry_rows
    end

    P_true = true_turning_matrices()
    sensor_cells = sensor_cells_for_config(spec; peak_noise_sigma=peak_noise_sigma)
    dataset = build_sparse_multi_scenario_dataset(regime, sensor_cells; peak_noise_sigma=peak_noise_sigma)
    observations = generate_physical_dataset_observations(P_true, dataset; seed=noise_seed, floor_fraction=floor_fraction)

    @printf("running %s seed=%d obs=%d (%.2fx)\n", spec.label, noise_seed, dataset_observation_length(dataset), observation_multiplier(dataset))
    flush(stdout)
    esmda = run_esmda_multi_scenario(
        dataset,
        observations.y_obs,
        observations.sigma_model;
        seed=1,
        ensemble_size=ensemble_size,
        esmda_maxiters=esmda_maxiters,
        P_true=P_true,
    )

    push!(metric_rows, sparse_metric_row(stage, spec, noise_seed, dataset, observations, esmda, P_true))
    append!(entry_rows, entry_rows_for_fit(spec.label, noise_seed, esmda, P_true))
    write_namedtuple_table(metric_rows, joinpath(output_dir, "fit_metrics.tsv"))
    write_namedtuple_table(entry_rows, joinpath(output_dir, "turning_entry_estimates.tsv"))

    @printf(
        "%s seed=%d done %.1fs turning=%.4f predictive=%.4f final_unobs=%.4f\n",
        spec.label,
        noise_seed,
        esmda.solve_seconds,
        last(metric_rows).turning_rmse,
        last(metric_rows).predictive_rmse,
        last(metric_rows).final_state_rmse_unobserved,
    )
    flush(stdout)
    return metric_rows, entry_rows
end

function summarize_sparse_configs(output_dir=SPARSE_SENSOR_OUTPUT_DIR)
    metrics_path = joinpath(output_dir, "fit_metrics.tsv")
    isfile(metrics_path) || return NamedTuple[]
    rows = read_namedtuple_table(metrics_path)
    configs = sort(unique(parse_string_value.(getproperty.(rows, :config_label))))
    summary_rows = NamedTuple[]

    for config_label in configs
        group_rows = [row for row in rows if parse_string_value(row.config_label) == config_label]
        turning = parse_float_value.(getproperty.(group_rows, :turning_rmse))
        predictive = parse_float_value.(getproperty.(group_rows, :predictive_rmse))
        final_unobs = parse_float_value.(getproperty.(group_rows, :final_state_rmse_unobserved))
        push!(
            summary_rows,
            (
                config_label=config_label,
                seed_count=length(group_rows),
                turning_rmse_mean=mean(turning),
                turning_rmse_min=minimum(turning),
                turning_rmse_max=maximum(turning),
                predictive_rmse_mean=mean(predictive),
                predictive_rmse_min=minimum(predictive),
                predictive_rmse_max=maximum(predictive),
                final_state_rmse_unobserved_mean=mean(final_unobs),
                final_state_rmse_unobserved_min=minimum(final_unobs),
                final_state_rmse_unobserved_max=maximum(final_unobs),
            ),
        )
    end

    write_namedtuple_table(summary_rows, joinpath(output_dir, "config_summary.tsv"))
    return summary_rows
end

function write_step1a_turning_plot(output_dir=SPARSE_SENSOR_OUTPUT_DIR)
    rows = read_namedtuple_table(joinpath(output_dir, "fit_metrics.tsv"))
    specs = step1a_config_specs()
    labels = [spec.label for spec in specs if any(row -> parse_string_value(row.config_label) == spec.label, rows)]
    isempty(labels) && return nothing
    means = [
        mean(parse_float_value.([row.turning_rmse for row in rows if parse_string_value(row.config_label) == label]))
        for label in labels
    ]
    short_labels = [replace(label, "_" => "\n") for label in labels]

    plt = bar(
        short_labels,
        means;
        legend=false,
        ylabel="Turning-fraction RMSE",
        title="Sparse single-sensor position sweep",
        color=:gray70,
        linecolor=:gray35,
        size=(980, 440),
        xrotation=15,
        bottom_margin=10Plots.mm,
    )

    for (idx, label) in enumerate(labels)
        vals = parse_float_value.([row.turning_rmse for row in rows if parse_string_value(row.config_label) == label])
        scatter!(plt, fill(idx, length(vals)), vals; color=:black, markersize=3, alpha=0.70)
    end

    path = joinpath(output_dir, "step1a_turning_rmse_by_config.png")
    savefig(plt, path)
    return path
end

function run_sparse_sensor_experiments(mode::String; output_dir=SPARSE_SENSOR_OUTPUT_DIR)
    mkpath(output_dir)
    specs = step1a_config_specs()
    sparse_sensor_layout_rows(specs; output_dir=output_dir)
    write_config_file(
        joinpath(output_dir, "experiment_config.txt"),
        [
            "experiment = sparse single-sensor square-network followups",
            "observed_roads = $(join(DEFAULT_OBSERVED_ROADS, ","))",
            "sensor_rule = one sensor per observed road; road-specific cell index",
            "near_junction = first or last grid cell adjacent to the junction",
            "regime = multi_scenario_12x",
            "scenario_count = 12",
            "ensemble_size = 192",
            "esmda_maxiters = 6",
            "floor_fraction = 0.15",
        ],
    )

    metric_rows = sparse_metric_rows(output_dir)
    entry_path = joinpath(output_dir, "turning_entry_estimates.tsv")
    entry_rows = isfile(entry_path) ? read_namedtuple_table(entry_path) : NamedTuple[]

    if mode == "step0"
        c5 = only([spec for spec in specs if spec.config == "C5"])
        run_sparse_fit!(metric_rows, entry_rows, "step0_rebaseline", c5; noise_seed=1, output_dir=output_dir)
    elseif mode == "step1a_pilot"
        for spec in specs
            run_sparse_fit!(metric_rows, entry_rows, "step1a_pilot", spec; noise_seed=1, output_dir=output_dir)
        end
    elseif mode == "step1a_full"
        for noise_seed in 1:3
            for spec in specs
                run_sparse_fit!(metric_rows, entry_rows, "step1a_full", spec; noise_seed=noise_seed, output_dir=output_dir)
            end
        end
    elseif mode == "pilot"
        c5 = only([spec for spec in specs if spec.config == "C5"])
        run_sparse_fit!(metric_rows, entry_rows, "step0_rebaseline", c5; noise_seed=1, output_dir=output_dir)
        for spec in specs
            run_sparse_fit!(metric_rows, entry_rows, "step1a_pilot", spec; noise_seed=1, output_dir=output_dir)
        end
    else
        error("Unknown mode $(mode). Use step0, step1a_pilot, step1a_full, or pilot.")
    end

    summary_rows = summarize_sparse_configs(output_dir)
    plot_path = isfile(joinpath(output_dir, "fit_metrics.tsv")) ? write_step1a_turning_plot(output_dir) : nothing
    println()
    println("Sparse sensor experiment outputs")
    println("--------------------------------")
    println(joinpath(output_dir, "fit_metrics.tsv"))
    println(joinpath(output_dir, "turning_entry_estimates.tsv"))
    println(joinpath(output_dir, "config_summary.tsv"))
    println(joinpath(output_dir, "sparse_sensor_layouts.tsv"))
    plot_path === nothing || println(plot_path)
    println()
    for row in summary_rows
        @printf(
            "%s n=%d turning %.4f [%.4f, %.4f] predictive %.4f final_unobs %.4f\n",
            row.config_label,
            row.seed_count,
            row.turning_rmse_mean,
            row.turning_rmse_min,
            row.turning_rmse_max,
            row.predictive_rmse_mean,
            row.final_state_rmse_unobserved_mean,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "pilot"
    run_sparse_sensor_experiments(mode)
end
