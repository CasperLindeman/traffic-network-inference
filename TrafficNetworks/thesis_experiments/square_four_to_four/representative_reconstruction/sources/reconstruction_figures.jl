import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using Printf
using Statistics
using Plots

include(joinpath(@__DIR__, "..", "..", "common", "multi_scenario", "common.jl"))

const REPRESENTATIVE_OUTPUT_DIR = joinpath(MULTI_SCENARIO_OUTPUT_DIR, "representative_reconstruction")
const REPRESENTATIVE_ADAM_ITERS = 813
const REPRESENTATIVE_HORIZON_MINUTES = 5.0

road_id_label(road_id::Int) = "r$(road_id)"

function observation_status_label(setup::SquareSingleScenarioSetup, road_id::Int)
    return road_id in setup.observed_road_ids ? "observed" : "unobserved"
end

road_status_title(setup::SquareSingleScenarioSetup, road_id::Int) = road_id_label(road_id)

function repeated_base_inflows(base_setup::SquareSingleScenarioSetup)
    return [
        let base_inflow=inflow, base_horizon=base_setup.T
            t -> base_inflow(wrap_time_to_horizon(t, base_horizon))
        end
        for inflow in base_setup.inflows
    ]
end

function build_representative_dataset(horizon_minutes::Real; peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA)
    base_setup = square_single_scenario_setup(peak_noise_sigma=peak_noise_sigma)
    horizon_hours = minutes_to_hours(horizon_minutes)
    dt = first(base_setup.control_times)
    control_times = collect(dt:dt:horizon_hours)
    label = @sprintf("%.0fmin_reconstruction", horizon_minutes)

    setup = SquareSingleScenarioSetup(
        horizon_hours,
        base_setup.CFL,
        base_setup.base_length_km,
        base_setup.cells_per_base_length,
        control_times,
        copy(base_setup.observed_road_ids),
        copy(base_setup.sensor_fractions),
        copy(base_setup.boundary_road_ids),
        repeated_base_inflows(base_setup),
        copy(base_setup.road_profiles),
        copy(base_setup.road_length_multipliers),
        copy(base_setup.speed_limits),
        base_setup.physical_noise_peak_sigma,
    )

    regime = MultiScenarioDataRegime(label, max(1, round(Int, horizon_minutes / 2.0)), 1)
    return MultiScenarioDataset(regime, [label], [setup])
end

function summarize_selected_final_states(
    param_samples::AbstractMatrix,
    weights::AbstractVector,
    setup::SquareSingleScenarioSetup,
    road_ids::AbstractVector{Int};
    level=0.90,
)
    weights_norm = normalize_weights(weights)
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q
    snapshots = [final_state_snapshot(view(param_samples, :, member), setup) for member in 1:size(param_samples, 2)]
    summaries = Dict{Int, NamedTuple}()

    for road_id in road_ids
        n_cells = road_cell_count(setup, road_id)
        mean_state = Vector{Float64}(undef, n_cells)
        lower_state = Vector{Float64}(undef, n_cells)
        upper_state = Vector{Float64}(undef, n_cells)

        for cell_id in 1:n_cells
            draws = [snapshots[member][road_id][cell_id] for member in eachindex(snapshots)]
            mean_state[cell_id] = dot(draws, weights_norm)
            lower_state[cell_id] = weighted_quantile(draws, weights_norm, lower_q)
            upper_state[cell_id] = weighted_quantile(draws, weights_norm, upper_q)
        end

        summaries[road_id] = (mean=mean_state, lower=lower_state, upper=upper_state)
    end

    return summaries
end

function road_state_rmse(estimate::AbstractVector, truth::AbstractVector, road_id::Int)
    return sqrt(mean((estimate[road_id] .- truth[road_id]) .^ 2))
end

road_profile_rmse(estimate::AbstractVector, truth::AbstractVector) = sqrt(mean((estimate .- truth) .^ 2))

function select_representative_roads(
    setup::SquareSingleScenarioSetup,
    true_state,
    adam_state,
    esmda_summary;
    n_observed=3,
    n_unobserved=3,
)
    rows = NamedTuple[]

    for road_id in all_square_road_ids()
        adam_error = road_state_rmse(adam_state, true_state, road_id)
        esmda_error = road_profile_rmse(esmda_summary[road_id].mean, true_state[road_id])
        push!(
            rows,
            (
                road_id=road_id,
                observed=road_id in setup.observed_road_ids,
                adam_rmse=adam_error,
                esmda_rmse=esmda_error,
                display_score=max(adam_error, esmda_error),
            ),
        )
    end

    observed_rows = filter(row -> row.observed, rows)
    unobserved_rows = filter(row -> !row.observed, rows)
    sort!(observed_rows; by=row -> row.display_score, rev=true)
    sort!(unobserved_rows; by=row -> row.display_score, rev=true)

    selected = vcat(first(observed_rows, n_observed), first(unobserved_rows, n_unobserved))
    return getproperty.(selected, :road_id), rows, selected
end

function plot_representative_final_state(
    setup::SquareSingleScenarioSetup,
    true_state,
    adam_state,
    esmda_summary;
    road_ids,
    output_path=nothing,
)
    plt = plot(
        layout=(2, 3),
        size=(1500, 850),
        dpi=220,
        legend=:top,
        left_margin=8Plots.mm,
        right_margin=4Plots.mm,
        top_margin=5Plots.mm,
        bottom_margin=7Plots.mm,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
    )
    first_observed_subplot = findfirst(road_id -> road_id in setup.observed_road_ids, road_ids)

    for (subplot_id, road_id) in enumerate(road_ids)
        x = road_cell_centers_meters(setup, road_id)
        summary = esmda_summary[road_id]

        plot!(
            plt,
            x,
            summary.mean;
            ribbon=(summary.mean .- summary.lower, summary.upper .- summary.mean),
            color=:steelblue,
            fillalpha=0.20,
            linewidth=2.4,
            label=subplot_id == 1 ? "ESMDA mean and 90% interval" : "",
            subplot=subplot_id,
        )
        plot!(
            plt,
            x,
            adam_state[road_id];
            color=:firebrick,
            linewidth=2.2,
            linestyle=:dashdot,
            label=subplot_id == 1 ? "MAP estimate" : "",
            subplot=subplot_id,
        )
        plot!(
            plt,
            x,
            true_state[road_id];
            color=:black,
            linewidth=2.5,
            label=subplot_id == 1 ? "Reference state" : "",
            subplot=subplot_id,
        )

        if road_id in setup.observed_road_ids
            measurement_x = x[road_sensor_cell_ids(setup, road_id)]
            vline!(
                plt,
                measurement_x;
                color=:goldenrod3,
                alpha=0.72,
                linestyle=:dot,
                linewidth=2.0,
                label=subplot_id == first_observed_subplot ? "Sensor cells" : "",
                subplot=subplot_id,
            )
        end

        plot!(
            plt;
            xlabel="Position (m)",
            ylabel="Density",
            ylims=(0.0, 1.0),
            title=road_status_title(setup, road_id),
            subplot=subplot_id,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function select_representative_junction(entry_mean::AbstractVector, entry_true::AbstractVector)
    entries_per_junction = 16
    scores = [
        mean(abs.(entry_mean[(entries_per_junction * (junction - 1) + 1):(entries_per_junction * junction)] .-
                  entry_true[(entries_per_junction * (junction - 1) + 1):(entries_per_junction * junction)]))
        for junction in 1:N_JUNCTIONS
    ]
    return argmax(scores), scores
end

function plot_representative_turning_boxplot(
    esmda,
    entry_adam::AbstractVector;
    junction::Int,
    output_path=nothing,
)
    stats = summarize_square_turning_fraction_samples(esmda.fraction_samples, esmda.weights)
    labels = square_turning_entry_labels()
    entries_per_junction = 16
    start_idx = entries_per_junction * (junction - 1) + 1
    indices = start_idx:(start_idx + entries_per_junction - 1)
    x = collect(1:entries_per_junction)
    box_half_width = 0.28
    whisker_half_width = 0.14

    plt = plot(
        size=(1350, 620),
        dpi=220,
        legend=:top,
        left_margin=8Plots.mm,
        right_margin=4Plots.mm,
        bottom_margin=10Plots.mm,
        top_margin=5Plots.mm,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
    )

    for (local_idx, global_idx) in enumerate(indices)
        xi = x[local_idx]
        lower90 = esmda.entry_ci_05[global_idx]
        upper90 = esmda.entry_ci_95[global_idx]

        plot!(
            plt,
            [xi, xi],
            [lower90, upper90];
            color=:gray45,
            linewidth=1.7,
            label=local_idx == 1 ? "ESMDA 90% interval" : "",
        )
        plot!(plt, [xi - whisker_half_width, xi + whisker_half_width], [lower90, lower90]; color=:gray45, linewidth=1.7, label="")
        plot!(plt, [xi - whisker_half_width, xi + whisker_half_width], [upper90, upper90]; color=:gray45, linewidth=1.7, label="")
        plot!(
            plt,
            Plots.Shape(
                [xi - box_half_width, xi + box_half_width, xi + box_half_width, xi - box_half_width],
                [stats.q1[global_idx], stats.q1[global_idx], stats.q3[global_idx], stats.q3[global_idx]],
            );
            color=:steelblue,
            fillalpha=0.22,
            linecolor=:steelblue4,
            linewidth=1.6,
            label=local_idx == 1 ? "ESMDA 50% interval" : "",
        )
        scatter!(
            plt,
            [xi],
            [stats.means[global_idx]];
            color=:steelblue4,
            markershape=:circle,
            markersize=5,
            label=local_idx == 1 ? "ESMDA mean" : "",
        )
        scatter!(
            plt,
            [xi],
            [entry_adam[global_idx]];
            color=:firebrick,
            markershape=:diamond,
            markersize=5,
            label=local_idx == 1 ? "MAP estimate" : "",
        )
        scatter!(
            plt,
            [xi],
            [esmda.entry_true[global_idx]];
            color=:black,
            markershape=:star5,
            markersize=7,
            label=local_idx == 1 ? "Reference value" : "",
        )
    end

    plot!(
        plt;
        xlabel="Turning coefficient",
        ylabel="Turning fraction",
        xticks=(x, labels),
        xrotation=45,
        xlims=(0.4, entries_per_junction + 0.6),
        ylims=(0.0, 1.0),
    )

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function run_square_multi_scenario_representative_figures(;
    seed=1,
    output_dir=REPRESENTATIVE_OUTPUT_DIR,
)
    mkpath(output_dir)
    P_true = true_turning_matrices()
    dataset = build_representative_dataset(REPRESENTATIVE_HORIZON_MINUTES)
    setup = only(dataset.setups)
    observations = generate_physical_dataset_observations(P_true, dataset; seed=seed)

    println("Square-network representative reconstruction figures")
    println("-----------------------------------------------------")
    println(@sprintf("horizon: %.1f minutes", REPRESENTATIVE_HORIZON_MINUTES))
    println(@sprintf("observations: %d", dataset_observation_length(dataset)))
    println(@sprintf("peak noise scale: %.4f", setup.physical_noise_peak_sigma))

    esmda = run_esmda_multi_scenario(
        dataset,
        observations.y_obs,
        observations.sigma_model;
        seed=seed,
        prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
        ensemble_size=MULTI_SCENARIO_ESMDA_ENSEMBLE_SIZE,
        esmda_maxiters=MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
        P_true=P_true,
    )
    adam = run_adam_map_multi_scenario(
        observations.y_obs,
        dataset,
        observations.sigma_model;
        prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
        maxiters=REPRESENTATIVE_ADAM_ITERS,
    )

    y_true = observations.y_true
    metric_rows = [
        multi_scenario_metric_row(
            dataset,
            :adam,
            "adam_$(REPRESENTATIVE_ADAM_ITERS)_iters",
            REPRESENTATIVE_ADAM_ITERS,
            adam.solve_seconds,
            adam.iterations,
            adam.P_est,
            adam.y_est,
            P_true,
            y_true,
        ),
        multi_scenario_metric_row(
            dataset,
            :esmda,
            "esmda_$(MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS)_iters",
            MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
            esmda.solve_seconds,
            MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
            esmda.P_post_mean,
            esmda.y_post_mean,
            P_true,
            y_true,
            state_metrics=ensemble_mean_state_metrics(esmda, dataset, P_true),
        ),
    ]
    metrics_path = write_namedtuple_table(metric_rows, joinpath(output_dir, "square_representative_metrics.tsv"))

    true_state = final_state_snapshot(P_true, setup)
    adam_state = final_state_snapshot(adam.P_est, setup)
    esmda_summary = summarize_selected_final_states(esmda.param_samples, esmda.weights, setup, all_square_road_ids())
    representative_roads, road_error_rows, selected_road_rows = select_representative_roads(
        setup,
        true_state,
        adam_state,
        esmda_summary,
    )

    final_state_path = joinpath(output_dir, "square_representative_final_state.png")
    plot_representative_final_state(
        setup,
        true_state,
        adam_state,
        esmda_summary;
        road_ids=representative_roads,
        output_path=final_state_path,
    )

    entry_adam = turning_entries(adam.P_est)
    selected_junction, junction_scores = select_representative_junction(esmda.entry_post_mean, esmda.entry_true)
    turning_path = joinpath(output_dir, "square_representative_turning_junction.png")
    plot_representative_turning_boxplot(
        esmda,
        entry_adam;
        junction=selected_junction,
        output_path=turning_path,
    )

    selection_path = joinpath(output_dir, "square_representative_selection.txt")
    open(selection_path, "w") do io
        println(io, @sprintf("horizon_minutes\t%.1f", REPRESENTATIVE_HORIZON_MINUTES))
        println(io, @sprintf("observation_count\t%d", dataset_observation_length(dataset)))
        println(io, @sprintf("selected_junction\t%d", selected_junction))
        println(io, @sprintf("selected_junction_label\t%s", JUNCTION_LABELS[selected_junction]))
        println(io, "mean_abs_errors\t", join([@sprintf("%.8f", value) for value in junction_scores], "\t"))
        println(io, "representative_roads\t", join(representative_roads, ","))
        println(io, "road_id\tobserved\tadam_rmse\tesmda_rmse\tdisplay_score")
        for row in selected_road_rows
            println(
                io,
                @sprintf(
                    "%d\t%s\t%.8f\t%.8f\t%.8f",
                    row.road_id,
                    row.observed ? "true" : "false",
                    row.adam_rmse,
                    row.esmda_rmse,
                    row.display_score,
                ),
            )
        end
    end

    write_namedtuple_table(road_error_rows, joinpath(output_dir, "square_representative_road_errors.tsv"))

    println()
    println("Outputs")
    println("-------")
    println(metrics_path)
    println(final_state_path)
    println(turning_path)
    println(selection_path)

    return (
        esmda=esmda,
        adam=adam,
        metric_rows=metric_rows,
        metrics_path=metrics_path,
        final_state_path=final_state_path,
        turning_path=turning_path,
        selection_path=selection_path,
        selected_junction=selected_junction,
        representative_roads=representative_roads,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_REPRESENTATIVE", "0") != "1"
    run_square_multi_scenario_representative_figures()
end
