using Plots

function plot_external_initial_profiles(setup::SquareFourToFourSetup; output_path=nothing)
    road_ids = vcat(EXTERNAL_INCOMING_ROADS, EXTERNAL_OUTGOING_ROADS)
    plt = plot(layout=(4, 4), size=(1450, 1100), dpi=180, legend=:top)

    for (subplot_id, road_id) in enumerate(road_ids)
        x_m = road_cell_centers_meters(setup, road_id)
        rho0 = road_initial_profile(setup, road_id)
        color = road_id in EXTERNAL_INCOMING_ROADS ? :darkorange3 : :seagreen4
        plot!(
            plt,
            x_m,
            rho0;
            color=color,
            linewidth=2.4,
            xlabel="Position (m)",
            ylabel="Density",
            ylims=(0.0, 1.0),
            title=road_label(road_id),
            label=subplot_id == 1 ? "Initial density" : "",
            subplot=subplot_id,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function subplot_layout_for_count(nplots::Int)
    if nplots <= 4
        return (2, 2)
    elseif nplots <= 8
        return (4, 2)
    elseif nplots <= 12
        return (4, 3)
    elseif nplots <= 16
        return (4, 4)
    elseif nplots <= 20
        return (5, 4)
    else
        return (6, 4)
    end
end

function plot_boundary_conditions(setup::SquareFourToFourSetup; output_path=nothing)
    times = collect(range(0.0, setup.T; length=400))
    plt = plot(layout=(4, 2), size=(1200, 1050), dpi=180, legend=:top)
    colors = palette(:tab10, length(setup.inflows))

    for k in eachindex(setup.inflows)
        road_id = setup.boundary_road_ids[k]
        vals = setup.inflows[k].(times)
        plot!(
            plt,
            60.0 .* times,
            vals;
            color=colors[k],
            linewidth=2.4,
            xlabel="Time (min)",
            ylabel="Inflow flux",
            title=road_label(road_id),
            label=k == 1 ? "Boundary inflow" : "",
            subplot=k,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_observed_road_means(y_true, y_esmda, y_adam, setup::SquareFourToFourSetup; output_path=nothing)
    Y_true = reshape_observations(y_true, setup)
    Y_esmda = reshape_observations(y_esmda, setup)
    Y_adam = reshape_observations(y_adam, setup)

    plt = plot(layout=(4, 2), size=(1200, 1050), dpi=180, legend=:topright)
    t_min = 60.0 .* setup.control_times

    for road_pos in eachindex(setup.observed_road_ids)
        road_id = setup.observed_road_ids[road_pos]
        true_mean = vec(mean(Y_true[:, :, road_pos]; dims=1))
        esmda_mean = vec(mean(Y_esmda[:, :, road_pos]; dims=1))
        adam_mean = vec(mean(Y_adam[:, :, road_pos]; dims=1))

        plot!(
            plt,
            t_min,
            true_mean;
            color=:black,
            linewidth=2.6,
            xlabel="Time (min)",
            ylabel="Mean observed density",
            title=road_label(road_id),
            label=road_pos == 1 ? "Truth" : "",
            subplot=road_pos,
        )
        plot!(
            plt,
            t_min,
            esmda_mean;
            color=:steelblue,
            linewidth=2.2,
            linestyle=:dash,
            label=road_pos == 1 ? "ESMDA" : "",
            subplot=road_pos,
        )
        plot!(
            plt,
            t_min,
            adam_mean;
            color=:firebrick,
            linewidth=2.2,
            linestyle=:dot,
            label=road_pos == 1 ? "ADAM" : "",
            subplot=road_pos,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_turning_matrix_heatmaps(P_true, P_esmda, P_adam; output_path=nothing)
    plt = plot(layout=(N_JUNCTIONS, 3), size=(1150, 1300), dpi=180)
    titles = ["Truth", "ESMDA", "ADAM"]
    matrices_per_column = [turning_matrices(P_true), turning_matrices(P_esmda), turning_matrices(P_adam)]

    for junction in 1:N_JUNCTIONS
        for col in 1:3
            subplot_id = 3 * (junction - 1) + col
            heatmap!(
                plt,
                1:4,
                1:4,
                matrices_per_column[col][junction];
                color=:viridis,
                clims=(0.0, 0.65),
                xlabel="Outgoing",
                ylabel="Incoming",
                title="$(JUNCTION_LABELS[junction]) - $(titles[col])",
                xticks=(1:4, ["1", "2", "3", "4"]),
                yticks=(1:4, ["1", "2", "3", "4"]),
                colorbar=col == 3,
                aspect_ratio=1,
                subplot=subplot_id,
            )
        end
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_turning_fraction_bars(P_true, P_esmda_mean, P_esmda_lower, P_esmda_upper, P_adam; output_path=nothing)
    true_mats = turning_matrices(P_true)
    esmda_mats = turning_matrices(P_esmda_mean)
    lower_mats = P_esmda_lower
    upper_mats = P_esmda_upper
    adam_mats = turning_matrices(P_adam)

    plt = plot(layout=(N_JUNCTIONS, 4), size=(1500, 1300), dpi=180, legend=:top)

    for junction in 1:N_JUNCTIONS
        for incoming_row in 1:4
            subplot_id = 4 * (junction - 1) + incoming_row
            x = 1:4
            mean_vals = esmda_mats[junction][incoming_row, :]
            lower_vals = lower_mats[junction][incoming_row, :]
            upper_vals = upper_mats[junction][incoming_row, :]

            bar!(
                plt,
                x,
                mean_vals;
                yerror=(mean_vals .- lower_vals, upper_vals .- mean_vals),
                color=:steelblue,
                alpha=0.75,
                xlabel="Outgoing road",
                ylabel="Turning fraction",
                title="$(JUNCTION_LABELS[junction]), incoming $(incoming_row)",
                xticks=(1:4, ["1", "2", "3", "4"]),
                ylims=(0.0, 0.75),
                label=subplot_id == 1 ? "ESMDA mean +/- 90%" : "",
                subplot=subplot_id,
            )
            scatter!(
                plt,
                x,
                true_mats[junction][incoming_row, :];
                color=:black,
                marker=:star5,
                markersize=6,
                label=subplot_id == 1 ? "Truth" : "",
                subplot=subplot_id,
            )
            scatter!(
                plt,
                x,
                adam_mats[junction][incoming_row, :];
                color=:firebrick,
                marker=:diamond,
                markersize=5,
                label=subplot_id == 1 ? "ADAM" : "",
                subplot=subplot_id,
            )
        end
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

square_turning_entry_labels() = ["α$(row)$(col)" for row in 1:4 for col in 1:4]

square_road_status_title(road_id::Int, observed_road_ids) =
    "r$(road_id) ($(road_id in observed_road_ids ? "observed" : "unobserved"))"

function summarize_square_turning_fraction_samples(fraction_samples::AbstractMatrix, weights::AbstractVector)
    @assert size(fraction_samples, 2) == length(weights)

    weights_norm = normalize_weights(weights)
    n_entries = size(fraction_samples, 1)

    mins = Vector{Float64}(undef, n_entries)
    q1 = Vector{Float64}(undef, n_entries)
    medians = Vector{Float64}(undef, n_entries)
    q3 = Vector{Float64}(undef, n_entries)
    maxs = Vector{Float64}(undef, n_entries)

    for idx in 1:n_entries
        draws = vec(fraction_samples[idx, :])
        mins[idx] = minimum(draws)
        q1[idx] = weighted_quantile(draws, weights_norm, 0.25)
        medians[idx] = weighted_quantile(draws, weights_norm, 0.50)
        q3[idx] = weighted_quantile(draws, weights_norm, 0.75)
        maxs[idx] = maximum(draws)
    end

    return (
        mins=mins,
        q1=q1,
        medians=medians,
        q3=q3,
        maxs=maxs,
        means=weighted_column_mean(fraction_samples, weights_norm),
    )
end

function plot_turning_fraction_uncertainty_by_junction(
    fraction_samples::AbstractMatrix,
    weights::AbstractVector,
    entry_true::AbstractVector,
    entry_adam::AbstractVector;
    output_path=nothing,
)
    labels = square_turning_entry_labels()
    entries_per_junction = length(labels)
    total_entries = entries_per_junction * N_JUNCTIONS

    @assert size(fraction_samples, 1) == total_entries
    @assert length(entry_true) == total_entries
    @assert length(entry_adam) == total_entries

    stats = summarize_square_turning_fraction_samples(fraction_samples, weights)
    x = collect(1:entries_per_junction)
    box_half_width = 0.28
    whisker_half_width = 0.14

    plt = plot(
        layout=(N_JUNCTIONS, 1),
        size=(1650, 1250),
        dpi=180,
        legend=:top,
    )

    for junction in 1:N_JUNCTIONS
        start_idx = entries_per_junction * (junction - 1) + 1
        end_idx = start_idx + entries_per_junction - 1

        for local_idx in 1:entries_per_junction
            global_idx = start_idx + local_idx - 1
            xi = x[local_idx]

            plot!(
                plt,
                [xi, xi],
                [stats.mins[global_idx], stats.maxs[global_idx]];
                color=:gray55,
                linewidth=1.6,
                label=junction == 1 && local_idx == 1 ? "Min-max" : "",
                subplot=junction,
            )
            plot!(
                plt,
                [xi - whisker_half_width, xi + whisker_half_width],
                [stats.mins[global_idx], stats.mins[global_idx]];
                color=:gray55,
                linewidth=1.6,
                label="",
                subplot=junction,
            )
            plot!(
                plt,
                [xi - whisker_half_width, xi + whisker_half_width],
                [stats.maxs[global_idx], stats.maxs[global_idx]];
                color=:gray55,
                linewidth=1.6,
                label="",
                subplot=junction,
            )
            plot!(
                plt,
                Plots.Shape(
                    [xi - box_half_width, xi + box_half_width, xi + box_half_width, xi - box_half_width],
                    [stats.q1[global_idx], stats.q1[global_idx], stats.q3[global_idx], stats.q3[global_idx]],
                );
                color=:steelblue,
                fillalpha=0.20,
                linecolor=:steelblue4,
                linewidth=1.6,
                label=junction == 1 && local_idx == 1 ? "Q1-Q3" : "",
                subplot=junction,
            )
            plot!(
                plt,
                [xi - box_half_width, xi + box_half_width],
                [stats.medians[global_idx], stats.medians[global_idx]];
                color=:navy,
                linewidth=2.0,
                label=junction == 1 && local_idx == 1 ? "Median" : "",
                subplot=junction,
            )
            scatter!(
                plt,
                [xi],
                [stats.means[global_idx]];
                color=:steelblue4,
                markershape=:circle,
                markersize=5,
                label=junction == 1 && local_idx == 1 ? "ESMDA mean" : "",
                subplot=junction,
            )
            scatter!(
                plt,
                [xi],
                [entry_adam[global_idx]];
                color=:firebrick,
                markershape=:diamond,
                markersize=5,
                label=junction == 1 && local_idx == 1 ? "ADAM" : "",
                subplot=junction,
            )
            scatter!(
                plt,
                [xi],
                [entry_true[global_idx]];
                color=:black,
                markershape=:star5,
                markersize=7,
                label=junction == 1 && local_idx == 1 ? "Truth" : "",
                subplot=junction,
            )
        end

        plot!(
            plt;
            xlabel="Turning-fraction coefficient",
            ylabel="Value",
            xticks=(x, labels),
            xrotation=45,
            xlims=(0.4, entries_per_junction + 0.6),
            ylims=(0.0, 1.0),
            title="$(JUNCTION_LABELS[junction]) coefficient uncertainty",
            subplot=junction,
        )
    end

    plot!(plt; plot_title="Turning-fraction uncertainty across ESMDA posterior samples")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_final_state_comparison(
    true_state::AbstractMatrix,
    setup::SquareFourToFourSetup;
    road_ids=setup.observed_road_ids,
    prior_summary::Union{Nothing, FinalStateSummary}=nothing,
    posterior_summary::Union{Nothing, FinalStateSummary}=nothing,
    adam_state::Union{Nothing, AbstractMatrix}=nothing,
    interval_level=0.90,
    output_path=nothing,
)
    interval_pct = Int(round(100 * interval_level))
    layout = subplot_layout_for_count(length(road_ids))
    plt = plot(layout=layout, size=(1600, 1800), dpi=180, legend=:top)
    first_observed_subplot = findfirst(road_id -> road_id in setup.observed_road_ids, road_ids)

    for (subplot_id, road_id) in enumerate(road_ids)
        x = road_cell_centers_meters(setup, road_id)

        if prior_summary !== nothing
            prior_mean = prior_summary.mean[:, road_id]
            prior_lower = prior_summary.lower[:, road_id]
            prior_upper = prior_summary.upper[:, road_id]
            plot!(
                plt,
                x,
                prior_mean;
                ribbon=(prior_mean .- prior_lower, prior_upper .- prior_mean),
                color=:gray50,
                fillalpha=0.17,
                linestyle=:dash,
                linewidth=2.0,
                label=subplot_id == 1 ? "Prior mean +/- $(interval_pct)%" : "",
                subplot=subplot_id,
            )
        end

        if posterior_summary !== nothing
            post_mean = posterior_summary.mean[:, road_id]
            post_lower = posterior_summary.lower[:, road_id]
            post_upper = posterior_summary.upper[:, road_id]
            plot!(
                plt,
                x,
                post_mean;
                ribbon=(post_mean .- post_lower, post_upper .- post_mean),
                color=:steelblue,
                fillalpha=0.20,
                linewidth=2.4,
                label=subplot_id == 1 ? "ESMDA mean +/- $(interval_pct)%" : "",
                subplot=subplot_id,
            )
        end

        if adam_state !== nothing
            plot!(
                plt,
                x,
                adam_state[:, road_id];
                color=:firebrick,
                linewidth=2.2,
                linestyle=:dashdot,
                label=subplot_id == 1 ? "ADAM estimate" : "",
                subplot=subplot_id,
            )
        end

        plot!(
            plt,
            x,
            true_state[:, road_id];
            color=:black,
            linewidth=2.5,
            xlabel="Position (m)",
            ylabel="Density",
            ylims=(0.0, 1.0),
            title=square_road_status_title(road_id, setup.observed_road_ids),
            label=subplot_id == 1 ? "True final state" : "",
            subplot=subplot_id,
        )

        if road_id in setup.observed_road_ids
            measurement_x = x[setup.observed_cell_ids]
            vline!(
                plt,
                measurement_x;
                color=:goldenrod3,
                alpha=0.72,
                linestyle=:dot,
                linewidth=2.0,
                label=subplot_id == first_observed_subplot ? "Observed cells" : "",
                subplot=subplot_id,
            )
        end
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_state_snapshot_comparison(snapshot_true, snapshot_esmda, snapshot_adam, setup::SquareFourToFourSetup; time_label="snapshot", output_path=nothing)
    road_ids = collect(1:length(ROAD_LABELS))
    layout = subplot_layout_for_count(length(road_ids))
    plt = plot(layout=layout, size=(1600, 1800), dpi=180, legend=:top)

    for (subplot_id, road_id) in enumerate(road_ids)
        x = road_cell_centers_meters(setup, road_id)
        plot!(
            plt,
            x,
            snapshot_true[:, road_id];
            color=:black,
            linewidth=2.4,
            xlabel="Position (m)",
            ylabel="Density",
            ylims=(0.0, 1.0),
            title=road_label(road_id),
            label=subplot_id == 1 ? "Truth" : "",
            subplot=subplot_id,
        )
        plot!(
            plt,
            x,
            snapshot_esmda[:, road_id];
            color=:steelblue,
            linewidth=2.0,
            linestyle=:dash,
            label=subplot_id == 1 ? "ESMDA" : "",
            subplot=subplot_id,
        )
        plot!(
            plt,
            x,
            snapshot_adam[:, road_id];
            color=:firebrick,
            linewidth=2.0,
            linestyle=:dot,
            label=subplot_id == 1 ? "ADAM" : "",
            subplot=subplot_id,
        )
    end

    plot!(plt; plot_title="All-road state comparison at $(time_label)")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_adam_diagnostics(adam_results; output_path=nothing)
    iterations = collect(1:adam_results.iterations)
    plt = plot(layout=(3, 1), size=(900, 950), dpi=180, legend=:topright)

    plot!(
        plt, iterations, adam_results.losses;
        color=:firebrick,
        linewidth=2.2,
        xlabel="Iteration",
        ylabel="MAP objective",
        title="ADAM loss trace",
        label="Loss",
        yaxis=:log10,
        subplot=1
    )

    plot!(
        plt, iterations, adam_results.raw_grad_norms;
        color=:navy,
        linewidth=2.2,
        xlabel="Iteration",
        ylabel="Gradient norm",
        title="ADAM raw gradient norm",
        label="Raw gradient",
        subplot=2
    )
    plot!(
        plt,
        iterations,
        adam_results.grad_norms;
        color=:firebrick,
        linewidth=2.0,
        xlabel="Iteration",
        ylabel="Post-clip norm",
        title="ADAM clipped gradient norm",
        label="After clipping",
        subplot=3,
    )

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_adam_restart_summary(adam_restarts, P_true, y_true; output_path=nothing)
    labels = [restart.label for restart in adam_restarts]
    turning_vals = [overall_turning_rmse(restart.adam.P_est, P_true) for restart in adam_restarts]
    predictive_vals = [predictive_rmse(restart.adam.y_est, y_true) for restart in adam_restarts]
    loss_vals = [restart.adam.best_loss for restart in adam_restarts]

    plt = plot(layout=(3, 1), size=(900, 900), dpi=180, legend=false)
    bar!(plt, labels, turning_vals; color=:steelblue, ylabel="Turning RMSE", title="ADAM restart turning error", subplot=1)
    bar!(plt, labels, predictive_vals; color=:seagreen4, ylabel="Predictive RMSE", title="ADAM restart predictive error", subplot=2)
    bar!(plt, labels, loss_vals; color=:firebrick, ylabel="Best loss", title="ADAM restart MAP objective", subplot=3)

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end
