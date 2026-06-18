using Plots

function apply_four_to_four_esmda_adam_plot_style!()
    default(
        tickfontsize=13,
        guidefontsize=15,
        legendfontsize=12,
        titlefontsize=18,
        linewidth=2.4,
        markersize=6,
    )
end

function plot_final_state_overview(
    true_state::AbstractMatrix,
    setup::FourToFourInferenceSetup;
    prior_summary::Union{Nothing, FinalStateSummary}=nothing,
    posterior_summary::Union{Nothing, FinalStateSummary}=nothing,
    adam_state::Union{Nothing, AbstractMatrix}=nothing,
    interval_level=0.90,
    output_path=nothing,
)
    _, n_roads = size(true_state)
    interval_pct = Int(round(100 * interval_level))
    first_observed_road = first(setup.observed_road_ids)

    plt = plot(
        layout=(4, 2),
        size=(1350, 1200),
        dpi=220,
        legend=:top,
        left_margin=8Plots.mm,
        right_margin=4Plots.mm,
        top_margin=5Plots.mm,
        bottom_margin=7Plots.mm,
    )

    for road_id in 1:n_roads
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
                fillalpha=0.18,
                linestyle=:dash,
                linewidth=2,
                label=road_id == 1 ? "Prior mean and $(interval_pct)% interval" : "",
                subplot=road_id,
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
                linewidth=2.5,
                label=road_id == 1 ? "ESMDA mean and $(interval_pct)% interval" : "",
                subplot=road_id,
            )
        end

        if adam_state !== nothing
            plot!(
                plt,
                x,
                adam_state[:, road_id];
                color=:firebrick,
                linewidth=2.5,
                linestyle=:dashdot,
                label=road_id == 1 ? "MAP estimate" : "",
                subplot=road_id,
            )
        end

        plot!(
            plt,
            x,
            true_state[:, road_id];
            color=:black,
            linewidth=2.5,
            label=road_id == 1 ? "Reference state" : "",
            xlabel="Position (m)",
            ylabel="Density",
            title=observed_road_label(road_id, setup),
            ylims=(0.0, 1.0),
            subplot=road_id,
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
                label=road_id == first_observed_road ? "Sensor cells" : "",
                subplot=road_id,
            )
        end
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

turning_entry_labels() = ["α$(row)$(col)" for row in 1:4 for col in 1:4]

function summarize_turning_fraction_samples(fraction_samples::AbstractMatrix, weights::AbstractVector)
    @assert size(fraction_samples, 2) == length(weights)

    weights_norm = TrafficNetworks.normalize_weights(weights)
    n_entries = size(fraction_samples, 1)

    mins = Vector{Float64}(undef, n_entries)
    q05 = Vector{Float64}(undef, n_entries)
    q1 = Vector{Float64}(undef, n_entries)
    q3 = Vector{Float64}(undef, n_entries)
    q95 = Vector{Float64}(undef, n_entries)
    maxs = Vector{Float64}(undef, n_entries)

    for idx in 1:n_entries
        draws = vec(fraction_samples[idx, :])
        mins[idx] = minimum(draws)
        q05[idx] = weighted_quantile(draws, weights_norm, 0.05)
        q1[idx] = weighted_quantile(draws, weights_norm, 0.25)
        q3[idx] = weighted_quantile(draws, weights_norm, 0.75)
        q95[idx] = weighted_quantile(draws, weights_norm, 0.95)
        maxs[idx] = maximum(draws)
    end

    return (
        mins=mins,
        q05=q05,
        q1=q1,
        q3=q3,
        q95=q95,
        maxs=maxs,
        means=weighted_column_mean(fraction_samples, weights_norm),
    )
end

function plot_turning_fraction_uncertainty(
    fraction_samples::AbstractMatrix,
    weights::AbstractVector,
    entry_true::AbstractVector,
    entry_adam::AbstractVector;
    output_path=nothing,
)
    stats = summarize_turning_fraction_samples(fraction_samples, weights)
    labels = turning_entry_labels()
    @assert size(fraction_samples, 1) == length(labels)
    @assert length(entry_true) == length(labels)
    @assert length(entry_adam) == length(labels)

    x = collect(1:length(labels))
    box_half_width = 0.28
    whisker_half_width = 0.14

    plt = plot(
        size=(1600, 650),
        dpi=220,
        legend=:top,
        xlabel="Turning coefficient",
        ylabel="Turning fraction",
        xticks=(x, labels),
        xrotation=45,
        xlims=(0.4, length(labels) + 0.6),
        ylims=(0.0, 1.0),
        left_margin=8Plots.mm,
        bottom_margin=14Plots.mm,
        top_margin=5Plots.mm,
        right_margin=4Plots.mm,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
    )

    for idx in eachindex(x)
        xi = x[idx]
        plot!(
            plt,
            [xi, xi],
            [stats.q05[idx], stats.q95[idx]];
            color=:gray45,
            linewidth=1.7,
            label=idx == 1 ? "ESMDA 90% interval" : "",
        )
        plot!(
            plt,
            [xi - whisker_half_width, xi + whisker_half_width],
            [stats.q05[idx], stats.q05[idx]];
            color=:gray45,
            linewidth=1.7,
            label="",
        )
        plot!(
            plt,
            [xi - whisker_half_width, xi + whisker_half_width],
            [stats.q95[idx], stats.q95[idx]];
            color=:gray45,
            linewidth=1.7,
            label="",
        )
        plot!(
            plt,
            Plots.Shape(
                [xi - box_half_width, xi + box_half_width, xi + box_half_width, xi - box_half_width],
                [stats.q1[idx], stats.q1[idx], stats.q3[idx], stats.q3[idx]],
            );
            color=:steelblue,
            fillalpha=0.22,
            linecolor=:steelblue4,
            linewidth=1.6,
            label=idx == 1 ? "ESMDA 50% interval" : "",
        )
        scatter!(
            plt,
            [xi],
            [stats.means[idx]];
            color=:steelblue4,
            markershape=:circle,
            markersize=5,
            label=idx == 1 ? "ESMDA mean" : "",
        )
        scatter!(
            plt,
            [xi],
            [entry_adam[idx]];
            color=:firebrick,
            markershape=:diamond,
            markersize=5,
            label=idx == 1 ? "MAP estimate" : "",
        )
        scatter!(
            plt,
            [xi],
            [entry_true[idx]];
            color=:black,
            markershape=:star5,
            markersize=7,
            label=idx == 1 ? "Reference value" : "",
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end
