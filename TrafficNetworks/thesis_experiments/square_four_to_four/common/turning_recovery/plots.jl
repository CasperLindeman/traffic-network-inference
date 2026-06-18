# Shared helpers for square-four-to-four turning-outlier diagnostics.

turning_row_source_label(row) = @sprintf("J%d r%d", row.junction, row.source_road)

function plot_error_vs_sensitivity(rows; output_path=nothing)
    classes = sort(unique(getproperty.(rows, :direct_observation_class)))
    colors = palette(:tab10, length(classes))
    plt = plot(
        size=(1050, 720),
        dpi=180,
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
        xlabel="log10 normalized observation sensitivity",
        ylabel="Absolute posterior mean error",
        title="Turning-fraction misses vs observation sensitivity",
        legend=:topright,
    )

    for (idx, class_label) in enumerate(classes)
        class_rows = [row for row in rows if row.direct_observation_class == class_label]
        sensitivities = max.(getproperty.(class_rows, :normalized_sensitivity), 1e-8)
        scatter!(
            plt,
            log10.(sensitivities),
            getproperty.(class_rows, :abs_error);
            color=colors[idx],
            markersize=6,
            alpha=0.82,
            label=class_label,
        )
    end

    top_rows = first(sort(rows; by=row -> -row.abs_error), min(8, length(rows)))
    for row in top_rows
        annotate!(
            plt,
            log10(max(row.normalized_sensitivity, 1e-8)),
            row.abs_error,
            text(row.entry_label, 7, :left),
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_error_by_direct_observation(rows; output_path=nothing)
    class_order = [
        "source+target observed",
        "source observed",
        "target observed",
        "neither directly observed",
    ]
    class_order = [label for label in class_order if any(row -> row.direct_observation_class == label, rows)]
    means = [
        mean([row.abs_error for row in rows if row.direct_observation_class == label])
        for label in class_order
    ]
    max_error = maximum(getproperty.(rows, :abs_error))

    plt = bar(
        1:length(class_order),
        means;
        color=:gray80,
        linecolor=:gray35,
        legend=false,
        size=(1050, 620),
        dpi=180,
        left_margin=8Plots.mm,
        bottom_margin=12Plots.mm,
        xticks=(1:length(class_order), class_order),
        xrotation=20,
        ylims=(0.0, max(0.05, 1.12 * max_error)),
        ylabel="Absolute posterior mean error",
        title="Turning-fraction error by direct road observation",
    )

    for (idx, label) in enumerate(class_order)
        class_rows = [row for row in rows if row.direct_observation_class == label]
        offsets = length(class_rows) == 1 ? [0.0] : collect(range(-0.24, 0.24; length=length(class_rows)))
        scatter!(
            plt,
            fill(idx, length(class_rows)) .+ offsets,
            getproperty.(class_rows, :abs_error);
            color=:steelblue4,
            markersize=5,
            alpha=0.84,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_abs_error_heatmaps(rows; output_path=nothing)
    max_error = max(maximum(getproperty.(rows, :abs_error)), 1e-8)
    plt = plot(
        layout=(2, 2),
        size=(980, 860),
        dpi=180,
        left_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
    )

    for junction in 1:N_JUNCTIONS
        matrix = fill(NaN, 4, 4)
        for row in rows
            row.junction == junction || continue
            matrix[row.incoming_row, row.outgoing_col] = row.abs_error
        end

        heatmap!(
            plt,
            1:4,
            1:4,
            matrix;
            subplot=junction,
            color=:viridis,
            clims=(0.0, max_error),
            colorbar=junction == N_JUNCTIONS,
            xlabel="outgoing column",
            ylabel="incoming row",
            xticks=1:4,
            yticks=1:4,
            yflip=true,
            title=JUNCTION_LABELS[junction],
        )

        for incoming_row in 1:4
            for outgoing_col in 1:4
                annotate!(
                    plt,
                    outgoing_col,
                    incoming_row,
                    text(@sprintf("%.2f", matrix[incoming_row, outgoing_col]), 8, :white);
                    subplot=junction,
                )
            end
        end
    end

    plot!(plt; plot_title="Absolute posterior mean error by turning entry")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_row_tv_errors(row_rows; output_path=nothing)
    ordered = sort(row_rows; by=row -> -row.row_tv_error)
    labels = [
        @sprintf("%s\nincoming %d", row.row_label, row.source_road)
        for row in ordered
    ]
    colors = [row.source_observed ? :steelblue4 : :firebrick3 for row in ordered]

    plt = bar(
        1:length(ordered),
        getproperty.(ordered, :row_tv_error);
        color=colors,
        linecolor=:gray30,
        legend=false,
        size=(1120, 620),
        dpi=180,
        left_margin=8Plots.mm,
        bottom_margin=13Plots.mm,
        xticks=(1:length(ordered), labels),
        xrotation=35,
        ylabel="Row total-variation error",
        title="Error of the whole turning row/source-road split",
    )

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_row_error_vs_observation_signal(row_rows; output_path=nothing)
    classes = ["source observed", "source unobserved"]
    colors = Dict("source observed" => :steelblue4, "source unobserved" => :firebrick3)
    plt = plot(
        size=(980, 660),
        dpi=180,
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
        xlabel="log10 row normalized observation signal",
        ylabel="Row total-variation error",
        title="Whole-row misses vs visibility of that row mistake",
        legend=:topright,
    )

    for class_label in classes
        class_rows = [
            row for row in row_rows
            if (row.source_observed ? "source observed" : "source unobserved") == class_label
        ]
        isempty(class_rows) && continue
        scatter!(
            plt,
            log10.(max.(getproperty.(class_rows, :row_normalized_observation_signal), 1e-8)),
            getproperty.(class_rows, :row_tv_error);
            color=colors[class_label],
            markersize=7,
            alpha=0.86,
            label=class_label,
        )
    end

    top_rows = first(sort(row_rows; by=row -> -row.row_tv_error), min(6, length(row_rows)))
    for row in top_rows
        annotate!(
            plt,
            log10(max(row.row_normalized_observation_signal, 1e-8)),
            row.row_tv_error,
            text(row.row_label, 7, :left),
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_worst_row_splits(row_rows; top_n=8, output_path=nothing)
    ordered = first(sort(row_rows; by=row -> -row.row_tv_error), min(top_n, length(row_rows)))
    layout = subplot_layout_for_count(length(ordered))
    plt = plot(
        layout=layout,
        size=(1500, 1000),
        dpi=180,
        left_margin=7Plots.mm,
        bottom_margin=7Plots.mm,
        legend=:top,
    )

    x = collect(1:4)
    for (subplot_id, row) in enumerate(ordered)
        bar!(
            plt,
            x .- 0.18,
            row.true_row_values;
            bar_width=0.34,
            color=:gray70,
            linecolor=:gray35,
            label=subplot_id == 1 ? "Truth" : "",
            subplot=subplot_id,
        )
        bar!(
            plt,
            x .+ 0.18,
            row.posterior_mean_row_values;
            bar_width=0.34,
            color=:steelblue4,
            linecolor=:steelblue4,
            alpha=0.75,
            label=subplot_id == 1 ? "Posterior mean" : "",
            subplot=subplot_id,
        )

        if !isempty(row.observed_target_col_values)
            scatter!(
                plt,
                row.observed_target_col_values,
                fill(1.03, length(row.observed_target_col_values));
                color=:black,
                markershape=:diamond,
                markersize=4,
                label=subplot_id == 1 ? "Observed target road" : "",
                subplot=subplot_id,
            )
        end

        plot!(
            plt;
            subplot=subplot_id,
            xticks=(x, [@sprintf("c%d\nr%d", col, row.target_road_values[col]) for col in x]),
            ylims=(0.0, 1.08),
            xlabel="target column / road",
            ylabel="fraction",
            title=@sprintf(
                "%s source road %d %s | TV %.3f",
                row.row_label,
                row.source_road,
                row.source_observed ? "observed" : "unobserved",
                row.row_tv_error,
            ),
        )
    end

    plot!(plt; plot_title="Worst whole-row turning split errors")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_row_swap_fit_improvement(swap_rows; output_path=nothing)
    ordered = sort(swap_rows; by=row -> -row.row_tv_error)
    labels = [turning_row_source_label(row) for row in ordered]
    x = collect(1:length(ordered))

    plt = plot(
        layout=(2, 1),
        size=(1180, 820),
        dpi=180,
        left_margin=8Plots.mm,
        bottom_margin=12Plots.mm,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
        legend=false,
    )

    bar!(
        plt,
        x,
        getproperty.(ordered, :truth_rmse_improvement);
        color=:steelblue4,
        linecolor=:gray35,
        ylabel="RMSE improvement",
        title="(a) Noiseless synthetic observations",
        subplot=1,
    )
    hline!(plt, [0.0]; color=:gray40, linestyle=:dash, subplot=1)

    bar!(
        plt,
        x,
        getproperty.(ordered, :fit_normalized_improvement);
        color=:darkorange3,
        linecolor=:gray35,
        ylabel="RMSE improvement",
        title="(b) Weighted noisy-data fit",
        xticks=(x, labels),
        xrotation=35,
        subplot=2,
    )
    hline!(plt, [0.0]; color=:gray40, linestyle=:dash, subplot=2)

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

