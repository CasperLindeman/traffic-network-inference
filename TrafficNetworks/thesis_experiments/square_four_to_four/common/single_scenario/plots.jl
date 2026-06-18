# Shared helpers for square-four-to-four single-scenario experiments.

function plot_adam_trace_panels(results, labels;
    normalize_losses=false,
    plot_title="ADAM diagnostics",
    output_path=nothing,
)
    colors = [:gray45, :firebrick, :steelblue, :darkorange3, :seagreen4]
    plt = plot(layout=(2, 2), size=(1300, 900), dpi=180, legend=:topright, left_margin=8Plots.mm, bottom_margin=8Plots.mm)

    for (idx, result) in enumerate(results)
        x = 1:result.iterations
        loss_vals = normalize_losses ? result.losses ./ max(result.losses[1], 1e-12) : result.losses
        loss_plot_vals = max.(loss_vals, 1e-12)
        color = colors[idx]

        plot!(plt, x, loss_plot_vals; color=color, linewidth=2.2, yscale=:log10, xlabel="Iteration", ylabel=normalize_losses ? "Loss / loss(1) [log]" : "Loss [log]", title="Loss trace (log scale)", label=labels[idx], subplot=1)
        plot!(plt, x, max.(result.raw_grad_norms, 1e-12); color=color, linewidth=2.2, yscale=:log10, xlabel="Iteration", ylabel="Raw gradient norm", title="Raw gradient norm", label=labels[idx], subplot=2)
        plot!(plt, x, result.learning_rates; color=color, linewidth=2.2, xlabel="Iteration", ylabel="Learning rate", title="Learning-rate schedule", label=labels[idx], subplot=3)
        plot!(plt, x, cumsum(Float64.(result.clipped_flags)) ./ x; color=color, linewidth=2.2, xlabel="Iteration", ylabel="Cumulative clip fraction", title="Gradient clipping activity", label=labels[idx], subplot=4)
    end

    plot!(plt; plot_title=plot_title)

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_gradient_component_bars(rows; output_path=nothing)
    point_labels = unique(getproperty.(rows, :point_label))
    component_order = ["incoming", "outgoing", "internal", "prior", "data_total", "objective_total"]
    plt = plot(layout=(1, length(point_labels)), size=(1400, 420), dpi=180, legend=false, left_margin=8Plots.mm, bottom_margin=8Plots.mm)

    for (subplot_id, point_label) in enumerate(point_labels)
        point_rows = filter(row -> row.point_label == point_label, rows)
        values = [first(filter(row -> row.component == component, point_rows)).grad_norm for component in component_order]
        bar!(plt, 1:length(component_order), values; color=:steelblue, xticks=(1:length(component_order), component_order), xrotation=25, ylabel="Gradient norm", title=point_label, subplot=subplot_id)
    end

    plot!(plt; plot_title="MAP gradient-component diagnostics")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_loss_comparison_restarts(rows; output_path=nothing)
    loss_labels = unique(getproperty.(rows, :loss_kind))
    restart_labels = unique(getproperty.(rows, :restart_label))
    color_map = Dict("map" => :firebrick, "mse" => :steelblue)

    specs = [
        (:turning_rmse, "Turning-fraction RMSE", "RMSE"),
        (:final_state_rmse_all, "Final-state RMSE", "RMSE"),
        (:fit_rmse, "Fit RMSE to noisy observations", "RMSE"),
        (:final_raw_grad_norm, "Final raw gradient norm", "Norm"),
    ]

    plt = plot(layout=(2, 2), size=(1320, 980), dpi=180, legend=:topright, left_margin=8Plots.mm, bottom_margin=8Plots.mm)

    for (subplot_id, (key, title_text, ylabel_text)) in enumerate(specs)
        for (loss_idx, loss_label) in enumerate(loss_labels)
            group_rows = filter(row -> row.loss_kind == loss_label, rows)
            ys = [first(filter(row -> row.restart_label == restart_label, group_rows))[key] for restart_label in restart_labels]
            xs = collect(1:length(restart_labels)) .+ 0.12 * (loss_idx - 1.5)
            scatter!(plt, xs, ys; color=color_map[loss_label], markersize=7, markerstrokewidth=0.0, label=subplot_id == 1 ? uppercase(loss_label) : "", xlabel="Restart", ylabel=ylabel_text, xticks=(1:length(restart_labels), restart_labels), xrotation=18, title=title_text, subplot=subplot_id)
        end
    end

    plot!(plt; plot_title="MAP versus MSE across restarts")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end
