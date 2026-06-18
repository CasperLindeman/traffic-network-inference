# Shared helpers for square-four-to-four multi-scenario experiments.

function plot_runtime_curves(rows; output_path=nothing)
    regimes = unique(getproperty.(rows, :regime_label))
    specs = [
        (:turning_rmse, "Turning-fraction RMSE", "RMSE"),
        (:final_state_rmse_all, "Final-state RMSE", "RMSE"),
    ]
    color_map = Dict("adam" => :firebrick, "esmda" => :steelblue)
    label_map = Dict("adam" => "ADAM", "esmda" => "ESMDA")

    plt = plot(layout=(length(regimes), length(specs)), size=(1320, 360 * length(regimes)), dpi=180, legend=:topright, left_margin=8Plots.mm, bottom_margin=8Plots.mm)

    for (row_id, regime_label) in enumerate(regimes)
        regime_rows = filter(row -> row.regime_label == regime_label, rows)
        obs_count = first(regime_rows).observation_count
        obs_multiplier = first(regime_rows).observation_multiplier

        for (col_id, (key, title_text, ylabel_text)) in enumerate(specs)
            subplot_id = (row_id - 1) * length(specs) + col_id

            for method in ("adam", "esmda")
                method_rows = sort(filter(row -> row.method == method, regime_rows); by=row -> row.solve_seconds)
                plot!(
                    plt,
                    getproperty.(method_rows, :solve_seconds),
                    getproperty.(method_rows, key);
                    color=color_map[method],
                    linewidth=2.4,
                    marker=:circle,
                    markersize=5,
                    xlabel="Runtime (s)",
                    ylabel=ylabel_text,
                    title="$(regime_label): $(title_text)\nobs $(obs_count) ($(round(obs_multiplier; digits=1))x)",
                    label=row_id == 1 && col_id == 1 ? label_map[method] : "",
                    subplot=subplot_id,
                )
            end
        end
    end

    plot!(plt; plot_title="Multi-scenario runtime convergence by data amount")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function plot_data_amount_summary(rows, budget_rows; output_path=nothing)
    regimes = getproperty.(budget_rows, :regime_label)
    x = collect(1:length(regimes))
    xtick_labels = ["$(row.regime_label)\n$(Int(round(row.observation_multiplier)))x" for row in budget_rows]

    plt = plot(layout=(2, 2), size=(1320, 980), dpi=180, legend=:topright, left_margin=8Plots.mm, bottom_margin=8Plots.mm)

    plot!(plt, x, getproperty.(budget_rows, :adam_recommended_seconds); color=:firebrick, linewidth=2.4, marker=:circle, label="ADAM budget", subplot=1)
    plot!(plt, x, getproperty.(budget_rows, :esmda_recommended_seconds); color=:steelblue, linewidth=2.4, marker=:circle, label="ESMDA budget", subplot=1)
    plot!(plt; xticks=(x, xtick_labels), ylabel="Seconds", title="Recommended method budgets", subplot=1)

    plot!(plt, x, getproperty.(budget_rows, :joint_recommended_seconds); color=:black, linewidth=2.4, marker=:diamond, label="Joint budget", subplot=2)
    plot!(plt; xticks=(x, xtick_labels), ylabel="Seconds", title="Recommended joint budget", subplot=2)

    plot!(plt, x, getproperty.(budget_rows, :best_turning_rmse); color=:darkorange3, linewidth=2.4, marker=:circle, label="Best turning RMSE", subplot=3)
    plot!(plt; xticks=(x, xtick_labels), ylabel="RMSE", title="Best turning RMSE vs data amount", subplot=3)

    plot!(plt, x, getproperty.(budget_rows, :best_final_state_rmse_all); color=:seagreen4, linewidth=2.4, marker=:circle, label="Best final-state RMSE", subplot=4)
    plot!(plt; xticks=(x, xtick_labels), ylabel="RMSE", title="Best final-state RMSE vs data amount", subplot=4)

    plot!(plt; plot_title="Multi-scenario data-amount summary")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end
