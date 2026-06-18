import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", ".."))

using LaTeXStrings
using Plots

const RUNTIME_CALIBRATION_OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "outputs"))
const RUNTIME_CALIBRATION_FIGURE_DIR = normpath(joinpath(@__DIR__, "..", "figures"))

function parse_tsv_value(raw::AbstractString)
    value = strip(raw)
    int_value = tryparse(Int, value)
    int_value !== nothing && return int_value
    float_value = tryparse(Float64, value)
    float_value !== nothing && return float_value
    return value
end

function read_namedtuple_tsv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]

    header = Symbol.(split(first(lines), '\t'))
    rows = NamedTuple[]

    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        values = parse_tsv_value.(split(line, '\t'; keepempty=true))
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end

    return rows
end

function runtime_budget_seconds(label::AbstractString)
    match_obj = match(r"(\d+)s", label)
    match_obj === nothing && error("Could not parse runtime budget from label '$label'.")
    return parse(Int, match_obj.captures[1])
end

function plot_runtime_objective(rows; output_path=nothing)
    selected = [
        row for row in rows
        if row.label in ("ADAM 30s", "ADAM 120s", "ADAM 300s", "L-BFGS 30s", "L-BFGS 120s", "L-BFGS 300s")
    ]
    isempty(selected) && error("No ADAM/L-BFGS runtime rows found.")

    methods = ["ADAM", "L-BFGS"]
    colors = Dict("ADAM" => :darkorange2, "L-BFGS" => :steelblue4)
    markers = Dict("ADAM" => :circle, "L-BFGS" => :diamond)

    plt = plot(
        size=(900, 520),
        dpi=180,
        legend=:topright,
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
        xlabel="Time budget (s)",
        ylabel=L"\ell_{\mathrm{MAP}}",
        xlims=(20, 310),
        xticks=([30, 120, 300], ["30", "120", "300"]),
        yaxis=:log10,
        ylims=(1e2, 1e4),
        yticks=([1e2, 3e2, 1e3, 3e3, 1e4], ["1e2", "3e2", "1e3", "3e3", "1e4"]),
        minorgrid=true,
    )

    for method in methods
        method_rows = sort(
            [row for row in selected if startswith(row.label, method)];
            by=row -> runtime_budget_seconds(row.label),
        )
        plot!(
            plt,
            runtime_budget_seconds.(getproperty.(method_rows, :label)),
            getproperty.(method_rows, :best_loss);
            color=colors[method],
            linewidth=2.8,
            markershape=markers[method],
            markersize=7,
            markerstrokecolor=:black,
            markerstrokewidth=1.1,
            label=method,
        )
    end

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function run_runtime_objective_plot(;
    metrics_path=joinpath(RUNTIME_CALIBRATION_OUTPUT_DIR, "optimizer_runtime_metrics.tsv"),
    figure_dir=RUNTIME_CALIBRATION_FIGURE_DIR,
)
    rows = read_namedtuple_tsv(metrics_path)
    figure_path = joinpath(figure_dir, "square_map_runtime_objective.png")
    plot_runtime_objective(rows; output_path=figure_path)

    println("Plotted MAP runtime objective")
    println("----------------------------")
    println(metrics_path)
    println(figure_path)

    return (metrics_path=metrics_path, figure_path=figure_path)
end

run_runtime_objective_plot()
