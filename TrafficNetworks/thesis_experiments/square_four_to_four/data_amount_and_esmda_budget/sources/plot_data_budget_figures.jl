import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using Printf

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT"] = "1"
include(joinpath(@__DIR__, "generate_data_budget_results.jl"))

const DATA_BUDGET_EXPERIMENT_REPLOT_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs")
const DATA_BUDGET_EXPERIMENT_REPLOT_FIGURE_DIR = joinpath(@__DIR__, "..", "figures")

function parse_tsv_value(raw::AbstractString)
    value = strip(raw)
    isempty(value) && return ""

    if lowercase(value) == "true"
        return true
    elseif lowercase(value) == "false"
        return false
    end

    if !occursin(',', value)
        int_value = tryparse(Int, value)
        int_value !== nothing && return int_value

        float_value = tryparse(Float64, value)
        float_value !== nothing && return float_value
    end

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

function run_square_data_budget_experiment_replot(;
    output_dir=DATA_BUDGET_EXPERIMENT_REPLOT_OUTPUT_DIR,
    figure_dir=DATA_BUDGET_EXPERIMENT_REPLOT_FIGURE_DIR,
)
    summary_path = joinpath(output_dir, "data_budget_summary.tsv")
    summary_rows = read_namedtuple_tsv(summary_path)
    isempty(summary_rows) && error("No rows found in $(summary_path).")

    data_plot_path = joinpath(figure_dir, "data_budget_data_amount_summary.png")
    compute_plot_path = joinpath(figure_dir, "data_budget_compute_cost_summary.png")

    plot_final_data_amount_summary(summary_rows; output_path=data_plot_path)
    plot_final_compute_cost_summary(summary_rows; output_path=compute_plot_path)

    println("Replotted data/budget figures")
    println("-----------------------------")
    println(summary_path)
    println(data_plot_path)
    println(compute_plot_path)

    return (
        summary_path=summary_path,
        data_plot_path=data_plot_path,
        compute_plot_path=compute_plot_path,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT_REPLOT", "0") != "1"
    run_square_data_budget_experiment_replot()
end
