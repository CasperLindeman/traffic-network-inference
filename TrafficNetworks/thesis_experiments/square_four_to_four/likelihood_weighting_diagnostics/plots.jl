import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

const _LIKELIHOOD_PLOT_ENV_KEYS = [
    "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_MULTI_SCENARIO_LIKELIHOOD_FLOOR_COMPARISON",
    "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_DATA_BUDGET_EXPERIMENT",
    "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_STRUCTURE",
]
const _LIKELIHOOD_PLOT_PREVIOUS_ENV = Dict(key => get(ENV, key, nothing) for key in _LIKELIHOOD_PLOT_ENV_KEYS)

for key in _LIKELIHOOD_PLOT_ENV_KEYS
    ENV[key] = "1"
end

include(joinpath(@__DIR__, "sources", "likelihood_floor_comparison.jl"))

for key in _LIKELIHOOD_PLOT_ENV_KEYS
    previous = _LIKELIHOOD_PLOT_PREVIOUS_ENV[key]
    if previous === nothing
        delete!(ENV, key)
    else
        ENV[key] = previous
    end
end

const LIKELIHOOD_DIAGNOSTIC_OUTPUT_DIR = joinpath(@__DIR__, "outputs")
const LIKELIHOOD_DIAGNOSTIC_FIGURE_DIR = joinpath(@__DIR__, "figures")

function parse_likelihood_tsv_value(raw::AbstractString)
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

function read_likelihood_namedtuple_tsv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]

    header = Symbol.(split(first(lines), '\t'))
    rows = NamedTuple[]

    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        values = parse_likelihood_tsv_value.(split(line, '\t'; keepempty=true))
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end

    return rows
end

function make_square_likelihood_weighting_diagnostic_figures()
    row_swap_rows = read_likelihood_namedtuple_tsv(
        joinpath(LIKELIHOOD_DIAGNOSTIC_OUTPUT_DIR, "row_replacement", "turning_row_swap_fit_metrics.tsv"),
    )
    floor_summary_rows = read_likelihood_namedtuple_tsv(
        joinpath(LIKELIHOOD_DIAGNOSTIC_OUTPUT_DIR, "floor_comparison", "likelihood_floor_summary.tsv"),
    )

    row_swap_path = joinpath(LIKELIHOOD_DIAGNOSTIC_FIGURE_DIR, "square_turning_row_swap_fit_improvement.png")
    floor_path = joinpath(LIKELIHOOD_DIAGNOSTIC_FIGURE_DIR, "square_likelihood_floor_comparison.png")

    plot_row_swap_fit_improvement(row_swap_rows; output_path=row_swap_path)
    plot_likelihood_floor_summary(floor_summary_rows; output_path=floor_path)

    println("Replotted likelihood diagnostic figures")
    println("---------------------------------------")
    println(row_swap_path)
    println(floor_path)

    return (
        row_swap_path=row_swap_path,
        floor_path=floor_path,
    )
end
