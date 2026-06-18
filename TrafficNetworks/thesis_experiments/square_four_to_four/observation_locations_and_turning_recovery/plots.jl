import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "sources", "observation_class_recovery.jl"))

const OBSERVATION_LOCATION_OUTPUT_DIR = joinpath(@__DIR__, "outputs")
const OBSERVATION_LOCATION_FIGURE_DIR = joinpath(@__DIR__, "figures")

function parse_observation_location_tsv_value(raw::AbstractString)
    value = strip(raw)
    isempty(value) && return ""

    int_value = tryparse(Int, value)
    int_value !== nothing && return int_value

    float_value = tryparse(Float64, value)
    float_value !== nothing && return float_value

    return value
end

function read_observation_location_namedtuple_tsv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]

    header = Symbol.(split(first(lines), '\t'))
    rows = NamedTuple[]

    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        values = parse_observation_location_tsv_value.(split(line, '\t'; keepempty=true))
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end

    return rows
end

function make_square_observation_locations_turning_recovery_figure()
    class_rows = read_observation_location_namedtuple_tsv(
        joinpath(OBSERVATION_LOCATION_OUTPUT_DIR, "exp_a_class_rmse_summary.tsv"),
    )
    seed_rows = read_observation_location_namedtuple_tsv(
        joinpath(OBSERVATION_LOCATION_OUTPUT_DIR, "exp_a_seed_class_rmse.tsv"),
    )

    figure_path = write_exp_a_rmse_plot(
        class_rows,
        seed_rows,
        joinpath(OBSERVATION_LOCATION_FIGURE_DIR, "square_observation_locations_turning_error.png"),
    )

    println("Replotted observation-location figure")
    println("-------------------------------------")
    println(figure_path)

    return (
        figure_path=figure_path,
    )
end
