#!/usr/bin/env julia

using Printf
using Statistics
using TOML
using TrafficNetworks

const HERE = @__DIR__
const NETWORK_TOML = get(ENV, "E18_SMOKE_NETWORK_TOML", joinpath(HERE, "e18_large_sim_network.toml"))
const OUTPUT_DIR = get(ENV, "E18_SMOKE_OUTPUT_DIR", joinpath(HERE, "smoke_outputs"))
const OUTPUT_BASENAME = get(ENV, "E18_SMOKE_OUTPUT_BASENAME", "e18_large_smoke")
const ROAD_HISTORY_CSV = joinpath(OUTPUT_DIR, "$(OUTPUT_BASENAME)_road_history.csv")
const CELL_SNAPSHOTS_CSV = joinpath(OUTPUT_DIR, "$(OUTPUT_BASENAME)_cell_snapshots.csv")
const SUMMARY_CSV = joinpath(OUTPUT_DIR, "$(OUTPUT_BASENAME)_summary.csv")

const HORIZON_SECONDS = 180.0
const SNAPSHOT_SECONDS = [0.0, 45.0, 90.0, 135.0, 180.0]

seconds_to_hours_local(x) = Float64(x) / 3600.0

function matrix_float(rows)
    nrow = length(rows)
    ncol = nrow == 0 ? 0 : length(rows[1])
    M = Matrix{Float64}(undef, nrow, ncol)
    for i in 1:nrow, j in 1:ncol
        M[i, j] = Float64(rows[i][j])
    end
    return M
end

function build_rule(junction_table)
    P = matrix_float(junction_table["turning_matrix"])
    return TurningFractionRule(P)
end

function road_context(row)
    road_id = Int(row["id"])
    road_type = String(row["road_type"])
    lanes = Float64(row["lanes"])
    speed_limit = Int(row["speed_limit"])
    length_m = Float64(row["original_length_m"])
    source_road_ids = String(row["source_road_ids"])
    mainish = speed_limit >= 60 || lanes > 1.0 || road_type == "Kanalisert veg"
    localish = speed_limit <= 40 && lanes <= 1.0 && road_type == "Enkel bilveg"
    rampish = road_type == "Rampe"
    artificial = occursin("Artificial", road_type) || occursin("Terminal", road_type)
    pseudo = mod(37 * road_id + 11, 101) / 100.0
    side_hotspot = localish && !artificial && (pseudo >= 0.82 || mod(road_id, 31) in (0, 7))
    side_pulse = localish && !artificial && !side_hotspot && (pseudo >= 0.58 || mod(road_id, 23) == 0)
    return (
        road_id=road_id,
        road_type=road_type,
        lanes=lanes,
        speed_limit=speed_limit,
        length_m=length_m,
        source_road_ids=source_road_ids,
        mainish=mainish,
        localish=localish,
        rampish=rampish,
        artificial=artificial,
        pseudo=pseudo,
        side_hotspot=side_hotspot,
        side_pulse=side_pulse,
    )
end

function varied_initial_condition(row, context)
    road_id = context.road_id
    blocks = max(Int(row["blocks"]), 1)

    base = 0.06 + 0.10 * context.pseudo
    if context.mainish
        base = 0.34 + 0.32 * context.pseudo
    end
    if context.lanes >= 2.0
        base += 0.08
    end
    if context.rampish
        base = 0.18 + 0.22 * context.pseudo
    end
    if context.localish
        base = 0.035 + 0.13 * context.pseudo
    end
    if context.length_m <= 36.0 && !context.artificial
        base = max(base, 0.22 + 0.42 * context.pseudo)
    end
    if context.side_pulse
        base = max(base, 0.12 + 0.24 * context.pseudo)
    end
    if context.side_hotspot
        base = max(base, 0.48 + 0.34 * context.pseudo)
    end
    if context.artificial
        base = 0.02 + 0.03 * context.pseudo
    end

    amplitude = context.artificial ? 0.005 : 0.025 + 0.030 * mod(road_id, 5) / 4.0
    if context.side_pulse
        amplitude = max(amplitude, 0.055)
    end
    if context.side_hotspot
        amplitude = max(amplitude, 0.080)
    end
    phase = 0.47 * road_id
    return x -> clamp(base + amplitude * sin(2.0 * pi * Float64(x) / blocks + phase), 0.01, 0.92)
end

function varied_boundary_signal(boundary_row, road_context)
    boundary_id = Int(boundary_row["id"])
    pseudo = mod(19 * boundary_id + 7, 97) / 96.0

    if road_context.artificial
        base = 0.002 + 0.004 * pseudo
        pulse_one = 0.006 + 0.008 * pseudo
        pulse_two = 0.008 + 0.010 * pseudo
    elseif road_context.mainish
        base = 0.030 + 0.030 * pseudo
        pulse_one = 0.130 + 0.070 * pseudo
        pulse_two = 0.150 + 0.060 * pseudo
    elseif road_context.rampish
        base = 0.018 + 0.022 * pseudo
        pulse_one = 0.080 + 0.050 * pseudo
        pulse_two = 0.090 + 0.050 * pseudo
    elseif road_context.side_hotspot
        base = 0.020 + 0.030 * pseudo
        pulse_one = 0.165 + 0.060 * pseudo
        pulse_two = 0.185 + 0.050 * pseudo
    elseif road_context.side_pulse
        base = 0.012 + 0.020 * pseudo
        pulse_one = 0.080 + 0.060 * pseudo
        pulse_two = 0.100 + 0.055 * pseudo
    else
        base = 0.006 + 0.014 * pseudo
        pulse_one = 0.025 + 0.035 * pseudo
        pulse_two = 0.030 + 0.040 * pseudo
    end

    start_one = 20.0 + 4.0 * mod(boundary_id, 7)
    stop_one = 85.0 + 5.0 * mod(boundary_id, 5)
    start_two = 100.0 + 3.0 * mod(boundary_id, 9)
    stop_two = 170.0

    return t -> begin
        seconds = 3600.0 * t
        value = if start_one <= seconds < stop_one
            pulse_one
        elseif start_two <= seconds < stop_two
            pulse_two
        else
            base
        end
        clamp(value, 0.0, 0.235)
    end
end

function build_smoke_network(path)
    data = TOML.parsefile(path)
    roads_table = sort(data["roads"]; by=row -> Int(row["id"]))
    junctions_table = sort(data["junctions"]; by=row -> Int(row["id"]))
    boundaries_table = sort(data["boundaries"]; by=row -> Int(row["id"]))
    contexts = Dict(Int(row["id"]) => road_context(row) for row in roads_table)

    basis_length_km = Float64(data["discretization"]["basis_length_m"]) / 1000.0
    cells_per_block = Int(data["discretization"]["cells_per_block"])

    roads = Road[
        make_road(
            Int(row["id"]),
            Int(row["blocks"]),
            basis_length_km,
            cells_per_block,
            varied_initial_condition(row, contexts[Int(row["id"])]),
            Int(row["speed_limit"]),
            Float64(row["lanes"]),
        )
        for row in roads_table
    ]

    junctions = Junction[
        Junction(
            Int.(row["incoming"]),
            Int.(row["outgoing"]),
            build_rule(row),
        )
        for row in junctions_table
    ]

    boundaries = Boundary[
        Boundary(Int(row["road_id"]), varied_boundary_signal(row, contexts[Int(row["road_id"])]))
        for row in boundaries_table
    ]

    T = seconds_to_hours_local(HORIZON_SECONDS)
    CFL = Float64(data["simulation"]["cfl"])
    return RoadNetwork(roads, junctions, boundaries, T, CFL), data, roads_table, junctions_table
end

function validate_topology(net::RoadNetwork)
    nroads = length(net.roads)
    has_upstream = falses(nroads)
    has_downstream = falses(nroads)

    for boundary in net.boundaries
        has_upstream[boundary.road_id] = true
    end

    for junction in net.junctions
        for road_id in junction.outgoing
            has_upstream[road_id] = true
        end
        for road_id in junction.incoming
            has_downstream[road_id] = true
        end
    end

    return findall(!, has_upstream), findall(!, has_downstream)
end

function write_road_history(path, hist)
    open(path, "w") do io
        println(io, "time_index,time_hours,time_seconds,road_id,density_mean,density_min,density_max")
        for (time_index, time) in enumerate(hist.times)
            time_seconds = 3600.0 * Float64(time)
            for (road_id, road_history) in enumerate(hist.road_histories)
                values = view(road_history, :, time_index)
                println(
                    io,
                    join(
                        (
                            time_index,
                            Float64(time),
                            time_seconds,
                            road_id,
                            Float64(mean(values)),
                            Float64(minimum(values)),
                            Float64(maximum(values)),
                        ),
                        ",",
                    ),
                )
            end
        end
    end
end

function write_cell_snapshots(path, hist)
    open(path, "w") do io
        println(io, "time_index,time_hours,time_seconds,road_id,cell_id,density")
        for (time_index, time) in enumerate(hist.times)
            time_seconds = 3600.0 * Float64(time)
            for (road_id, road_history) in enumerate(hist.road_histories)
                values = view(road_history, :, time_index)
                for cell_id in eachindex(values)
                    println(
                        io,
                        join(
                            (
                                time_index,
                                Float64(time),
                                time_seconds,
                                road_id,
                                cell_id,
                                Float64(values[cell_id]),
                            ),
                            ",",
                        ),
                    )
                end
            end
        end
    end
end

function count_by(rows, key)
    counts = Dict{String, Int}()
    for row in rows
        value = string(row[key])
        counts[value] = get(counts, value, 0) + 1
    end
    return join(["$key=$value:$count" for (value, count) in sort(collect(counts))], "; ")
end

function write_summary(path, net, roads_table, junctions_table, elapsed_seconds, hist, missing_upstream, missing_downstream)
    all_values = reduce(vcat, vec.(hist.road_histories))
    total_cells = sum(length(road.rho) for road in net.roads)
    total_rounded_length_km = sum(Float64(row["length_m"]) for row in roads_table) / 1000.0
    total_original_length_km = sum(Float64(row["original_length_m"]) for row in roads_table) / 1000.0
    multilane_roads = sum(Float64(row["lanes"]) > 1.0 for row in roads_table)
    roundabout_nodes = sum(String(row["node_type"]) == "roundabout" for row in junctions_table)
    road_type_counts = count_by(roads_table, "road_type")
    lane_counts = count_by(roads_table, "lanes")
    speed_limit_counts = count_by(roads_table, "speed_limit")

    open(path, "w") do io
        println(io, "quantity,value")
        println(io, "horizon_seconds,$HORIZON_SECONDS")
        println(io, "saved_times,$(join(SNAPSHOT_SECONDS, ' '))")
        println(io, "runtime_seconds,$elapsed_seconds")
        println(io, "roads,$(length(net.roads))")
        println(io, "total_cells,$total_cells")
        println(io, "total_rounded_length_km,$total_rounded_length_km")
        println(io, "total_original_length_km,$total_original_length_km")
        println(io, "junctions,$(length(net.junctions))")
        println(io, "roundabout_nodes,$roundabout_nodes")
        println(io, "boundaries,$(length(net.boundaries))")
        println(io, "multilane_roads,$multilane_roads")
        println(io, "road_types,\"$road_type_counts\"")
        println(io, "lane_counts,\"$lane_counts\"")
        println(io, "speed_limits,\"$speed_limit_counts\"")
        println(io, "density_min,$(minimum(all_values))")
        println(io, "density_mean,$(mean(all_values))")
        println(io, "density_max,$(maximum(all_values))")
        println(io, "missing_upstream_count,$(length(missing_upstream))")
        println(io, "open_downstream_count,$(length(missing_downstream))")
    end
end

function main()
    mkpath(OUTPUT_DIR)
    net, _, roads_table, junctions_table = build_smoke_network(NETWORK_TOML)
    missing_upstream, missing_downstream = validate_topology(net)
    @assert isempty(missing_upstream) "Roads missing upstream boundary/junction: $(missing_upstream)"

    times = seconds_to_hours_local.(SNAPSHOT_SECONDS)
    elapsed_seconds = @elapsed history = simulate!(net; times=times)

    write_road_history(ROAD_HISTORY_CSV, history)
    write_cell_snapshots(CELL_SNAPSHOTS_CSV, history)
    write_summary(SUMMARY_CSV, net, roads_table, junctions_table, elapsed_seconds, history, missing_upstream, missing_downstream)

    all_values = reduce(vcat, vec.(history.road_histories))
    @printf("E18 large smoke simulation\n")
    @printf("runtime: %.3f s\n", elapsed_seconds)
    @printf("roads: %d\n", length(net.roads))
    @printf("junctions: %d\n", length(net.junctions))
    @printf("boundaries: %d\n", length(net.boundaries))
    @printf("cells: %d\n", sum(length(road.rho) for road in net.roads))
    @printf("saved times: %d\n", length(history.times))
    @printf("density min/mean/max: %.6f / %.6f / %.6f\n", minimum(all_values), mean(all_values), maximum(all_values))
    @printf("open downstream roads: %d\n", length(missing_downstream))
    @printf("wrote: %s\n", OUTPUT_DIR)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
