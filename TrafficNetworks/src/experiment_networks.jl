"""
TOML-backed network specifications and builders used by the thesis experiments.

This layer turns a compact experiment description into concrete `Road`,
`Junction`, `Boundary`, and `RoadNetwork` objects, while preserving the metadata
needed to place sensors and interpret simulation results.
"""
module ExperimentNetworks

using TOML

using ..Boundaries: Boundary, boundary_flux!, make_inflow_signal, make_piecewise_signal
using ..Junctions: Junction
using ..RoadNetworks: RoadNetwork
using ..Roads: Road, make_road, update_road!
using ..Solvers
using ..Utils: minutes_to_hours, seconds_to_hours, make_piecewise_profile, make_profile_sum

export
    DEFAULT_EXPERIMENT_CFL,
    ExperimentRoadSpec,
    ExperimentJunctionSpec,
    ExperimentBoundarySpec,
    ExperimentNetworkSpec,
    load_experiment_network_spec,
    road_ids,
    road_labels,
    road_blocks,
    road_lengths,
    speed_limits,
    road_profiles,
    boundary_road_ids,
    boundary_inflows,
    observed_road_ids,
    observed_cell_ids,
    road_cell_count,
    road_length_km,
    cell_center_distance_km,
    sensor_cell_id,
    paired_sensor_cell_ids,
    build_experiment_network,
    build_block_experiment_network,
    simulation_step_count,
    make_boundaries,
    make_roads_from_blocks,
    make_roads_from_lengths

const DEFAULT_EXPERIMENT_CFL = 0.5

"""
    ExperimentRoadSpec

Road description loaded from an experiment network specification.
For block-discretized networks `blocks` is set; for fixed-cell networks
`length_km` is set.
"""
struct ExperimentRoadSpec
    id::Int
    label::String
    blocks::Union{Nothing, Int}
    length_km::Union{Nothing, Float64}
    speed_limit::Int
    profile::Function
end

struct ExperimentJunctionSpec
    incoming::Vector{Int}
    outgoing::Vector{Int}
end

struct ExperimentBoundarySpec
    road_id::Int
    inflow::Function
end

"""
    ExperimentNetworkSpec

Complete reusable description of an experiment network: road geometry,
junction topology, boundary inflows, control times, observation metadata, and
solver defaults. It is intentionally independent of a specific turning-rule
parameter value.
"""
struct ExperimentNetworkSpec
    name::String
    roads::Vector{ExperimentRoadSpec}
    junctions::Vector{ExperimentJunctionSpec}
    boundaries::Vector{ExperimentBoundarySpec}
    discretization_mode::Symbol
    basis_length_km::Union{Nothing, Float64}
    cells_per_block::Union{Nothing, Int}
    n_cells::Union{Nothing, Int}
    control_times::Vector{Float64}
    T::Float64
    CFL::Float64
    observed_road_ids::Vector{Int}
    observation::Dict{String, Any}
    metadata::Dict{String, Any}
end

function _require_key(table, key::String)
    @assert haskey(table, key) "Missing required key '$key'."
    return table[key]
end

function _as_int_vector(values)
    return Int[round(Int, value) for value in values]
end

function _as_float_vector(values)
    return Float64[Float64(value) for value in values]
end

function _as_float_rows(rows)
    return [Float64[Float64(value) for value in row] for row in rows]
end

function _time_hours(table, stem::String)
    hours_key = "$(stem)_hours"
    minutes_key = "$(stem)_minutes"
    seconds_key = "$(stem)_seconds"

    if haskey(table, hours_key)
        return Float64(table[hours_key])
    elseif haskey(table, minutes_key)
        return minutes_to_hours(table[minutes_key])
    elseif haskey(table, seconds_key)
        return seconds_to_hours(table[seconds_key])
    end

    error("Missing one of '$hours_key', '$minutes_key', or '$seconds_key'.")
end

function _regular_control_times(table)
    if haskey(table, "control_times_hours")
        return _as_float_vector(table["control_times_hours"])
    elseif haskey(table, "control_times_minutes")
        return minutes_to_hours.(_as_float_vector(table["control_times_minutes"]))
    elseif haskey(table, "control_times_seconds")
        return seconds_to_hours.(_as_float_vector(table["control_times_seconds"]))
    end

    step = _time_hours(table, "control_step")
    horizon = _time_hours(table, "horizon")
    return collect(step:step:horizon)
end

function _profile_scale(raw_scale, road_blocks)
    if raw_scale isa AbstractString
        @assert raw_scale in ("blocks", "road_blocks") "Unknown profile scale '$raw_scale'."
        @assert road_blocks !== nothing "Profile scale '$raw_scale' requires block discretization."
        return road_blocks
    end
    return Float64(raw_scale)
end

function _parse_profile(table, road_blocks)
    profile = _require_key(table, "profile")
    profile_type = String(get(profile, "type", "piecewise"))

    if profile_type == "piecewise"
        default_value = Float64(_require_key(profile, "default"))
        segments = _as_float_rows(_require_key(profile, "segments"))
        lower = Float64(get(profile, "lower", 0.0))
        upper = Float64(get(profile, "upper", 0.95))
        scale = _profile_scale(get(profile, "scale", 1.0), road_blocks)
        return make_piecewise_profile(default_value, segments...; lower=lower, upper=upper, scale=scale)
    elseif profile_type == "constant"
        value = Float64(_require_key(profile, "value"))
        return x -> value
    elseif profile_type in ("gaussian_sum", "profile_sum")
        base = Float64(haskey(profile, "base") ? profile["base"] : _require_key(profile, "default"))
        peaks = _as_float_rows(_require_key(profile, "peaks"))
        lower = Float64(get(profile, "lower", 0.0))
        upper = Float64(get(profile, "upper", 0.95))
        return make_profile_sum(base, peaks...; lower=lower, upper=upper)
    end

    error("Unknown profile type '$profile_type'.")
end

function _parse_signal(table)
    signal = _require_key(table, "signal")
    signal_type = String(get(signal, "type", "piecewise"))

    if signal_type == "piecewise"
        default_value = Float64(_require_key(signal, "default"))
        segments = if haskey(signal, "segments_hours")
            _as_float_rows(signal["segments_hours"])
        elseif haskey(signal, "segments_minutes")
            [
                [minutes_to_hours(row[1]), minutes_to_hours(row[2]), Float64(row[3])]
                for row in signal["segments_minutes"]
            ]
        elseif haskey(signal, "segments_seconds")
            [
                [seconds_to_hours(row[1]), seconds_to_hours(row[2]), Float64(row[3])]
                for row in signal["segments_seconds"]
            ]
        else
            _as_float_rows(_require_key(signal, "segments"))
        end

        lower = Float64(get(signal, "lower", 0.0))
        upper = Float64(get(signal, "upper", 0.24))
        return make_piecewise_signal(default_value, segments...; lower=lower, upper=upper)
    elseif signal_type == "gaussian"
        base = Float64(_require_key(signal, "base"))
        pulses = if haskey(signal, "pulses_hours")
            _as_float_rows(signal["pulses_hours"])
        elseif haskey(signal, "pulses_minutes")
            [
                [Float64(row[1]), minutes_to_hours(row[2]), minutes_to_hours(row[3])]
                for row in signal["pulses_minutes"]
            ]
        elseif haskey(signal, "pulses_seconds")
            [
                [Float64(row[1]), seconds_to_hours(row[2]), seconds_to_hours(row[3])]
                for row in signal["pulses_seconds"]
            ]
        else
            _as_float_rows(get(signal, "pulses", []))
        end

        lower = Float64(get(signal, "lower", 0.0))
        upper = Float64(get(signal, "upper", 0.24))
        return make_inflow_signal(base, pulses...; lower=lower, upper=upper)
    elseif signal_type == "constant"
        value = Float64(_require_key(signal, "value"))
        return t -> value
    end

    error("Unknown signal type '$signal_type'.")
end

function _parse_roads(tables)
    roads = ExperimentRoadSpec[]
    for table in tables
        id = Int(_require_key(table, "id"))
        label = String(get(table, "label", "Road $id"))
        blocks = haskey(table, "blocks") ? Int(table["blocks"]) : nothing
        length_km = haskey(table, "length_km") ? Float64(table["length_km"]) : nothing
        speed_limit = Int(_require_key(table, "speed_limit"))
        profile = _parse_profile(table, blocks)
        push!(roads, ExperimentRoadSpec(id, label, blocks, length_km, speed_limit, profile))
    end

    sort!(roads; by=road -> road.id)
    @assert road_ids(roads) == collect(1:length(roads)) "Road ids must be consecutive and 1-based for the current solver."
    return roads
end

function _parse_junctions(tables)
    return ExperimentJunctionSpec[
        ExperimentJunctionSpec(_as_int_vector(table["incoming"]), _as_int_vector(table["outgoing"]))
        for table in tables
    ]
end

function _parse_boundaries(tables)
    return ExperimentBoundarySpec[
        ExperimentBoundarySpec(Int(_require_key(table, "road_id")), _parse_signal(table))
        for table in tables
    ]
end

"""
    load_experiment_network_spec(path)

Load a TOML network specification. The resulting `ExperimentNetworkSpec` is the
first object in the thesis workflow: it defines topology, road profiles,
boundary inflows, discretization, control times, and observation metadata.
"""
function load_experiment_network_spec(path::AbstractString)
    data = TOML.parsefile(path)
    simulation = _require_key(data, "simulation")
    discretization = _require_key(data, "discretization")
    observation = Dict{String, Any}(get(data, "observations", Dict{String, Any}()))

    mode = Symbol(String(_require_key(discretization, "mode")))
    @assert mode in (:blocks, :lengths) "Discretization mode must be 'blocks' or 'lengths'."

    roads = _parse_roads(_require_key(data, "roads"))
    junctions = _parse_junctions(_require_key(data, "junctions"))
    boundaries = _parse_boundaries(_require_key(data, "boundaries"))
    control_times = _regular_control_times(simulation)
    observed = haskey(observation, "road_ids") ? _as_int_vector(observation["road_ids"]) : Int[]

    basis_length = mode == :blocks ? Float64(_require_key(discretization, "basis_length_km")) : nothing
    cells_per_block = mode == :blocks ? Int(_require_key(discretization, "cells_per_block")) : nothing
    n_cells = mode == :lengths ? Int(_require_key(discretization, "n_cells")) : nothing

    if mode == :blocks
        @assert all(road -> road.blocks !== nothing, roads) "Block discretization requires every road to define 'blocks'."
    else
        @assert all(road -> road.length_km !== nothing, roads) "Length discretization requires every road to define 'length_km'."
    end

    metadata = Dict{String, Any}(
        key => value for (key, value) in data
        if !(key in ("simulation", "discretization", "roads", "junctions", "boundaries", "observations"))
    )

    return ExperimentNetworkSpec(
        String(get(data, "name", splitext(basename(path))[1])),
        roads,
        junctions,
        boundaries,
        mode,
        basis_length,
        cells_per_block,
        n_cells,
        control_times,
        maximum(control_times),
        Float64(get(simulation, "cfl", DEFAULT_EXPERIMENT_CFL)),
        observed,
        observation,
        metadata,
    )
end

road_ids(roads::AbstractVector{ExperimentRoadSpec}) = Int[road.id for road in roads]
road_ids(spec::ExperimentNetworkSpec) = road_ids(spec.roads)
road_labels(spec::ExperimentNetworkSpec) = String[road.label for road in spec.roads]
road_blocks(spec::ExperimentNetworkSpec) = Int[road.blocks::Int for road in spec.roads]
road_lengths(spec::ExperimentNetworkSpec) = Float64[road.length_km::Float64 for road in spec.roads]
speed_limits(spec::ExperimentNetworkSpec) = Int[road.speed_limit for road in spec.roads]
road_profiles(spec::ExperimentNetworkSpec) = Function[road.profile for road in spec.roads]
boundary_road_ids(spec::ExperimentNetworkSpec) = Int[boundary.road_id for boundary in spec.boundaries]
boundary_inflows(spec::ExperimentNetworkSpec) = Function[boundary.inflow for boundary in spec.boundaries]
observed_road_ids(spec::ExperimentNetworkSpec) = copy(spec.observed_road_ids)

function road_cell_count(spec::ExperimentNetworkSpec, road_id::Int)
    road = spec.roads[road_id]
    if spec.discretization_mode == :blocks
        return spec.cells_per_block::Int * (road.blocks::Int)
    end
    return spec.n_cells::Int
end

function road_length_km(spec::ExperimentNetworkSpec, road_id::Int)
    road = spec.roads[road_id]
    if spec.discretization_mode == :blocks
        return (spec.basis_length_km::Float64) * (road.blocks::Int)
    end
    return road.length_km::Float64
end

cell_center_distance_km(spec::ExperimentNetworkSpec, road_id::Int, cell_id::Int) =
    road_length_km(spec, road_id) * (cell_id - 0.5) / road_cell_count(spec, road_id)

sensor_cell_id(n_cells::Integer; fraction=0.70) =
    clamp(round(Int, fraction * n_cells), 2, Int(n_cells) - 1)

sensor_cell_id(spec::ExperimentNetworkSpec, road_id::Int; fraction=0.70) =
    sensor_cell_id(road_cell_count(spec, road_id); fraction=fraction)

function _default_sensor_cells(n_cells::Integer, n_sensors::Integer)
    return unique(round.(Int, collect(range(2, Int(n_cells) - 1; length=Int(n_sensors)))))
end

"""
    observed_cell_ids(spec; mode=:shared, reference_road_id=nothing)

Return the sensor cells described by `spec.observation`. Shared sensors use one
cell list for every observed road; paired sensors choose one cell per observed
road from paired road/fraction metadata.
"""
function observed_cell_ids(spec::ExperimentNetworkSpec; mode::Symbol=:shared, reference_road_id=nothing)
    obs = spec.observation
    if haskey(obs, "cell_ids")
        return _as_int_vector(obs["cell_ids"])
    elseif mode == :paired && haskey(obs, "paired_cell_fractions")
        return paired_sensor_cell_ids(spec, observed_road_ids(spec), _as_float_vector(obs["paired_cell_fractions"]))
    elseif haskey(obs, "cell_fractions")
        road_id = reference_road_id === nothing ? first(observed_road_ids(spec)) : Int(reference_road_id)
        return Int[sensor_cell_id(spec, road_id; fraction=fraction) for fraction in _as_float_vector(obs["cell_fractions"])]
    elseif haskey(obs, "n_sensors")
        road_id = reference_road_id === nothing ? first(observed_road_ids(spec)) : Int(reference_road_id)
        return _default_sensor_cells(road_cell_count(spec, road_id), Int(obs["n_sensors"]))
    end

    return Int[]
end

function paired_sensor_cell_ids(
    spec::ExperimentNetworkSpec,
    road_ids::AbstractVector{<:Integer},
    fractions::AbstractVector{<:Real},
)
    @assert length(road_ids) == length(fractions) "Paired road ids and sensor fractions must have equal length."
    return Int[
        sensor_cell_id(spec, Int(road_id); fraction=Float64(fraction))
        for (road_id, fraction) in zip(road_ids, fractions)
    ]
end

"""
    make_boundaries(road_ids, inflows)

Create boundary objects from matched boundary road ids and inflow functions.
"""
function make_boundaries(road_ids::AbstractVector{<:Integer}, inflows::AbstractVector)
    @assert length(road_ids) == length(inflows) "Boundary road ids and inflows must have equal length."
    return Boundary[Boundary(Int(road_id), inflow) for (road_id, inflow) in zip(road_ids, inflows)]
end

function make_roads_from_blocks(
    road_blocks::AbstractVector{<:Integer},
    basis_length_km::Real,
    cells_per_block::Integer,
    road_profiles::AbstractVector,
    speed_limits::AbstractVector;
    road_ids=collect(1:length(road_profiles)),
    rho_max=1,
    state_eltype::Type{<:Real}=Float64,
)
    @assert length(road_ids) == length(road_blocks) == length(road_profiles) == length(speed_limits)

    return Road[
        make_road(
            Int(road_id),
            Int(road_blocks[idx]),
            basis_length_km,
            Int(cells_per_block),
            road_profiles[idx],
            speed_limits[idx],
            rho_max;
            state_eltype=state_eltype,
        )
        for (idx, road_id) in enumerate(road_ids)
    ]
end

function make_roads_from_lengths(
    road_lengths::AbstractVector,
    n_cells::Integer,
    road_profiles::AbstractVector,
    speed_limits::AbstractVector;
    road_ids=collect(1:length(road_profiles)),
    rho_max=1,
    state_eltype::Type{<:Real}=Float64,
)
    @assert length(road_ids) == length(road_lengths) == length(road_profiles) == length(speed_limits)

    return Road[
        make_road(
            Int(road_id),
            1,
            road_lengths[idx],
            Int(n_cells),
            road_profiles[idx],
            speed_limits[idx],
            rho_max;
            state_eltype=state_eltype,
        )
        for (idx, road_id) in enumerate(road_ids)
    ]
end

"""
    build_experiment_network(spec, junction_rules; kwargs...)

Build a `RoadNetwork` from an `ExperimentNetworkSpec` and one junction rule per
junction. Optional keyword overrides make it possible to reuse the same topology
with altered road lengths, profiles, speed limits, boundary inflows, or numeric
state type.
"""
function build_experiment_network(
    spec::ExperimentNetworkSpec,
    junction_rules::AbstractVector;
    T=spec.T,
    CFL=spec.CFL,
    road_block_values=nothing,
    road_length_values=nothing,
    road_profile_values=nothing,
    speed_limit_values=nothing,
    boundary_inflow_values=nothing,
    state_eltype::Type{<:Real}=Float64,
)
    profiles = road_profile_values === nothing ? road_profiles(spec) : road_profile_values
    limits = speed_limit_values === nothing ? speed_limits(spec) : speed_limit_values

    roads = if spec.discretization_mode == :blocks
        blocks = road_block_values === nothing ? road_blocks(spec) : road_block_values
        make_roads_from_blocks(
            blocks,
            spec.basis_length_km::Float64,
            spec.cells_per_block::Int,
            profiles,
            limits;
            road_ids=road_ids(spec),
            state_eltype=state_eltype,
        )
    else
        lengths = road_length_values === nothing ? road_lengths(spec) : road_length_values
        make_roads_from_lengths(
            lengths,
            spec.n_cells::Int,
            profiles,
            limits;
            road_ids=road_ids(spec),
            state_eltype=state_eltype,
        )
    end

    junctions = _junctions_from_rules(spec, junction_rules)
    inflows = boundary_inflow_values === nothing ? boundary_inflows(spec) : boundary_inflow_values
    boundaries = make_boundaries(boundary_road_ids(spec), inflows)

    return RoadNetwork(roads, junctions, boundaries, T, CFL)
end

function _junctions_from_rules(spec::ExperimentNetworkSpec, junction_rules::AbstractVector)
    @assert length(junction_rules) == length(spec.junctions) "Expected one junction rule per junction in the network spec."
    return Junction[
        Junction(junction.incoming, junction.outgoing, junction_rules[idx])
        for (idx, junction) in enumerate(spec.junctions)
    ]
end

"""
    build_block_experiment_network(spec, junction_rules, road_blocks, basis_length_km,
                                   cells_per_block, profiles, speed_limits; kwargs...)

Lower-level builder for block-discretized experiments where the caller supplies
the discretization and road data explicitly. This is useful for phase-one
experiments that vary the common base length or cell resolution.
"""
function build_block_experiment_network(
    spec::ExperimentNetworkSpec,
    junction_rules::AbstractVector,
    road_block_values::AbstractVector{<:Integer},
    basis_length_km::Real,
    cells_per_block::Integer,
    road_profile_values::AbstractVector,
    speed_limit_values::AbstractVector;
    T=spec.T,
    CFL=spec.CFL,
    boundary_inflow_values=boundary_inflows(spec),
    state_eltype::Type{<:Real}=Float64,
)
    roads = make_roads_from_blocks(
        road_block_values,
        basis_length_km,
        cells_per_block,
        road_profile_values,
        speed_limit_values;
        road_ids=road_ids(spec),
        state_eltype=state_eltype,
    )
    junctions = _junctions_from_rules(spec, junction_rules)
    boundaries = make_boundaries(boundary_road_ids(spec), boundary_inflow_values)
    return RoadNetwork(roads, junctions, boundaries, T, CFL)
end

"""
    simulation_step_count(net; times=nothing)

Count the CFL time steps a simulation would take without storing the full
history. Used for runtime diagnostics and budget comparisons.
"""
function simulation_step_count(net::RoadNetwork; times=nothing)
    roads = net.roads
    junctions = net.junctions
    boundaries = net.boundaries
    nroads = length(roads)
    state_T = eltype(roads[1].rho)
    has_downstream_junction = Solvers.downstream_junction_mask(junctions, nroads)
    control_times = Solvers.normalize_control_times(times, net.T)
    next_time_idx = 1
    t = 0.0
    steps = 0

    while t < net.T - Solvers.TIME_TOL
        left_flux, right_flux = Solvers.endpoint_fluxes_for_cfl(
            roads,
            junctions,
            boundaries,
            has_downstream_junction,
            t,
            state_T,
        )

        dt = Solvers.compute_dt(t, net, roads, control_times, next_time_idx, left_flux, right_flux)

        for b in boundaries
            left_flux[b.road_id] = boundary_flux!(b, roads[b.road_id], t, dt)
        end

        for i in 1:nroads
            update_road!(roads[i], dt, left_flux[i], right_flux[i])
        end

        t += dt
        steps += 1

        if next_time_idx <= length(control_times) &&
           isapprox(t, control_times[next_time_idx]; atol=Solvers.TIME_TOL, rtol=0.0)
            next_time_idx += 1
        end
    end

    return steps
end

end # module ExperimentNetworks
