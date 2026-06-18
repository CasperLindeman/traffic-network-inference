# Shared helpers for square-four-to-four single-scenario experiments.

all_square_road_ids() = copy(ALL_SQUARE_ROAD_IDS)
unobserved_square_road_ids(setup::SquareSingleScenarioSetup) = setdiff(all_square_road_ids(), setup.observed_road_ids)

function copy_sensor_spec(sensor_spec)
    if sensor_spec isa AbstractDict
        return Dict(Int(road_id) => copy(collect(values)) for (road_id, values) in sensor_spec)
    end

    return Float64.(collect(sensor_spec))
end

function road_role_symbol(road_id::Int)
    if road_id in EXTERNAL_INCOMING_ROADS
        return :incoming
    elseif road_id in EXTERNAL_OUTGOING_ROADS
        return :outgoing
    end
    return :internal
end

function single_scenario_road_length_multipliers(base_length_km::Real)
    multipliers = round.(Int, SINGLE_SCENARIO_LEGACY_TEMPLATE.road_lengths ./ base_length_km)
    reconstructed = Float64(base_length_km) .* multipliers
    @assert maximum(abs.(reconstructed .- SINGLE_SCENARIO_LEGACY_TEMPLATE.road_lengths)) <= 1e-12 "Road lengths are not integer multiples of the shared base length."
    return Int.(multipliers)
end

function square_single_scenario_setup(;
    peak_noise_sigma=DEFAULT_PEAK_NOISE_SIGMA,
    base_length_km=DEFAULT_BASE_LENGTH_KM,
    cells_per_base_length=DEFAULT_CELLS_PER_BASE_LENGTH,
    sensor_fractions=DEFAULT_SENSOR_FRACTIONS,
)
    return SquareSingleScenarioSetup(
        SINGLE_SCENARIO_LEGACY_TEMPLATE.T,
        SINGLE_SCENARIO_LEGACY_TEMPLATE.CFL,
        Float64(base_length_km),
        Int(cells_per_base_length),
        copy(SINGLE_SCENARIO_LEGACY_TEMPLATE.control_times),
        copy(SINGLE_SCENARIO_LEGACY_TEMPLATE.observed_road_ids),
        copy_sensor_spec(sensor_fractions),
        copy(SINGLE_SCENARIO_LEGACY_TEMPLATE.boundary_road_ids),
        copy(SINGLE_SCENARIO_LEGACY_TEMPLATE.inflows),
        copy(SINGLE_SCENARIO_LEGACY_TEMPLATE.road_profiles),
        single_scenario_road_length_multipliers(base_length_km),
        copy(SINGLE_SCENARIO_LEGACY_TEMPLATE.speed_limits),
        Float64(peak_noise_sigma),
    )
end

road_length_multiplier(setup::SquareSingleScenarioSetup, road_id::Int) = setup.road_length_multipliers[road_id]
road_length_km(setup::SquareSingleScenarioSetup, road_id::Int) = setup.base_length_km * road_length_multiplier(setup, road_id)
road_cell_count(setup::SquareSingleScenarioSetup, road_id::Int) = setup.cells_per_base_length * road_length_multiplier(setup, road_id)
uniform_physical_dx_meters(setup::SquareSingleScenarioSetup) = 1000.0 * setup.base_length_km / setup.cells_per_base_length

function road_unit_cell_centers(setup::SquareSingleScenarioSetup, road_id::Int)
    n_cells = road_cell_count(setup, road_id)
    return collect(range(0.5 / n_cells, 1.0 - 0.5 / n_cells; length=n_cells))
end

function road_initial_profile(setup::SquareSingleScenarioSetup, road_id::Int)
    return [setup.road_profiles[road_id](x) for x in road_unit_cell_centers(setup, road_id)]
end

function road_cell_centers_meters(setup::SquareSingleScenarioSetup, road_id::Int)
    n_cells = road_cell_count(setup, road_id)
    dx_m = 1000.0 * road_length_km(setup, road_id) / n_cells
    road_length_m = 1000.0 * road_length_km(setup, road_id)
    return collect(range(dx_m / 2, road_length_m - dx_m / 2; length=n_cells))
end

function sensor_cell_id(n_cells::Int, fraction::Real)
    return clamp(round(Int, Float64(fraction) * n_cells + 0.5), 2, n_cells - 1)
end

function road_sensor_cell_ids(setup::SquareSingleScenarioSetup, road_id::Int)
    n_cells = road_cell_count(setup, road_id)
    sensor_spec = setup.sensor_fractions
    values = sensor_spec isa AbstractDict ? sensor_spec[road_id] : sensor_spec
    ids = [
        value isa Integer ? clamp(Int(value), 1, n_cells) : sensor_cell_id(n_cells, value)
        for value in values
    ]
    @assert length(unique(ids)) == length(ids) "Sensor fractions collapse to duplicate cells on road $(road_id)."
    return ids
end

function observation_blocks(setup::SquareSingleScenarioSetup)
    blocks = NamedTuple[]
    start_idx = 1

    for road_id in setup.observed_road_ids
        sensor_ids = road_sensor_cell_ids(setup, road_id)
        block_length = length(sensor_ids) * length(setup.control_times)
        block_range = start_idx:(start_idx + block_length - 1)
        push!(blocks, (road_id=road_id, sensor_ids=sensor_ids, indices=block_range))
        start_idx += block_length
    end

    return blocks
end

observation_length(setup::SquareSingleScenarioSetup) = sum(length(block.indices) for block in observation_blocks(setup))

function observation_group_indices(setup::SquareSingleScenarioSetup)
    groups = Dict(:incoming => Int[], :outgoing => Int[], :internal => Int[])
    for block in observation_blocks(setup)
        append!(groups[road_role_symbol(block.road_id)], block.indices)
    end
    return groups
end

