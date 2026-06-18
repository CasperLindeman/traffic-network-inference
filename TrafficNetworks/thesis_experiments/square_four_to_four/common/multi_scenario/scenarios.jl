# Shared helpers for square-four-to-four multi-scenario experiments.

function base_cycle_hours(setup::SquareSingleScenarioSetup)
    return first(SINGLE_SCENARIO_LEGACY_TEMPLATE.control_times)
end

wrap_time_to_horizon(t, T) = mod(t, T)

function piecewise_linear_value(x::Real, knots, values)
    @assert length(knots) == length(values)
    x_clamped = clamp(x, first(knots), last(knots))

    for idx in 1:(length(knots) - 1)
        x_left = knots[idx]
        x_right = knots[idx + 1]
        if x_clamped <= x_right
            width = x_right - x_left
            if width == 0
                return values[idx + 1]
            end
            weight = (x_clamped - x_left) / width
            return (one(weight) - weight) * values[idx] + weight * values[idx + 1]
        end
    end

    return values[end]
end

function incoming_mode_weight(road_id::Int, mode::Symbol)
    idx = findfirst(==(road_id), EXTERNAL_INCOMING_ROADS)
    idx === nothing && error("Road $(road_id) is not an external incoming road.")

    group = fld(idx - 1, 2) + 1
    row_sign = group in (1, 2) ? -1.0 : 1.0
    col_sign = group in (1, 3) ? -1.0 : 1.0
    pair_sign = isodd(idx) ? -1.0 : 1.0

    if mode == :balanced
        return 0.0
    elseif mode == :north_heavy
        return row_sign < 0 ? 1.0 : -1.0
    elseif mode == :south_heavy
        return row_sign > 0 ? 1.0 : -1.0
    elseif mode == :west_heavy
        return col_sign < 0 ? 1.0 : -1.0
    elseif mode == :east_heavy
        return col_sign > 0 ? 1.0 : -1.0
    elseif mode == :alternating
        return pair_sign
    elseif mode == :checker
        return row_sign * col_sign
    elseif mode == :corner_focus
        return group in (1, 4) ? 1.0 : -1.0
    end

    error("Unknown BC mode $(mode).")
end

function road_spatial_signs(road_id::Int)
    if 1 <= road_id <= 4
        return (-1.0, -1.0)
    elseif 5 <= road_id <= 8
        return (-1.0, 1.0)
    elseif 9 <= road_id <= 12
        return (1.0, -1.0)
    elseif 13 <= road_id <= 16
        return (1.0, 1.0)
    elseif road_id in (17, 18)
        return (-1.0, 0.0)
    elseif road_id in (19, 20)
        return (0.0, -1.0)
    elseif road_id in (21, 22)
        return (0.0, 1.0)
    elseif road_id in (23, 24)
        return (1.0, 0.0)
    end

    error("Unknown road id $(road_id).")
end

function profile_mode_weight(road_id::Int, mode::Symbol)
    row_sign, col_sign = road_spatial_signs(road_id)
    pair_sign = isodd(road_id) ? -1.0 : 1.0
    role = road_role_symbol(road_id)

    if mode == :balanced
        return 0.0
    elseif mode == :north_heavy
        return row_sign < 0 ? 1.0 : row_sign > 0 ? -1.0 : 0.4
    elseif mode == :south_heavy
        return row_sign > 0 ? 1.0 : row_sign < 0 ? -1.0 : 0.4
    elseif mode == :west_heavy
        return col_sign < 0 ? 1.0 : col_sign > 0 ? -1.0 : 0.4
    elseif mode == :east_heavy
        return col_sign > 0 ? 1.0 : col_sign < 0 ? -1.0 : 0.4
    elseif mode == :checker
        return row_sign == 0.0 || col_sign == 0.0 ? 0.5 * pair_sign : row_sign * col_sign
    elseif mode == :incoming_loaded
        return role == :incoming ? 1.0 : role == :internal ? 0.2 : -0.8
    elseif mode == :outgoing_loaded
        return role == :outgoing ? 1.0 : role == :internal ? 0.2 : -0.8
    elseif mode == :internal_loaded
        return role == :internal ? 1.0 : -0.4
    elseif mode == :corner_focus
        return row_sign * col_sign
    end

    error("Unknown IC mode $(mode).")
end

function multi_scenario_scenario_library()
    return MultiScenarioScenarioSpec[
        MultiScenarioScenarioSpec(
            "balanced_reference",
            (0.20, 0.22, 0.08, 0.21, 0.10),
            :balanced,
            0.00,
            (0.62, 0.38, 0.14, 0.30, 0.54),
            :balanced,
            0.00,
        ),
        MultiScenarioScenarioSpec(
            "heavy_from_start",
            (0.24, 0.23, 0.16, 0.10, 0.06),
            :balanced,
            0.00,
            (0.78, 0.60, 0.36, 0.20, 0.12),
            :incoming_loaded,
            0.06,
        ),
        MultiScenarioScenarioSpec(
            "quiet_start_then_build",
            (0.03, 0.05, 0.09, 0.18, 0.23),
            :balanced,
            0.00,
            (0.08, 0.10, 0.16, 0.28, 0.36),
            :balanced,
            0.03,
        ),
        MultiScenarioScenarioSpec(
            "north_heavy",
            (0.16, 0.19, 0.10, 0.17, 0.09),
            :north_heavy,
            0.045,
            (0.54, 0.32, 0.14, 0.24, 0.40),
            :north_heavy,
            0.08,
        ),
        MultiScenarioScenarioSpec(
            "south_heavy",
            (0.15, 0.18, 0.11, 0.19, 0.10),
            :south_heavy,
            0.045,
            (0.20, 0.18, 0.18, 0.34, 0.58),
            :south_heavy,
            0.08,
        ),
        MultiScenarioScenarioSpec(
            "west_heavy",
            (0.17, 0.20, 0.11, 0.18, 0.09),
            :west_heavy,
            0.045,
            (0.58, 0.42, 0.22, 0.18, 0.24),
            :west_heavy,
            0.07,
        ),
        MultiScenarioScenarioSpec(
            "east_heavy_late",
            (0.09, 0.13, 0.18, 0.21, 0.15),
            :east_heavy,
            0.045,
            (0.16, 0.20, 0.24, 0.42, 0.60),
            :east_heavy,
            0.07,
        ),
        MultiScenarioScenarioSpec(
            "alternating_inlets",
            (0.20, 0.09, 0.22, 0.10, 0.21),
            :alternating,
            0.040,
            (0.40, 0.18, 0.30, 0.18, 0.44),
            :checker,
            0.06,
        ),
        MultiScenarioScenarioSpec(
            "early_spike_then_clear",
            (0.25, 0.17, 0.08, 0.06, 0.05),
            :corner_focus,
            0.030,
            (0.74, 0.50, 0.22, 0.10, 0.08),
            :incoming_loaded,
            0.06,
        ),
        MultiScenarioScenarioSpec(
            "late_surge_sparse_start",
            (0.04, 0.05, 0.10, 0.20, 0.24),
            :corner_focus,
            0.030,
            (0.10, 0.12, 0.18, 0.34, 0.56),
            :outgoing_loaded,
            0.06,
        ),
        MultiScenarioScenarioSpec(
            "upstream_loaded",
            (0.13, 0.16, 0.12, 0.16, 0.11),
            :balanced,
            0.00,
            (0.84, 0.60, 0.22, 0.10, 0.06),
            :internal_loaded,
            0.06,
        ),
        MultiScenarioScenarioSpec(
            "downstream_loaded_sparse",
            (0.06, 0.10, 0.14, 0.18, 0.12),
            :checker,
            0.030,
            (0.06, 0.10, 0.24, 0.54, 0.80),
            :outgoing_loaded,
            0.06,
        ),
    ]
end

function multi_scenario_scenario_specs(count::Int)
    specs = multi_scenario_scenario_library()
    count <= length(specs) || error("Requested $(count) scenarios, but the multi-scenario library only defines $(length(specs)).")
    return specs[1:count]
end

