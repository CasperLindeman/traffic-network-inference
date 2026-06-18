# Shared helpers for square-four-to-four turning-outlier diagnostics.

function dataset_observed_road_indices(dataset::MultiScenarioDataset)
    indices_by_road = Dict{Int, Vector{Int}}()
    global_offset = 0

    for setup in dataset.setups
        for block in observation_blocks(setup)
            road_indices = get!(indices_by_road, block.road_id, Int[])
            append!(road_indices, global_offset .+ collect(block.indices))
        end
        global_offset += observation_length(setup)
    end

    return indices_by_road
end

function observation_component_metadata(dataset::MultiScenarioDataset)
    rows = NamedTuple[]
    global_offset = 0

    for (scenario_idx, setup) in enumerate(dataset.setups)
        for block in observation_blocks(setup)
            sensor_ids = collect(block.sensor_ids)
            n_sensors = length(sensor_ids)
            road_n_cells = road_cell_count(setup, block.road_id)

            for (time_idx, time_hour) in enumerate(setup.control_times)
                for (sensor_pos, sensor_cell) in enumerate(sensor_ids)
                    local_index = first(block.indices) + (time_idx - 1) * n_sensors + sensor_pos - 1
                    global_index = global_offset + local_index

                    push!(
                        rows,
                        (
                            global_index=global_index,
                            scenario_index=scenario_idx,
                            scenario_label=dataset.labels[scenario_idx],
                            local_index=local_index,
                            road_id=block.road_id,
                            road_role=String(road_role_symbol(block.road_id)),
                            sensor_position=sensor_pos,
                            sensor_cell=sensor_cell,
                            sensor_fraction=(sensor_cell - 0.5) / road_n_cells,
                            time_index=time_idx,
                            time_hour=time_hour,
                            time_min=60.0 * time_hour,
                        ),
                    )
                end
            end
        end

        global_offset += observation_length(setup)
    end

    return rows
end

function observation_metadata_lookup(dataset::MultiScenarioDataset)
    return Dict(row.global_index => row for row in observation_component_metadata(dataset))
end

function top_sum_fraction(values::AbstractVector, count::Int)
    isempty(values) && return 0.0
    total = sum(values)
    total <= 0.0 && return 0.0
    ordered = sort(Float64.(values); rev=true)
    return sum(first(ordered, min(count, length(ordered)))) / total
end

function hard_floor_uniform_probability(effective_shape_floor::Real)
    a = clamp(Float64(effective_shape_floor), 0.0, 1.0)
    return 1.0 - sqrt(1.0 - a)
end

hard_floor_variant_label(floor_fraction::Real) =
    replace(@sprintf("hard_floor_%.2f", Float64(floor_fraction)), "." => "p")

function sigma_variant(y_obs::AbstractVector, peak_noise_sigma::Real, variant_kind::Symbol, floor_fraction::Real)
    peak = Float64(peak_noise_sigma)
    floor_value = max(1e-3, Float64(floor_fraction) * peak)
    flux_sigma = physical_noise_sigma(y_obs, peak)

    if variant_kind == :hard_floor
        return max.(flux_sigma, floor_value)
    elseif variant_kind == :additive_floor
        return sqrt.(flux_sigma .^ 2 .+ floor_value^2)
    end

    error("Unknown sigma variant $(variant_kind).")
end

function sigma_variant_specs()
    return [
        (label="hard_floor_0p03", kind=:hard_floor, floor_fraction=0.03),
        (label="hard_floor_0p05", kind=:hard_floor, floor_fraction=0.05),
        (label="hard_floor_0p10", kind=:hard_floor, floor_fraction=0.10),
        (label="hard_floor_0p15", kind=:hard_floor, floor_fraction=0.15),
        (label="additive_floor_0p03", kind=:additive_floor, floor_fraction=0.03),
        (label="additive_floor_0p10", kind=:additive_floor, floor_fraction=0.10),
    ]
end

function noise_weight_summary_rows(observations, peak_noise_sigma::Real; specs=sigma_variant_specs())
    rows = NamedTuple[]
    n_obs = length(observations.y_obs)

    for spec in specs
        sigma = sigma_variant(observations.y_obs, peak_noise_sigma, spec.kind, spec.floor_fraction)
        peak = Float64(peak_noise_sigma)
        floor_value = max(1e-3, Float64(spec.floor_fraction) * peak)
        weights = 1.0 ./ sigma .^ 2
        true_standardized = (observations.y_true .- observations.y_obs) ./ sigma
        true_contrib = 0.5 .* true_standardized .^ 2
        near_floor = sigma .<= 1.05 * floor_value
        at_floor = sigma .<= floor_value * (1.0 + 1e-10)
        effective_shape_floor = floor_value / peak
        uniform_floor_fraction = spec.kind == :hard_floor ? hard_floor_uniform_probability(effective_shape_floor) : NaN

        push!(
            rows,
            (
                variant_label=spec.label,
                variant_kind=String(spec.kind),
                peak_noise_sigma=peak,
                floor_fraction=spec.floor_fraction,
                floor_value=floor_value,
                observation_count=n_obs,
                sigma_min=minimum(sigma),
                sigma_q01=quantile(sigma, 0.01),
                sigma_q05=quantile(sigma, 0.05),
                sigma_median=median(sigma),
                sigma_mean=mean(sigma),
                sigma_q95=quantile(sigma, 0.95),
                sigma_max=maximum(sigma),
                at_floor_count=count(at_floor),
                at_floor_fraction=mean(at_floor),
                near_floor_5pct_count=count(near_floor),
                near_floor_5pct_fraction=mean(near_floor),
                uniform_floor_fraction=uniform_floor_fraction,
                uniform_expected_floor_count=uniform_floor_fraction * n_obs,
                max_to_min_sigma=maximum(sigma) / minimum(sigma),
                max_to_min_weight=maximum(weights) / minimum(weights),
                max_abs_true_standardized=max(abs.(true_standardized)...),
                count_abs_true_residual_gt_3=count(abs.(true_standardized) .> 3.0),
                count_abs_true_residual_gt_5=count(abs.(true_standardized) .> 5.0),
                count_abs_true_residual_gt_10=count(abs.(true_standardized) .> 10.0),
                count_abs_true_residual_gt_20=count(abs.(true_standardized) .> 20.0),
                count_abs_true_residual_gt_50=count(abs.(true_standardized) .> 50.0),
                true_loss_top1_fraction=top_sum_fraction(true_contrib, 1),
                true_loss_top5_fraction=top_sum_fraction(true_contrib, 5),
                true_loss_top10_fraction=top_sum_fraction(true_contrib, 10),
            ),
        )
    end

    return rows
end

function noise_observation_outlier_rows(dataset::MultiScenarioDataset, observations, peak_noise_sigma::Real; variant_label="hard_floor_0p03", variant_kind=:hard_floor, floor_fraction=0.03, max_rows=100)
    metadata = observation_metadata_lookup(dataset)
    sigma = sigma_variant(observations.y_obs, peak_noise_sigma, variant_kind, floor_fraction)
    standardized = (observations.y_true .- observations.y_obs) ./ sigma
    contrib = 0.5 .* standardized .^ 2
    order = sortperm(abs.(standardized); rev=true)
    rows = NamedTuple[]

    for (rank, idx) in enumerate(first(order, min(max_rows, length(order))))
        meta = metadata[idx]
        push!(
            rows,
            merge(
                (
                    rank=rank,
                    variant_label=variant_label,
                    y_true=observations.y_true[idx],
                    y_obs=observations.y_obs[idx],
                    sigma_true=observations.sigma_true[idx],
                    sigma_model=sigma[idx],
                    flux_shape_obs=parabolic_noise_shape(observations.y_obs[idx]),
                    true_minus_obs=observations.y_true[idx] - observations.y_obs[idx],
                    true_standardized_residual=standardized[idx],
                    true_loss_contribution=contrib[idx],
                    at_floor=sigma[idx] <= max(1e-3, floor_fraction * peak_noise_sigma) * (1.0 + 1e-10),
                ),
                meta,
            ),
        )
    end

    return rows
end

