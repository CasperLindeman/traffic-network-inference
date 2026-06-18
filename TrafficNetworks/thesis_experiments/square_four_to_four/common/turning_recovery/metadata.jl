# Shared helpers for square-four-to-four turning-outlier diagnostics.

turning_entry_global_index(junction::Int, incoming_row::Int, outgoing_col::Int) =
    16 * (junction - 1) + 4 * (incoming_row - 1) + outgoing_col

turning_entry_label(junction::Int, incoming_row::Int, outgoing_col::Int) =
    @sprintf("J%d P%d%d", junction, incoming_row, outgoing_col)

function direct_observation_class(incoming_observed::Bool, outgoing_observed::Bool)
    if incoming_observed && outgoing_observed
        return "source+target observed"
    elseif incoming_observed
        return "source observed"
    elseif outgoing_observed
        return "target observed"
    end

    return "neither directly observed"
end

function downstream_observation_class(distance::Real)
    if !isfinite(distance)
        return "no downstream observed road"
    elseif distance == 0
        return "target observed"
    elseif distance == 1
        return "observed after 1 turn"
    end

    return @sprintf("observed after %.0f turns", distance)
end

function downstream_observation_distances(setup::SquareSingleScenarioSetup)
    memo = Dict{Int, Float64}()
    visiting = Set{Int}()

    function distance_from_road(road_id::Int)
        road_id in setup.observed_road_ids && return 0.0
        haskey(memo, road_id) && return memo[road_id]
        road_id in visiting && return Inf

        push!(visiting, road_id)
        best = Inf

        for junction in 1:N_JUNCTIONS
            road_id in JUNCTION_INCOMING_ROADS[junction] || continue
            child_distances = [distance_from_road(outgoing_road) for outgoing_road in JUNCTION_OUTGOING_ROADS[junction]]
            finite_children = [value for value in child_distances if isfinite(value)]
            if !isempty(finite_children)
                best = min(best, 1.0 + minimum(finite_children))
            end
        end

        delete!(visiting, road_id)
        memo[road_id] = best
        return best
    end

    return Dict(road_id => distance_from_road(road_id) for road_id in all_square_road_ids())
end

function turning_entry_metadata(setup::SquareSingleScenarioSetup)
    distances = downstream_observation_distances(setup)
    rows = NamedTuple[]

    for junction in 1:N_JUNCTIONS
        for incoming_row in 1:4
            incoming_road = JUNCTION_INCOMING_ROADS[junction][incoming_row]
            incoming_observed = incoming_road in setup.observed_road_ids

            for outgoing_col in 1:4
                outgoing_road = JUNCTION_OUTGOING_ROADS[junction][outgoing_col]
                outgoing_observed = outgoing_road in setup.observed_road_ids
                distance = distances[outgoing_road]

                push!(
                    rows,
                    (
                        global_entry=turning_entry_global_index(junction, incoming_row, outgoing_col),
                        entry_label=turning_entry_label(junction, incoming_row, outgoing_col),
                        junction=junction,
                        junction_label=JUNCTION_LABELS[junction],
                        incoming_row=incoming_row,
                        outgoing_col=outgoing_col,
                        source_road=incoming_road,
                        target_road=outgoing_road,
                        incoming_road=incoming_road,
                        outgoing_road=outgoing_road,
                        source_role=String(road_role_symbol(incoming_road)),
                        target_role=String(road_role_symbol(outgoing_road)),
                        incoming_role=String(road_role_symbol(incoming_road)),
                        outgoing_role=String(road_role_symbol(outgoing_road)),
                        source_observed=incoming_observed,
                        target_observed=outgoing_observed,
                        incoming_observed=incoming_observed,
                        outgoing_observed=outgoing_observed,
                        direct_observed_count=Int(incoming_observed) + Int(outgoing_observed),
                        direct_observation_class=direct_observation_class(incoming_observed, outgoing_observed),
                        target_observation_distance=distance,
                        downstream_observation_class=downstream_observation_class(distance),
                    ),
                )
            end
        end
    end

    return rows
end

function road_activity_summary(P_true, dataset::MultiScenarioDataset)
    road_values = Dict(road_id => Float64[] for road_id in all_square_road_ids())

    for setup in dataset.setups
        hist = simulate_history(P_true, setup)
        for road_id in all_square_road_ids()
            append!(road_values[road_id], vec(Float64.(hist.road_histories[road_id])))
        end
    end

    return Dict(
        road_id => (
            mean_density=mean(values),
            density_span=maximum(values) - minimum(values),
            density_std=length(values) > 1 ? std(values) : 0.0,
        )
        for (road_id, values) in road_values
    )
end

