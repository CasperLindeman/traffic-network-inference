module Solvers

export simulate!, SimulationHistory

struct SimulationHistory
    times::Vector{Float64}
    road_histories::Vector{<:AbstractMatrix{<:Real}}
end

using ..RoadNetworks: RoadNetwork
using ..Fluxes: demand
using ..Roads: cfl_dt, update_road!
using ..Boundaries: boundary_cfl_flux, boundary_flux!
using ..Junctions: compute_junction_fluxes

const TIME_TOL = 1e-10

function simulate!(net; times=nothing, save_every::Union{Nothing, Int}=nothing)
    t = 0.0
    control_times = normalize_control_times(times, net.T)
    next_time_idx = 1
    if save_every !== nothing
        @assert save_every > 0 "save_every must be positive"
    end

    roads = net.roads
    junctions = net.junctions
    boundaries = net.boundaries
    state_T = state_eltype(roads)

    nroads = length(roads)
    has_downstream_junction = downstream_junction_mask(junctions, nroads)

    itr = 0
    save_idx = 0

    # Preallocate history if needed
    history_times = Float64[]
    history_roads = nothing

    if control_times !== nothing
        history_times = copy(control_times)
        history_roads = [Matrix{state_T}(undef, length(road.rho), length(control_times)) for road in roads]
    elseif save_every !== nothing
        # Estimate number of saves: roughly T/dt/save_every
        # Conservative estimate: assume dt ~ 0.01 * T / (2 * CFL) as a rough guess
        estimated_saves = max(10, ceil(Int, net.T / (0.01 * net.CFL) / save_every))
        history_times = sizehint!(Float64[], estimated_saves)

        # Preallocate matrices for each road: (ncells, estimated_saves)
        history_roads = [Matrix{state_T}(undef, length(road.rho), estimated_saves) for road in roads]
    end

    if control_times !== nothing && next_time_idx <= length(control_times) &&
       isapprox(control_times[next_time_idx], 0.0; atol=TIME_TOL, rtol=0.0)
        save_idx += 1
        store_state!(history_roads, roads, save_idx)
        next_time_idx += 1
    end

    while t < net.T - TIME_TOL

        ##### Endpoint fluxes and time stepping
        left_flux, right_flux = endpoint_fluxes_for_cfl(
            roads,
            junctions,
            boundaries,
            has_downstream_junction,
            t,
            state_T,
        )

        dt = compute_dt(t, net, roads, control_times, next_time_idx, left_flux, right_flux)

        # Replace the queue-free CFL estimates by the actual queue-aware
        # boundary fluxes used in the conservative road update.
        for b in boundaries
            left_flux[b.road_id] = boundary_flux!(b, roads[b.road_id], t, dt)
        end
        #####

        for i in 1:nroads
            update_road!(roads[i], dt, left_flux[i], right_flux[i])
        end

        t += dt
        itr += 1

        ##### Save history if needed
        if control_times !== nothing
            if next_time_idx <= length(control_times) &&
               isapprox(t, control_times[next_time_idx]; atol=TIME_TOL, rtol=0.0)
                save_idx += 1
                store_state!(history_roads, roads, save_idx)
                next_time_idx += 1
            end
        elseif save_every !== nothing && itr % save_every == 0
            save_idx += 1
            # Expand history if needed
            if save_idx > length(history_times)
                # Double the capacity
                new_capacity = save_idx * 2
                resize!(history_times, new_capacity)
                for i in 1:nroads
                    new_matrix = Matrix{state_T}(undef, size(history_roads[i], 1), new_capacity)
                    copyto!(new_matrix, 1, history_roads[i], 1, size(history_roads[i], 1) * (save_idx - 1))
                    history_roads[i] = new_matrix
                end
            end

            # Store the current state
            history_times[save_idx] = Float64(t)
            store_state!(history_roads, roads, save_idx)
        #####
        end
    end

    if control_times !== nothing
        @assert save_idx == length(control_times) "Failed to hit all requested control times"
        return SimulationHistory(history_times, history_roads)
    elseif save_every !== nothing
        # Trim to actual size
        resize!(history_times, save_idx)
        for i in 1:nroads
            history_roads[i] = history_roads[i][:, 1:save_idx]
        end
        return SimulationHistory(history_times, history_roads)
    else
        return nothing
    end
end

function compute_dt(t, net, roads, control_times, next_time_idx, left_flux, right_flux)
    dt = minimum(cfl_dt(roads[i], net.CFL, left_flux[i], right_flux[i]) for i in eachindex(roads))
    return clip_dt_to_time_targets(dt, t, net, control_times, next_time_idx)
end

function clip_dt_to_time_targets(dt, t, net, control_times, next_time_idx)
    dt = min(dt, net.T - t)

    if control_times !== nothing && next_time_idx <= length(control_times)
        t_target = control_times[next_time_idx]
        dt_to_target = t_target - t
        if dt_to_target > TIME_TOL && dt > dt_to_target
            dt = dt_to_target
        end
    end

    return dt
end

function endpoint_fluxes_for_cfl(roads, junctions, boundaries, has_downstream_junction, t, state_T)
    nroads = length(roads)
    left_flux  = zeros(state_T, nroads)
    right_flux = zeros(state_T, nroads)

    # The queue release term is proportional to 1/dt, so including it here
    # would make the CFL estimate circular. Omitting it gives a smaller flux
    # and therefore a larger endpoint wave speed for Greenshields flux.
    for b in boundaries
        left_flux[b.road_id] = boundary_cfl_flux(b, roads[b.road_id], t)
    end

    for j in junctions
        fin, fout = compute_junction_fluxes(j, roads, t)

        for (k, rid) in enumerate(j.incoming)
            right_flux[rid] = fin[k]
        end
        for (k, rid) in enumerate(j.outgoing)
            left_flux[rid] = fout[k]
        end
    end

    # Roads that do not feed into another junction should drain through
    # an open downstream boundary instead of being artificially closed.
    for i in 1:nroads
        if !has_downstream_junction[i]
            right_flux[i] = open_downstream_flux(roads[i])
        end
    end

    return left_flux, right_flux
end

function normalize_control_times(times, T)
    if times === nothing
        return nothing
    end

    control_times = Float64.(collect(times))
    @assert issorted(control_times) "Control times must be sorted"
    @assert all(diff(control_times) .> 0.0) "Control times must be strictly increasing"
    @assert all((-TIME_TOL .<= control_times) .& (control_times .<= T + TIME_TOL)) "Control times must lie in [0, T]"

    return clamp.(control_times, 0.0, T)
end

function downstream_junction_mask(junctions, nroads)
    has_downstream_junction = falses(nroads)
    for j in junctions
        for rid in j.incoming
            has_downstream_junction[rid] = true
        end
    end
    return has_downstream_junction
end

function state_eltype(roads)
    return promote_type(map(road -> eltype(road.rho), roads)...)
end

@inline function open_downstream_flux(road)
    return demand(road) / (road.rho_max * road.speed_limit)
end

function store_state!(history_roads, roads, save_idx)
    for i in eachindex(roads)
        copyto!(view(history_roads[i], :, save_idx), roads[i].rho)
    end
    return nothing
end

end # module Solvers
