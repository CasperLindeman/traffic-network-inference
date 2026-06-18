"""
Boundary inflow objects and reusable inflow-signal constructors.
"""
module Boundaries

export
    Boundary,
    boundary_flux,
    boundary_cfl_flux,
    boundary_flux!,
    make_inflow_signal,
    make_piecewise_signal

const MAX_NORMALIZED_FLUX = 0.25

"""
    Boundary(road_id, inflow; queue=0.0)

Upstream boundary condition for a road. The `inflow` function is evaluated at
simulation time `t` and returns a normalized external arrival flux. The queue
stores physical waiting traffic at the boundary.
"""
mutable struct Boundary
    road_id::Int
    inflow::Function
    queue::Real
end

function Boundary(road_id::Integer, inflow; queue::Real=0.0)
    return Boundary(Int(road_id), inflow, queue)
end

@inline function external_arrival_flux(b::Boundary, t::Real)
    arrival = b.inflow(t)
    return max(zero(arrival), arrival)
end

@inline function boundary_flux(b::Boundary, t::Real)
    return min(MAX_NORMALIZED_FLUX, external_arrival_flux(b, t))
end

@inline flow_scale(road) = road.rho_max * road.speed_limit

@inline function upstream_supply(road)
    rho_L = road.rho[1]
    qmax = MAX_NORMALIZED_FLUX * flow_scale(road)

    if rho_L <= 0.5
        return qmax
    else
        return rho_L * (1 - rho_L) * flow_scale(road)
    end
end

"""
    boundary_cfl_flux(b, road, t)

Compute a conservative, queue-free inflow flux for the CFL estimate. The actual
time-step update uses `boundary_flux!`, which releases queued traffic over the
chosen time step.
"""
function boundary_cfl_flux(b::Boundary, road, t::Real)
    scale = flow_scale(road)
    arrival = external_arrival_flux(b, t)
    arrival_physical = scale * arrival
    max_physical = scale * MAX_NORMALIZED_FLUX

    external_demand = min(max_physical, arrival_physical)
    accepted_physical = min(external_demand, upstream_supply(road))

    return accepted_physical / scale
end

"""
    boundary_flux!(b, road, t, dt)

Compute the queue-aware inflow boundary flux for `road` and advance the
boundary queue over one time step. The returned flux is dimensionless, while
the queue is stored in physical flow-time units.
"""
function boundary_flux!(b::Boundary, road, t::Real, dt::Real)
    scale = flow_scale(road)
    arrival = external_arrival_flux(b, t)
    arrival_physical = scale * arrival
    max_physical = scale * MAX_NORMALIZED_FLUX

    external_demand = min(max_physical, arrival_physical + b.queue / dt)
    accepted_physical = min(external_demand, upstream_supply(road))
    b.queue += dt * (arrival_physical - accepted_physical)

    return accepted_physical / scale
end

"""
    make_inflow_signal(base, pulses...; lower=0.0, upper=0.24)

Clipped base inflow plus Gaussian pulses. Each pulse is
`(amplitude, center_time, width)` in hours.
"""
function make_inflow_signal(base::Real, pulses...; lower::Real=0.0, upper::Real=0.24)
    return t -> clamp(
        base + sum(amp * exp(-0.5 * ((t - center) / width)^2) for (amp, center, width) in pulses),
        lower,
        upper,
    )
end

"""
    make_piecewise_signal(default_value, segments...; lower=0.0, upper=0.24)

Piecewise-constant boundary inflow. Each segment is
`(t_start, t_stop, value)` in hours.
"""
function make_piecewise_signal(default_value::Real, segments...; lower::Real=0.0, upper::Real=0.24)
    return t -> begin
        value = default_value
        for (t_start, t_stop, segment_value) in segments
            if t_start <= t < t_stop
                value = segment_value
            end
        end
        clamp(value, lower, upper)
    end
end

end # module Boundaries
