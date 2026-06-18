"""
Small helpers for time units, control times, and initial-density profiles.
"""
module Utils

export
    clamp01,
    seconds_to_hours,
    minutes_to_hours,
    control_times_seconds,
    regular_control_times,
    make_profile,
    scale_profile_domain,
    make_profile_sum,
    make_piecewise_profile

@inline clamp01(x::Float64) = min(max(x, 0.0), 1.0)

@inline seconds_to_hours(seconds::Real) = seconds / 3600.0

@inline minutes_to_hours(minutes::Real) = minutes / 60.0

control_times_seconds(control_times::AbstractVector) =
    round.(Int, 3600.0 .* control_times)

control_times_seconds(setup) =
    control_times_seconds(getproperty(setup, :control_times))

"""
    regular_control_times(; step_seconds, horizon_seconds=nothing, horizon_minutes=nothing)

Return regularly spaced control/output times in hours. Exactly one horizon
keyword must be supplied.
"""
function regular_control_times(; step_seconds::Real, horizon_seconds=nothing, horizon_minutes=nothing)
    @assert xor(horizon_seconds !== nothing, horizon_minutes !== nothing) "Specify exactly one horizon."
    stop_time = horizon_seconds === nothing ? minutes_to_hours(horizon_minutes) : seconds_to_hours(horizon_seconds)
    return collect(seconds_to_hours(step_seconds):seconds_to_hours(step_seconds):stop_time)
end

"""
    make_profile(base, amp, rate, center)

Single Gaussian-like initial-density profile on the unit road coordinate.
"""
function make_profile(base::Real, amp::Real, rate::Real, center::Real)
    return x -> base + amp * exp(-rate * (x - center)^2)
end

"""
    scale_profile_domain(profile, scale)

Adapt a unit-coordinate profile to a block-scaled road coordinate.
"""
scale_profile_domain(profile::Function, scale::Real) =
    x -> profile(x / scale)

"""
    make_profile_sum(base, peaks...; lower=0.0, upper=0.95)

Initial-density profile made from a clipped base level plus Gaussian peaks.
Each peak is `(amplitude, rate, center)`.
"""
function make_profile_sum(base::Real, peaks...; lower::Real=0.0, upper::Real=0.95)
    return x -> clamp(
        base + sum(amp * exp(-rate * (x - center)^2) for (amp, rate, center) in peaks),
        lower,
        upper,
    )
end

"""
    make_piecewise_profile(default_value, segments...; lower=0.0, upper=0.95, scale=1.0)

Piecewise-constant initial-density profile. Each segment is
`(x_start, x_stop, value)` on the scaled unit coordinate.
"""
function make_piecewise_profile(default_value::Real, segments...; lower::Real=0.0, upper::Real=0.95, scale::Real=1.0)
    return x -> begin
        x_scaled = x / scale
        value = default_value
        for (x_start, x_stop, segment_value) in segments
            if x_start <= x_scaled < x_stop
                value = segment_value
            end
        end
        clamp(value, lower, upper)
    end
end

end # module Utils
