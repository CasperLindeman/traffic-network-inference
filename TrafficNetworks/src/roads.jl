module Roads

export Road, make_road, cfl_dt, update_road!

using ..Fluxes: maxwavespeed, endpoint_wavespeed, godunov_flux

struct Road
    id::Int
    dx::Real
    rho::AbstractVector{<:Real}
    F::AbstractVector{<:Real}
    speed_limit::Real
    rho_max::Real
    gamma::Real
end

function make_road(id, b, L, N, rho_0, speed_limit, rho_max; state_eltype::Type{<:Real}=Float64)
    dx = 1 / N
    x = range(dx/2, b - dx/2, length=b*N)
    rho = state_eltype[rho_0(xi) for xi in x]
    F = zeros(state_eltype, b*N + 1)
    gamma = speed_limit / (b*L)
    return Road(id, dx, rho, F, speed_limit, rho_max, gamma)
end

function interior_wavespeed(road)
    vmax = zero(eltype(road.rho))
    @inbounds @simd for r in road.rho
        vmax = max(vmax, maxwavespeed(r))
    end
    return vmax
end

function cfl_dt(road, CFL)
    vmax = interior_wavespeed(road)
    return CFL * road.dx / (road.gamma * max(vmax, 1e-12 * one(vmax)))
end

function cfl_dt(road, CFL, left_flux, right_flux)
    vmax = max(
        interior_wavespeed(road),
        endpoint_wavespeed(left_flux),
        endpoint_wavespeed(right_flux),
    )
    return CFL * road.dx / (road.gamma * max(vmax, 1e-12 * one(vmax)))
end

"""
function update_road!(road, dt, left_flux, right_flux)
    rho = road.rho
    F   = road.F
    N   = length(rho)
    a = road.gamma*dt/road.dx

    F[1] = left_flux
    @inbounds for i in 2:N
        F[i] = godunov_flux(rho[i-1], rho[i])
    end
    F[N+1] = right_flux

    @inbounds for i in 1:N
        rho[i] -= a * (F[i+1] - F[i])
    end

    return nothing
end
"""

function update_road!(road, dt, left_flux, right_flux)
    """
    Alternative version. We dont use F as a vector at all here.
    """
    rho = road.rho
    N   = length(rho)
    a   = road.gamma * dt / road.dx

    F_left = left_flux
    @inbounds for i in 1:N
        F_right = (i == N) ? right_flux : godunov_flux(rho[i], rho[i+1])
        rho[i] -= a * (F_right - F_left)
        F_left = F_right
    end
    return nothing
end

end # module Roads
