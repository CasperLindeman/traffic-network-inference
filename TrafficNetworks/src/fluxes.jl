module Fluxes

export flux, dflux, godunov_flux, demand, supply, maxwavespeed, endpoint_wavespeed

@inline flux(rho) = rho * (1.0 - rho)

@inline dflux(rho) = 1.0 - 2.0 * rho

@inline function godunov_flux(rho_L, rho_R)
    rho_c = 0.5
    f_max = 0.25

    demand = rho_L <= rho_c ? flux(rho_L) : f_max
    supply = rho_R <= rho_c ? f_max : flux(rho_R)

    return min(demand, supply)
end

@inline function demand(road)
    rho_R = road.rho[end]
    qmax = 0.25 * road.rho_max * road.speed_limit # Maximal allowed outflow flux

    if rho_R <= 0.5
        return flux(rho_R) * road.rho_max * road.speed_limit
    else
        return qmax
    end
end

@inline function supply(road)
    rho_L = road.rho[1]
    qmax = 0.25 * road.rho_max * road.speed_limit

    if rho_L <= 0.5
        return qmax
    else
        return flux(rho_L) * road.rho_max * road.speed_limit
    end
end

@inline maxwavespeed(rho) = abs(dflux(rho))

@inline endpoint_wavespeed(F) = sqrt(max(one(F) - 4 * F, zero(F)))

end # module Fluxes
