module Junctions

export
    AbstractJunctionRule,
    Junction,
    TurningFractionRule,
    compute_junction_fluxes,
    update_turning_fractions!

using ..Fluxes: demand, supply

abstract type AbstractJunctionRule end

mutable struct Junction{R<:AbstractJunctionRule}
    incoming::Vector{Int}
    outgoing::Vector{Int}
    junction_rule::R
end

@inline Qscale(road) = road.rho_max * road.speed_limit

junction_state_eltype(j::Junction, roads) =
    promote_type(map(rid -> eltype(roads[rid].rho), vcat(j.incoming, j.outgoing))...)

function validate_turning_matrix(P::AbstractMatrix{<:Real}, nin::Int, nout::Int)
    @assert size(P) == (nin, nout) "Turning matrix must match the junction dimensions."
    @assert all(P .>= -1e-12) "Turning fractions must be nonnegative."
    @assert all(sum(P[i, :]) <= 1 + 1e-10 for i in 1:nin) "Each row of the turning matrix must sum to <= 1."
    return max.(copy(P), zero(eltype(P)))
end

physical_demands(j::Junction, roads) =
    junction_state_eltype(j, roads)[demand(roads[rid]) for rid in j.incoming]

physical_supplies(j::Junction, roads) =
    junction_state_eltype(j, roads)[supply(roads[rid]) for rid in j.outgoing]

incoming_edge_densities(j::Junction, roads) =
    junction_state_eltype(j, roads)[roads[rid].rho[end] for rid in j.incoming]

function movement_flows_to_boundary_fluxes(j::Junction, roads, movement_flows::AbstractMatrix{<:Real})
    nin, nout = size(movement_flows)
    @assert nin == length(j.incoming)
    @assert nout == length(j.outgoing)

    fin_phys = vec(sum(movement_flows; dims=2))
    fout_phys = vec(sum(movement_flows; dims=1))

    fin = similar(fin_phys)
    fout = similar(fout_phys)

    for (k, rid) in enumerate(j.incoming)
        fin[k] = fin_phys[k] / Qscale(roads[rid])
    end
    for (k, rid) in enumerate(j.outgoing)
        fout[k] = fout_phys[k] / Qscale(roads[rid])
    end

    return fin, fout
end

function solve_movement_fluxes(
    desired::AbstractMatrix{<:Real},
    supplies::AbstractVector{<:Real};
    max_iter::Int=50,
    tol::Float64=1e-10,
)
    T = promote_type(typeof(first(desired)), typeof(first(supplies)))
    flows = Matrix{T}(undef, size(desired)...)
    flows .= desired
    nin, nout = size(flows)
    @assert length(supplies) == nout
    zero_T = zero(T)
    one_T = one(T)
    tol_T = tol * one_T

    if sum(flows) == zero_T || sum(supplies) == zero_T
        return zeros(T, nin, nout)
    end

    for _ in 1:max_iter
        outgoing_usage = vec(sum(flows; dims=1))
        ratios = ones(T, nout)

        for o in 1:nout
            if outgoing_usage[o] > zero_T && outgoing_usage[o] > supplies[o]
                ratios[o] = supplies[o] / outgoing_usage[o]
            end
        end

        minimum(ratios) >= one_T - tol_T && break

        row_scale = ones(T, nin)
        for i in 1:nin
            for o in 1:nout
                if flows[i, o] > zero_T
                    row_scale[i] = min(row_scale[i], ratios[o])
                end
            end
        end

        for i in 1:nin
            @views flows[i, :] .*= row_scale[i]
        end
    end

    return flows
end

mutable struct TurningFractionRule <: AbstractJunctionRule
    P::AbstractMatrix{<:Real}
    max_iter::Int
    tol::Float64
end

function TurningFractionRule(P::AbstractMatrix{<:Real}; max_iter::Int=50, tol::Float64=1e-10)
    return TurningFractionRule(copy(P), max_iter, tol)
end

function update_turning_fractions!(rule::TurningFractionRule, Pnew::AbstractMatrix{<:Real})
    @assert size(Pnew) == size(rule.P) "New turning matrix must have the same size as the old one."
    rule.P = copy(Pnew)
    return rule
end

function update_turning_fractions!(j::Junction, Pnew::AbstractMatrix{<:Real})
    return update_turning_fractions!(j.junction_rule, Pnew)
end

function junction_fluxes(rule::TurningFractionRule, j::Junction, roads, t::Real)
    nin = length(j.incoming)
    nout = length(j.outgoing)
    P = validate_turning_matrix(rule.P, nin, nout)
    D = physical_demands(j, roads)
    S = physical_supplies(j, roads)

    desired = reshape(D, :, 1) .* P
    flows = solve_movement_fluxes(desired, S; max_iter=rule.max_iter, tol=rule.tol)
    return movement_flows_to_boundary_fluxes(j, roads, flows)
end

function compute_junction_fluxes(j::Junction, roads, t::Real)
    return junction_fluxes(j.junction_rule, j, roads, t)
end

function compute_junction_fluxes(j::Junction, roads)
    return compute_junction_fluxes(j, roads, 0.0)
end

function (rule::AbstractJunctionRule)(j::Junction, roads, t::Real)
    return junction_fluxes(rule, j, roads, t)
end

end
