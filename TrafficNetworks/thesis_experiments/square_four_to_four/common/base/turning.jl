function true_turning_matrices()
    return [
        [0.10 0.12 0.50 0.28; 0.08 0.18 0.22 0.52; 0.42 0.24 0.08 0.26; 0.20 0.46 0.24 0.10],
        [0.18 0.08 0.16 0.58; 0.14 0.10 0.56 0.20; 0.48 0.20 0.08 0.24; 0.18 0.42 0.30 0.10],
        [0.16 0.18 0.42 0.24; 0.10 0.22 0.18 0.50; 0.44 0.18 0.10 0.28; 0.22 0.40 0.26 0.12],
        [0.12 0.16 0.54 0.18; 0.16 0.10 0.20 0.54; 0.46 0.24 0.10 0.20; 0.20 0.48 0.22 0.10],
    ]
end

function parameter_vector(p::AbstractVector)
    return Float64.(TrafficNetworks.parameter_vector(p, SQUARE_TURNING_PARAMETERIZATION))
end

parameter_vector(p::NamedTuple) = TrafficNetworks.parameter_vector(p, SQUARE_TURNING_PARAMETERIZATION)

stable_row_softmax(z::AbstractVector) = TrafficNetworks.stable_row_softmax(z)

function validate_turning_matrix(P::AbstractMatrix)
    return Float64.(TrafficNetworks.validate_row_stochastic_matrix(P, 4, 4))
end

function validate_turning_matrices(Ps::AbstractVector{<:AbstractMatrix})
    @assert length(Ps) == N_JUNCTIONS "Expected $(N_JUNCTIONS) turning matrices."
    return [validate_turning_matrix(P) for P in Ps]
end

function turning_matrices(p::AbstractVector)
    return TrafficNetworks.turning_matrices_from_logits(p, SQUARE_TURNING_PARAMETERIZATION)
end

turning_matrices(p::NamedTuple) = turning_matrices(parameter_vector(p))
turning_matrices(Ps::AbstractVector{<:AbstractMatrix}) = validate_turning_matrices(Ps)

function turning_entries(Ps)
    matrices = turning_matrices(Ps)
    return TrafficNetworks.turning_entries(matrices)
end

function turning_entry_samples(param_samples::AbstractMatrix)
    return TrafficNetworks.turning_entry_samples(param_samples, sample -> turning_entries(sample))
end

function entry_vector_to_turning_matrices(entries::AbstractVector; validate=true)
    return TrafficNetworks.entry_vector_to_turning_matrices(
        entries,
        SQUARE_TURNING_PARAMETERIZATION;
        validate=validate,
    )
end
