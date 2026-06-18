"""
Helpers for turning-fraction inference and observation-vector construction.

The functions here do not prescribe a particular inference algorithm. They
provide the shared parameterization, sensor flattening, weighted residuals, and
posterior predictive summaries used by the experiment scripts.
"""
module InferenceTools

using ..ExperimentUtils: weighted_column_mean

export
    RowSoftmaxTurningParameterization,
    parameter_count,
    parameter_vector,
    stable_row_softmax,
    validate_row_stochastic_matrix,
    validate_row_stochastic_matrices,
    turning_matrix_from_logits,
    turning_matrices_from_logits,
    turning_entries,
    turning_entry_samples,
    entry_vector_to_turning_matrices,
    cell_observation_length,
    flatten_cell_observations,
    flatten_observation_blocks,
    paired_cell_observation_length,
    flatten_paired_cell_observations,
    reshape_cell_observations,
    weighted_residual,
    prediction_ensemble_mean

"""
    RowSoftmaxTurningParameterization(n_matrices, n_rows, n_cols; latent_names=nothing)

Parameterization for one or more row-stochastic turning matrices. Each row uses
`n_cols - 1` unconstrained logits and an implicit zero reference logit.
"""
struct RowSoftmaxTurningParameterization
    n_matrices::Int
    n_rows::Int
    n_cols::Int
    latent_names::Tuple{Vararg{Symbol}}
end

function RowSoftmaxTurningParameterization(
    n_matrices::Integer,
    n_rows::Integer,
    n_cols::Integer;
    latent_names=nothing,
)
    @assert n_matrices >= 1 "Expected at least one turning matrix."
    @assert n_rows >= 1 "Expected at least one row per turning matrix."
    @assert n_cols >= 2 "Expected at least two columns per turning row."

    names = latent_names === nothing ? () : Tuple(latent_names)
    spec = RowSoftmaxTurningParameterization(Int(n_matrices), Int(n_rows), Int(n_cols), names)
    @assert isempty(names) || length(names) == parameter_count(spec) "latent_names must match parameter_count(spec)."
    return spec
end

logits_per_row(spec::RowSoftmaxTurningParameterization) = spec.n_cols - 1

parameter_count(spec::RowSoftmaxTurningParameterization) =
    spec.n_matrices * spec.n_rows * logits_per_row(spec)

"""
    parameter_vector(p, spec; allow_extra=false)

Extract the latent vector used by `spec` from either a vector or a named tuple.
"""
function parameter_vector(
    p::AbstractVector,
    spec::RowSoftmaxTurningParameterization;
    allow_extra=false,
)
    n_params = parameter_count(spec)
    if allow_extra
        @assert length(p) >= n_params "Expected at least $(n_params) parameters."
    else
        @assert length(p) == n_params "Expected $(n_params) parameters."
    end

    return collect(view(p, 1:n_params))
end

function parameter_vector(p::NamedTuple, spec::RowSoftmaxTurningParameterization)
    @assert !isempty(spec.latent_names) "NamedTuple parameters require latent_names in the parameterization."
    return [getfield(p, name) for name in spec.latent_names]
end

"""
    stable_row_softmax(z)

Convert `length(z)` logits plus an implicit zero reference logit into one
row-stochastic vector.
"""
function stable_row_softmax(z::AbstractVector)
    values = collect(z)
    T = promote_type(eltype(values), Float64)
    full_z = Vector{T}(undef, length(values) + 1)
    full_z[1:end-1] .= values
    full_z[end] = zero(T)

    shift = maximum(full_z)
    weights = exp.(full_z .- shift)
    return weights ./ sum(weights)
end

function validate_row_stochastic_matrix(
    P::AbstractMatrix,
    n_rows::Integer,
    n_cols::Integer;
    tol=1e-10,
)
    @assert size(P) == (n_rows, n_cols) "Turning matrix must be $(n_rows)x$(n_cols)."
    @assert all(P .>= -tol) "Turning fractions must be nonnegative."
    @assert maximum(abs.(sum(P; dims=2) .- 1.0)) <= tol "Each row must sum to 1."
    return collect(P)
end

function validate_row_stochastic_matrices(
    Ps::AbstractVector{<:AbstractMatrix},
    spec::RowSoftmaxTurningParameterization;
    tol=1e-10,
)
    @assert length(Ps) == spec.n_matrices "Expected $(spec.n_matrices) turning matrices."
    return [validate_row_stochastic_matrix(P, spec.n_rows, spec.n_cols; tol=tol) for P in Ps]
end

"""
    turning_matrices_from_logits(p, spec; validate=true, allow_extra=false)

Transform latent logits into row-stochastic turning matrices.
"""
function turning_matrices_from_logits(
    p,
    spec::RowSoftmaxTurningParameterization;
    validate=true,
    allow_extra=false,
)
    z = parameter_vector(p, spec; allow_extra=allow_extra)
    T = promote_type(eltype(z), Float64)
    matrices = Matrix{T}[]

    for matrix_idx in 1:spec.n_matrices
        P = Matrix{T}(undef, spec.n_rows, spec.n_cols)
        for row in 1:spec.n_rows
            offset = logits_per_row(spec) * ((matrix_idx - 1) * spec.n_rows + (row - 1))
            P[row, :] = stable_row_softmax(view(z, (offset + 1):(offset + logits_per_row(spec))))
        end
        push!(
            matrices,
            validate ? validate_row_stochastic_matrix(P, spec.n_rows, spec.n_cols) : P,
        )
    end

    return matrices
end

function turning_matrix_from_logits(
    p,
    spec::RowSoftmaxTurningParameterization;
    validate=true,
    allow_extra=false,
)
    @assert spec.n_matrices == 1 "turning_matrix_from_logits expects a one-matrix parameterization."
    return only(turning_matrices_from_logits(p, spec; validate=validate, allow_extra=allow_extra))
end

turning_entries(P::AbstractMatrix) = vec(permutedims(P))

function turning_entries(Ps::AbstractVector{<:AbstractMatrix})
    isempty(Ps) && return Float64[]
    return reduce(vcat, turning_entries.(Ps))
end

"""
    turning_entry_samples(param_samples, transform)

Apply a parameter-to-entry transform to every ensemble column in
`param_samples`, returning one matrix with entries in rows and samples in
columns.
"""
function turning_entry_samples(param_samples::AbstractMatrix, transform::Function)
    n_samples = size(param_samples, 2)
    @assert n_samples >= 1 "Expected at least one parameter sample."

    first_sample = collect(transform(view(param_samples, :, 1)))
    samples = Matrix{eltype(first_sample)}(undef, length(first_sample), n_samples)
    samples[:, 1] = first_sample

    for sample_idx in 2:n_samples
        samples[:, sample_idx] = transform(view(param_samples, :, sample_idx))
    end

    return samples
end

function entry_vector_to_turning_matrices(
    entries::AbstractVector,
    spec::RowSoftmaxTurningParameterization;
    validate=true,
)
    n_entries_per_matrix = spec.n_rows * spec.n_cols
    @assert length(entries) == n_entries_per_matrix * spec.n_matrices

    values = collect(entries)
    T = promote_type(eltype(values), Float64)
    matrices = Matrix{T}[]

    for matrix_idx in 1:spec.n_matrices
        start_idx = n_entries_per_matrix * (matrix_idx - 1) + 1
        P = permutedims(reshape(values[start_idx:(start_idx + n_entries_per_matrix - 1)], spec.n_cols, spec.n_rows))
        push!(
            matrices,
            validate ? validate_row_stochastic_matrix(P, spec.n_rows, spec.n_cols) : Matrix{T}(P),
        )
    end

    return matrices
end

cell_observation_length(observed_road_ids, observed_cell_ids, control_times) =
    length(observed_road_ids) * length(observed_cell_ids) * length(control_times)

"""
    flatten_cell_observations(hist, observed_road_ids, observed_cell_ids, control_times)

Flatten selected road/cell/time density values into the observation vector used
by the inference routines.
"""
function flatten_cell_observations(hist, observed_road_ids, observed_cell_ids, control_times)
    T = eltype(hist.road_histories[first(observed_road_ids)])
    y = Vector{T}(undef, cell_observation_length(observed_road_ids, observed_cell_ids, control_times))
    write_pos = 1

    for road_id in observed_road_ids
        block = hist.road_histories[road_id][observed_cell_ids, :]
        block_vec = vec(block)
        copyto!(y, write_pos, block_vec, 1, length(block_vec))
        write_pos += length(block_vec)
    end

    return y
end

"""
    flatten_observation_blocks(hist, blocks, n_obs)

Flatten heterogeneous sensor blocks. Each block must provide `road_id`,
`sensor_ids`, and `indices` fields.
"""
function flatten_observation_blocks(hist, blocks, n_obs::Integer)
    T = eltype(hist.road_histories[first(blocks).road_id])
    y = Vector{T}(undef, n_obs)
    write_pos = 1

    for block in blocks
        values = vec(hist.road_histories[block.road_id][block.sensor_ids, :])
        copyto!(y, write_pos, values, 1, length(values))
        write_pos += length(values)
    end

    return y
end

paired_cell_observation_length(observed_road_ids, control_times) =
    length(observed_road_ids) * length(control_times)

function flatten_paired_cell_observations(hist, observed_road_ids, observed_cell_ids, control_times)
    @assert length(observed_road_ids) == length(observed_cell_ids)

    T = eltype(hist.road_histories[first(observed_road_ids)])
    y = Vector{T}(undef, paired_cell_observation_length(observed_road_ids, control_times))
    write_pos = 1

    for (idx, road_id) in enumerate(observed_road_ids)
        cell_id = observed_cell_ids[idx]
        values = vec(hist.road_histories[road_id][cell_id, :])
        copyto!(y, write_pos, values, 1, length(values))
        write_pos += length(values)
    end

    return y
end

function reshape_cell_observations(y::AbstractVector, observed_road_ids, observed_cell_ids, control_times)
    n_cells = length(observed_cell_ids)
    n_times = length(control_times)
    n_roads = length(observed_road_ids)
    return reshape(y, n_cells, n_times, n_roads)
end

"""
    weighted_residual(simulator, p, setup, y_obs, sigma_model)

Evaluate a simulator and return observation residuals scaled by the inference
standard deviations.
"""
weighted_residual(simulator::Function, p, setup, y_obs::AbstractVector, sigma_model::AbstractVector) =
    (simulator(p, setup) .- y_obs) ./ sigma_model

"""
    prediction_ensemble_mean(pred_ens, weights; y_obs=nothing, sigma_model=nothing)

Compute a weighted predictive mean. When predictions are stored as normalized
residuals, pass `y_obs` and `sigma_model` to map the mean back to observation
space.
"""
function prediction_ensemble_mean(pred_ens, weights::AbstractVector; y_obs=nothing, sigma_model=nothing)
    pred_matrix = reshape(pred_ens, size(pred_ens, 1), :)
    @assert size(pred_matrix, 2) == length(weights)
    mean_vector = weighted_column_mean(pred_matrix, weights)

    if y_obs === nothing
        return mean_vector
    end

    @assert sigma_model !== nothing "sigma_model is required when y_obs is supplied."
    return y_obs .+ sigma_model .* mean_vector
end

end # module InferenceTools
