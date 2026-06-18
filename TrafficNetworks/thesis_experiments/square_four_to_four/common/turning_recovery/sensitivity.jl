# Shared helpers for square-four-to-four turning-outlier diagnostics.

function shift_turning_row(row_values::AbstractVector, target_col::Int, delta::Real)
    row = Float64.(collect(row_values))
    others = [col for col in 1:length(row) if col != target_col]
    amount = abs(Float64(delta))

    if amount == 0.0
        return row
    elseif delta > 0
        available = sum(row[others])
        actual = min(amount, 0.5 * available)
        actual == 0.0 && return row
        original_others = copy(row[others])
        row[target_col] += actual
        row[others] .-= actual .* original_others ./ sum(original_others)
    else
        available = row[target_col]
        actual = min(amount, 0.5 * available)
        actual == 0.0 && return row
        original_others = copy(row[others])
        row[target_col] -= actual
        row[others] .+= actual .* original_others ./ sum(original_others)
    end

    return row ./ sum(row)
end

function perturb_turning_entry(P_true, junction::Int, incoming_row::Int, outgoing_col::Int, delta::Real)
    matrices = [copy(Matrix{Float64}(matrix)) for matrix in turning_matrices(P_true)]
    matrices[junction][incoming_row, :] = shift_turning_row(matrices[junction][incoming_row, :], outgoing_col, delta)
    return turning_matrices(matrices)
end

function replace_turning_row(P_base, junction::Int, incoming_row::Int, row_values::AbstractVector)
    row = Float64.(collect(row_values))
    @assert length(row) == 4 "A square-network turning row must have four entries."
    @assert all(row .>= -1e-12) "Turning rows must be nonnegative."
    @assert isapprox(sum(row), 1.0; atol=1e-8, rtol=1e-8) "Turning rows must sum to one."

    matrices = [copy(Matrix{Float64}(matrix)) for matrix in turning_matrices(P_base)]
    matrices[junction][incoming_row, :] = max.(row, 0.0) ./ sum(max.(row, 0.0))
    return turning_matrices(matrices)
end

function perturb_turning_pair(P_true, junction::Int, incoming_row::Int, plus_col::Int, minus_col::Int, delta::Real)
    @assert plus_col != minus_col
    matrices = [copy(Matrix{Float64}(matrix)) for matrix in turning_matrices(P_true)]
    row = copy(matrices[junction][incoming_row, :])
    step = min(Float64(delta), 0.5 * row[minus_col])
    row[plus_col] += step
    row[minus_col] -= step
    matrices[junction][incoming_row, :] = row ./ sum(row)
    return turning_matrices(matrices)
end

function turning_entry_sensitivity_rows(
    metadata_rows,
    P_true,
    dataset::MultiScenarioDataset,
    sigma_model::AbstractVector;
    step=1e-3,
)
    rows = NamedTuple[]

    for meta in metadata_rows
        plus = perturb_turning_entry(P_true, meta.junction, meta.incoming_row, meta.outgoing_col, step)
        minus = perturb_turning_entry(P_true, meta.junction, meta.incoming_row, meta.outgoing_col, -step)
        y_plus = simulator_dataset(plus, dataset)
        y_minus = simulator_dataset(minus, dataset)
        derivative = (y_plus .- y_minus) ./ (2.0 * Float64(step))

        push!(
            rows,
            (
                global_entry=meta.global_entry,
                raw_sensitivity=sqrt(mean(derivative .^ 2)),
                normalized_sensitivity=sqrt(mean((derivative ./ sigma_model) .^ 2)),
                max_abs_normalized_sensitivity=maximum(abs.(derivative ./ sigma_model)),
            ),
        )
    end

    return rows
end

function pair_contrast_sensitivity(
    P_true,
    dataset::MultiScenarioDataset,
    sigma_model::AbstractVector,
    junction::Int,
    incoming_row::Int,
    col_a::Int,
    col_b::Int;
    step=1e-3,
)
    plus = perturb_turning_pair(P_true, junction, incoming_row, col_a, col_b, step)
    minus = perturb_turning_pair(P_true, junction, incoming_row, col_b, col_a, step)
    derivative = (simulator_dataset(plus, dataset) .- simulator_dataset(minus, dataset)) ./ (2.0 * Float64(step))
    return sqrt(mean((derivative ./ sigma_model) .^ 2))
end

function finite_correlation(xs::AbstractVector, ys::AbstractVector)
    pairs = [(Float64(x), Float64(y)) for (x, y) in zip(xs, ys) if isfinite(x) && isfinite(y)]
    length(pairs) < 2 && return NaN
    xvals = first.(pairs)
    yvals = last.(pairs)
    std(xvals) == 0.0 || std(yvals) == 0.0 ? NaN : cor(xvals, yvals)
end

rmse_values(x::AbstractVector, y::AbstractVector) = sqrt(mean((Float64.(x) .- Float64.(y)) .^ 2))
normalized_rmse_values(x::AbstractVector, y::AbstractVector, sigma::AbstractVector) =
    sqrt(mean(((Float64.(x) .- Float64.(y)) ./ Float64.(sigma)) .^ 2))

