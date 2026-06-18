"""
Statistical utilities for experiment diagnostics, noisy observations, and
tabular output.

These functions are deliberately algorithm-agnostic: they summarize weighted
ensembles, generate synthetic noisy data, evaluate recovery quality, and write
small result tables used by the thesis experiments.
"""
module ExperimentUtils

using Printf
using Random: AbstractRNG, randn
using Statistics: mean

export
    normalize_weights,
    weighted_quantile,
    weighted_column_mean,
    sample_summary,
    weighted_std,
    trapezoid_integral,
    normalize_importance_weights,
    gaussian_kernel1d,
    smooth_vector_clamped,
    smooth_matrix_clamped,
    weighted_kde_unit_interval,
    weighted_histogram_1d,
    weighted_histogram_2d,
    rmse,
    normalized_rmse,
    interval_coverage_mask,
    interval_coverage_rate,
    mean_interval_width,
    recovery_summary,
    simulate_noisy_observations,
    parabolic_noise_shape,
    physical_noise_sigma,
    inference_sigma_from_observation,
    generate_physical_observations,
    table_value_string,
    write_namedtuple_table,
    write_config_file

"""
    normalize_weights(weights)

Return a floating-point copy of `weights` scaled to sum to one.
"""
function normalize_weights(weights::AbstractVector)
    normalized = Float64.(collect(weights))
    normalized ./= sum(normalized)
    return normalized
end

"""
    weighted_quantile(xs, weights, q)

Weighted empirical quantile for ensemble summaries and credible intervals.
"""
function weighted_quantile(xs::AbstractVector, ws::AbstractVector, q::Real)
    @assert 0.0 <= q <= 1.0

    order = sortperm(xs)
    xs_sorted = xs[order]
    ws_sorted = normalize_weights(ws[order])
    cdf = cumsum(ws_sorted)
    idx = min(searchsortedfirst(cdf, q), length(xs_sorted))
    return xs_sorted[idx]
end

function weighted_column_mean(samples::AbstractMatrix, weights::AbstractVector)
    @assert size(samples, 2) == length(weights)
    return vec(sum(samples .* weights', dims=2))
end

"""
    sample_summary(samples, weights)

Return weighted means, medians, and 5/95 percentiles for samples stored as
parameters-by-ensemble columns.
"""
function sample_summary(samples::AbstractMatrix, weights::AbstractVector)
    means = weighted_column_mean(samples, weights)
    medians = [weighted_quantile(vec(samples[i, :]), weights, 0.5) for i in 1:size(samples, 1)]
    ci_05 = [weighted_quantile(vec(samples[i, :]), weights, 0.05) for i in 1:size(samples, 1)]
    ci_95 = [weighted_quantile(vec(samples[i, :]), weights, 0.95) for i in 1:size(samples, 1)]
    return means, medians, ci_05, ci_95
end

function weighted_std(xs::AbstractVector, ws::AbstractVector)
    weights = normalize_weights(ws)
    mu = sum(xs .* weights)
    return sqrt(sum(weights .* (xs .- mu) .^ 2))
end

function trapezoid_integral(xs::AbstractVector, ys::AbstractVector)
    @assert length(xs) == length(ys)
    area = 0.0

    for idx in 1:(length(xs) - 1)
        area += 0.5 * (ys[idx] + ys[idx + 1]) * (xs[idx + 1] - xs[idx])
    end

    return area
end

"""
    normalize_importance_weights(raw_weights)

Normalize either nonnegative weights or log weights into probability weights.
"""
function normalize_importance_weights(raw_weights::AbstractVector)
    values = Float64.(collect(raw_weights))

    if all(isfinite, values) && all(>=(0.0), values) && sum(values) > 0.0
        values ./= sum(values)
        return values
    end

    max_log_weight = maximum(values)
    weights = exp.(values .- max_log_weight)
    weights ./= sum(weights)
    return weights
end

function gaussian_kernel1d(sigma_bins::Real)
    radius = max(1, ceil(Int, 3 * sigma_bins))
    offsets = collect(-radius:radius)
    kernel = exp.(-0.5 .* (offsets ./ sigma_bins) .^ 2)
    kernel ./= sum(kernel)
    return offsets, kernel
end

function smooth_vector_clamped(values::AbstractVector, offsets::AbstractVector{Int}, kernel::AbstractVector)
    smoothed = zeros(Float64, length(values))

    for i in eachindex(values)
        acc = 0.0
        for (offset, weight) in zip(offsets, kernel)
            j = clamp(i + offset, firstindex(values), lastindex(values))
            acc += weight * values[j]
        end
        smoothed[i] = acc
    end

    return smoothed
end

function smooth_matrix_clamped(values::AbstractMatrix, offsets::AbstractVector{Int}, kernel::AbstractVector)
    tmp = zeros(Float64, size(values))
    smoothed = similar(tmp)

    for j in axes(values, 2)
        tmp[:, j] = smooth_vector_clamped(view(values, :, j), offsets, kernel)
    end

    for i in axes(values, 1)
        smoothed[i, :] = smooth_vector_clamped(view(tmp, i, :), offsets, kernel)
    end

    return smoothed
end

function weighted_kde_unit_interval(
    grid::AbstractVector,
    samples::AbstractVector,
    weights::AbstractVector;
    bandwidth=nothing,
)
    weights_norm = normalize_weights(weights)
    n_eff = 1 / sum(weights_norm .^ 2)

    h = if bandwidth === nothing
        sd = weighted_std(samples, weights_norm)
        q25 = weighted_quantile(samples, weights_norm, 0.25)
        q75 = weighted_quantile(samples, weights_norm, 0.75)
        iqr_scale = (q75 - q25) / 1.34
        scale = min(sd, iqr_scale)
        if !isfinite(scale) || scale <= 0.0
            scale = max(sd, 0.02)
        end
        clamp(1.06 * scale * n_eff^(-1 / 5), 0.005, 0.08)
    else
        bandwidth
    end

    density = zeros(Float64, length(grid))
    normalizer = inv(sqrt(2pi) * h)

    for (sample, weight) in zip(samples, weights_norm)
        density .+= weight .* normalizer .* (
            exp.(-0.5 .* ((grid .- sample) ./ h) .^ 2) .+
            exp.(-0.5 .* ((grid .+ sample) ./ h) .^ 2) .+
            exp.(-0.5 .* ((grid .- (2 - sample)) ./ h) .^ 2)
        )
    end

    area = trapezoid_integral(grid, density)
    if isfinite(area) && area > 0.0
        density ./= area
    end

    return density
end

function weighted_histogram_1d(samples::AbstractVector, weights::AbstractVector; bins=90, smooth_sigma_bins=1.4)
    edges = collect(range(0.0, 1.0; length=bins + 1))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    bin_width = edges[2] - edges[1]
    hist = zeros(Float64, bins)

    for (x, weight) in zip(samples, weights)
        idx = clamp(floor(Int, x * bins) + 1, 1, bins)
        hist[idx] += weight
    end

    offsets, kernel = gaussian_kernel1d(smooth_sigma_bins)
    density = smooth_vector_clamped(hist, offsets, kernel) ./ bin_width
    area = trapezoid_integral(centers, density)
    if isfinite(area) && area > 0.0
        density ./= area
    end

    return centers, density
end

function weighted_histogram_2d(
    particles::AbstractMatrix,
    weights::AbstractVector;
    bins=80,
    smooth_sigma_bins=1.25,
)
    @assert size(particles, 2) == 2
    @assert size(particles, 1) == length(weights)

    edges = collect(range(0.0, 1.0; length=bins + 1))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    bin_width = edges[2] - edges[1]
    hist = zeros(Float64, bins, bins)

    for j in axes(particles, 1)
        ix = clamp(floor(Int, particles[j, 1] * bins) + 1, 1, bins)
        iy = clamp(floor(Int, particles[j, 2] * bins) + 1, 1, bins)
        hist[ix, iy] += weights[j]
    end

    offsets, kernel = gaussian_kernel1d(smooth_sigma_bins)
    density = smooth_matrix_clamped(hist, offsets, kernel) ./ bin_width^2
    mass = sum(density) * bin_width^2
    if isfinite(mass) && mass > 0.0
        density ./= mass
    end

    return centers, density
end

"""
    rmse(estimate, truth)

Root mean squared error with broadcasting over the supplied arrays.
"""
rmse(estimate, truth) = sqrt(mean((estimate .- truth) .^ 2))

function normalized_rmse(estimate, truth, sigma::AbstractVector)
    @assert length(estimate) == length(truth) == length(sigma)
    return rmse((estimate .- truth) ./ sigma, zeros(length(sigma)))
end

function interval_coverage_mask(truth, lower, upper)
    @assert length(truth) == length(lower) == length(upper)
    return (truth .>= lower) .& (truth .<= upper)
end

interval_coverage_rate(truth, lower, upper) = mean(interval_coverage_mask(truth, lower, upper))

function mean_interval_width(lower, upper)
    @assert length(lower) == length(upper)
    return mean(upper .- lower)
end

"""
    recovery_summary(estimate, truth; lower=nothing, upper=nothing)

Compact recovery diagnostics. With interval bounds, also reports mean interval
width and coverage.
"""
function recovery_summary(estimate, truth; lower=nothing, upper=nothing)
    error_values = estimate .- truth
    base = (
        rmse=rmse(estimate, truth),
        mean_abs=mean(abs.(error_values)),
        max_abs=maximum(abs.(error_values)),
    )

    if lower === nothing || upper === nothing
        return base
    end

    coverage = interval_coverage_mask(truth, lower, upper)
    return merge(
        base,
        (
            mean_interval_width=mean_interval_width(lower, upper),
            interval_coverage_count=sum(coverage),
            interval_coverage_rate=mean(coverage),
        ),
    )
end

"""
    parabolic_noise_shape(rho)

Density-dependent noise envelope with zero variance at empty/full density and a
peak near `rho = 0.5`.
"""
parabolic_noise_shape(rho::Real) =
    4.0 * clamp(Float64(rho), 0.0, 1.0) * (1.0 - clamp(Float64(rho), 0.0, 1.0))

"""
    physical_noise_sigma(y, peak_noise_sigma)

Physical synthetic-noise standard deviation for each true observation.
"""
function physical_noise_sigma(y::AbstractVector, peak_noise_sigma::Real)
    peak = Float64(peak_noise_sigma)
    return peak .* parabolic_noise_shape.(y)
end

"""
    inference_sigma_from_observation(y_obs, peak_noise_sigma; floor_fraction=0.03,
                                     absolute_floor=1e-3)

Standard deviations used by the inference likelihood. The floor prevents nearly
zero densities from receiving infinite weight.
"""
function inference_sigma_from_observation(
    y_obs::AbstractVector,
    peak_noise_sigma::Real;
    floor_fraction=0.03,
    absolute_floor=1e-3,
)
    peak = Float64(peak_noise_sigma)
    floor_value = max(absolute_floor, floor_fraction * peak)
    sigma = physical_noise_sigma(y_obs, peak)
    return max.(sigma, floor_value)
end

"""
    generate_physical_observations(y_true, peak_noise_sigma, rng; kwargs...)

Generate clipped synthetic observations, physical noise scales, and inference
noise scales from a noiseless truth vector.
"""
function generate_physical_observations(
    y_true::AbstractVector,
    peak_noise_sigma::Real,
    rng::AbstractRNG;
    floor_fraction=0.03,
    absolute_floor=1e-3,
)
    sigma_true = physical_noise_sigma(y_true, peak_noise_sigma)
    y_raw = y_true .+ sigma_true .* randn(rng, length(y_true))
    y_obs = clamp.(y_raw, 0.0, 1.0)
    sigma_model = inference_sigma_from_observation(
        y_obs,
        peak_noise_sigma;
        floor_fraction=floor_fraction,
        absolute_floor=absolute_floor,
    )

    return (
        y_true=Float64.(y_true),
        y_obs=Float64.(y_obs),
        sigma_true=Float64.(sigma_true),
        sigma_model=Float64.(sigma_model),
        clip_fraction=mean((y_raw .< 0.0) .| (y_raw .> 1.0)),
    )
end

"""
    simulate_noisy_observations(simulator, parameters, setup, peak_noise_sigma, rng; kwargs...)

Convenience wrapper that runs `simulator(parameters, setup)` and then calls
`generate_physical_observations`.
"""
function simulate_noisy_observations(
    simulator::Function,
    parameters,
    setup,
    peak_noise_sigma::Real,
    rng::AbstractRNG;
    kwargs...
)
    return generate_physical_observations(
        simulator(parameters, setup),
        peak_noise_sigma,
        rng;
        kwargs...,
    )
end

table_value_string(value::AbstractFloat) = @sprintf("%.10g", value)
table_value_string(value::Integer) = string(value)
table_value_string(value::Bool) = value ? "true" : "false"
table_value_string(value::AbstractVector) = join(table_value_string.(value), ",")
table_value_string(value) = string(value)

"""
    write_namedtuple_table(rows, output_path)

Write a vector of named tuples as a tab-separated table.
"""
function write_namedtuple_table(rows, output_path)
    isempty(rows) && return nothing

    header = collect(keys(first(rows)))
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(io, join(string.(header), '\t'))
        for row in rows
            values = [table_value_string(getproperty(row, key)) for key in header]
            println(io, join(values, '\t'))
        end
    end

    return output_path
end

function write_config_file(output_path, lines)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    return output_path
end

end # module ExperimentUtils
