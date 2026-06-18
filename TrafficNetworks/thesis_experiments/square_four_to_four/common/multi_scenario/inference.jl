# Shared helpers for square-four-to-four multi-scenario experiments.

function map_loss_dataset_weighted_forwarddiff(
    z,
    y_obs::AbstractVector,
    dataset::MultiScenarioDataset,
    sigma_model::AbstractVector;
    prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
)
    residual = weighted_residual_dataset_forwarddiff(z, dataset, y_obs, sigma_model)
    prior_term = z ./ prior_scale
    return 0.5 * sum(abs2, residual) + 0.5 * sum(abs2, prior_term)
end

function run_adam_map_multi_scenario(
    y_obs::AbstractVector,
    dataset::MultiScenarioDataset,
    sigma_model::AbstractVector;
    prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
    z0=zeros(N_PARAMS),
    maxiters=360,
)
    loss_fn = z -> map_loss_dataset_weighted_forwarddiff(z, y_obs, dataset, sigma_model; prior_scale=prior_scale)
    result = run_adam_forwarddiff(
        loss_fn;
        z0=z0,
        learning_rate=MULTI_SCENARIO_ADAM_LEARNING_RATE,
        maxiters=maxiters,
        grad_clip=MULTI_SCENARIO_ADAM_GRAD_CLIP,
        decay_start=MULTI_SCENARIO_ADAM_DECAY_START,
        final_lr_scale=MULTI_SCENARIO_ADAM_FINAL_LR_SCALE,
    )

    return (
        result...,
        P_est=turning_matrices(result.z_best),
        y_est=simulator_dataset(result.z_best, dataset),
    )
end

function simulate_ensemble_dataset(param_ensemble::AbstractMatrix, dataset::MultiScenarioDataset)
    n_obs = dataset_observation_length(dataset)
    ensemble_size = size(param_ensemble, 2)
    predictions = Matrix{Float64}(undef, n_obs, ensemble_size)

    for member in 1:ensemble_size
        predictions[:, member] = simulator_dataset(view(param_ensemble, :, member), dataset)
    end

    return predictions
end

function ensemble_mean_dataset_prediction(param_ensemble::AbstractMatrix, weights::AbstractVector, dataset::MultiScenarioDataset)
    predictions = simulate_ensemble_dataset(param_ensemble, dataset)
    weights_norm = normalize_weights(weights)
    return vec(predictions * weights_norm)
end

function ensemble_space_esmda_update(
    param_ensemble::AbstractMatrix,
    pred_ensemble::AbstractMatrix,
    y_obs::AbstractVector,
    sigma_model::AbstractVector,
    alpha::Real,
    rng::AbstractRNG,
)
    ensemble_size = size(param_ensemble, 2)
    param_mean = mean(param_ensemble; dims=2)
    pred_mean = mean(pred_ensemble; dims=2)

    param_anom = param_ensemble .- param_mean
    pred_anom = pred_ensemble .- pred_mean

    sqrt_inv_r = 1.0 ./ (sqrt(Float64(alpha)) .* sigma_model)
    pred_whitened = pred_anom .* sqrt_inv_r

    perturbed_obs = y_obs .+ sqrt(Float64(alpha)) .* sigma_model .* randn(rng, length(y_obs), ensemble_size)
    innovation_whitened = (perturbed_obs .- pred_ensemble) .* sqrt_inv_r

    system_matrix = pred_whitened' * pred_whitened + (ensemble_size - 1) * I
    weights = system_matrix \ (pred_whitened' * innovation_whitened)

    return param_ensemble + param_anom * weights
end

function esmda_alpha_schedule(maxiters::Integer)
    @assert maxiters >= 1 "ESMDA requires at least one assimilation iteration."
    schedule = fill(Float64(maxiters), Int(maxiters))
    @assert isapprox(sum(1.0 ./ schedule), 1.0; atol=1e-12, rtol=1e-12)
    return schedule
end

function inflate_ensemble_covariance(param_ensemble::AbstractMatrix, covariance_inflation::Real)
    inflation = Float64(covariance_inflation)
    @assert inflation >= 1.0 "Covariance inflation factor must be at least 1."
    inflation == 1.0 && return param_ensemble

    ensemble_mean = mean(param_ensemble; dims=2)
    return ensemble_mean .+ sqrt(inflation) .* (param_ensemble .- ensemble_mean)
end

function run_esmda_multi_scenario(
    dataset::MultiScenarioDataset,
    y_obs::AbstractVector,
    sigma_model::AbstractVector;
    seed=1,
    prior_scale=MULTI_SCENARIO_PRIOR_SCALE,
    ensemble_size=MULTI_SCENARIO_ESMDA_ENSEMBLE_SIZE,
    esmda_maxiters=MULTI_SCENARIO_ESMDA_STANDARD_MAXITERS,
    covariance_inflation=MULTI_SCENARIO_ESMDA_COVARIANCE_INFLATION,
    alpha_schedule=esmda_alpha_schedule(esmda_maxiters),
    P_true=true_turning_matrices(),
)
    rng = MersenneTwister(seed)
    param_ensemble = prior_scale .* randn(rng, N_PARAMS, ensemble_size)
    alphas = Float64.(collect(alpha_schedule))
    @assert length(alphas) == esmda_maxiters "Alpha schedule length must match esmda_maxiters."
    @assert isapprox(sum(1.0 ./ alphas), 1.0; atol=1e-12, rtol=1e-12) "ESMDA alpha schedule must satisfy sum(1 / alpha_n) = 1."

    solve_seconds = @elapsed begin
        for iter in 1:esmda_maxiters
            pred_ensemble = simulate_ensemble_dataset(param_ensemble, dataset)
            param_ensemble = ensemble_space_esmda_update(param_ensemble, pred_ensemble, y_obs, sigma_model, alphas[iter], rng)
            if iter < esmda_maxiters
                param_ensemble = inflate_ensemble_covariance(param_ensemble, covariance_inflation)
            end
        end
    end

    weights = fill(1.0 / ensemble_size, ensemble_size)
    fraction_samples = turning_entry_samples(param_ensemble)
    entry_true = turning_entries(P_true)
    entry_post_mean, entry_post_median, entry_ci_05, entry_ci_95 = sample_summary(fraction_samples, weights)
    P_post_mean = entry_vector_to_turning_matrices(entry_post_mean)
    y_mean_parameter = simulator_dataset(P_post_mean, dataset)
    y_post_mean = ensemble_mean_dataset_prediction(param_ensemble, weights, dataset)

    return (
        weights=weights,
        param_samples=param_ensemble,
        fraction_samples=fraction_samples,
        entry_true=entry_true,
        entry_post_mean=entry_post_mean,
        entry_post_median=entry_post_median,
        entry_ci_05=entry_ci_05,
        entry_ci_95=entry_ci_95,
        P_post_mean=P_post_mean,
        y_mean_parameter=y_mean_parameter,
        y_post_mean=y_post_mean,
        solve_seconds=solve_seconds,
        ensemble_size=ensemble_size,
        esmda_maxiters=esmda_maxiters,
        covariance_inflation=Float64(covariance_inflation),
        alpha_schedule=alphas,
    )
end

