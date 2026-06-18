using Random
using Printf
using Statistics
using Distributions
using SimulationBasedInference
using TrafficNetworks
using DynamicHMC
using ForwardDiff
import TrafficNetworks: ExperimentNetworkSpec, load_experiment_network_spec
import TrafficNetworks: physical_noise_sigma, inference_sigma_from_observation, generate_physical_observations
import TrafficNetworks: weighted_quantile, weighted_column_mean, sample_summary

import SimulationBasedInference.LogDensityProblems: logdensity, logdensity_and_gradient, capabilities, dimension, LogDensityOrder

struct FourToFourInferenceSetup
    T::Float64
    CFL::Float64
    n_cells::Int
    control_times::Vector{Float64}
    observed_road_ids::Vector{Int}
    observed_cell_ids::Vector{Int}
    inflows::Vector{Function}
    road_profiles::Vector{Function}
    road_lengths::Vector{Float64}
    speed_limits::Vector{Int}
    network_spec::ExperimentNetworkSpec
end

struct MethodTrial
    method::Symbol
    kwargs::NamedTuple
end

const LATENT_NAMES = (
    :z11, :z12, :z13,
    :z21, :z22, :z23,
    :z31, :z32, :z33,
    :z41, :z42, :z43,
)

const METHOD_TIME_BUDGET_SECONDS = 60.0
const SINGLE_FOUR_TO_FOUR_NETWORK_SPEC_PATH = joinpath(@__DIR__, "..", "..", "network_specs", "single_four_to_four.toml")
const SINGLE_FOUR_TO_FOUR_NETWORK_SPEC = load_experiment_network_spec(SINGLE_FOUR_TO_FOUR_NETWORK_SPEC_PATH)
const FOUR_TO_FOUR_TURNING_PARAMETERIZATION =
    TrafficNetworks.RowSoftmaxTurningParameterization(1, 4, 4; latent_names=LATENT_NAMES)

function default_setup()
    return physical_mixed_speed_setup()
end

function physical_mixed_speed_setup(; network_spec=SINGLE_FOUR_TO_FOUR_NETWORK_SPEC)
    return FourToFourInferenceSetup(
        network_spec.T,
        network_spec.CFL,
        network_spec.n_cells::Int,
        copy(network_spec.control_times),
        TrafficNetworks.observed_road_ids(network_spec),
        TrafficNetworks.observed_cell_ids(network_spec),
        TrafficNetworks.boundary_inflows(network_spec),
        TrafficNetworks.road_profiles(network_spec),
        TrafficNetworks.road_lengths(network_spec),
        TrafficNetworks.speed_limits(network_spec),
        network_spec,
    )
end

function true_turning_matrix()
    return [
        0.52 0.20 0.16 0.12
        0.15 0.47 0.23 0.15
        0.14 0.18 0.46 0.22
        0.18 0.16 0.21 0.45
    ]
end

function parameter_vector(p::AbstractVector)
    return TrafficNetworks.parameter_vector(p, FOUR_TO_FOUR_TURNING_PARAMETERIZATION; allow_extra=true)
end

function parameter_vector(p::NamedTuple)
    return TrafficNetworks.parameter_vector(p, FOUR_TO_FOUR_TURNING_PARAMETERIZATION)
end

stable_row_softmax(z::AbstractVector) = TrafficNetworks.stable_row_softmax(z)

function validate_turning_matrix(P::AbstractMatrix)
    return TrafficNetworks.validate_row_stochastic_matrix(P, 4, 4)
end

function turning_matrix(p::AbstractMatrix)
    return validate_turning_matrix(p)
end

function turning_matrix(p)
    return TrafficNetworks.turning_matrix_from_logits(
        p,
        FOUR_TO_FOUR_TURNING_PARAMETERIZATION;
        allow_extra=true,
    )
end

function turning_entries(P::AbstractMatrix)
    return TrafficNetworks.turning_entries(P)
end

function turning_entry_samples(param_samples::AbstractMatrix)
    return TrafficNetworks.turning_entry_samples(
        param_samples,
        sample -> turning_entries(turning_matrix(sample)),
    )
end

function build_four_to_four_network(p, setup::FourToFourInferenceSetup)
    return TrafficNetworks.build_experiment_network(
        setup.network_spec,
        [TurningFractionRule(turning_matrix(p))];
        T=setup.T,
        CFL=setup.CFL,
        road_length_values=setup.road_lengths,
        road_profile_values=setup.road_profiles,
        speed_limit_values=setup.speed_limits,
        boundary_inflow_values=setup.inflows,
    )
end

observation_length(setup::FourToFourInferenceSetup) =
    TrafficNetworks.cell_observation_length(
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )

function flatten_observations(hist::SimulationHistory, setup::FourToFourInferenceSetup)
    return TrafficNetworks.flatten_cell_observations(
        hist,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end

function simulator(p, setup::FourToFourInferenceSetup)
    net = build_four_to_four_network(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return flatten_observations(hist, setup)
end

parameter_vector_forwarddiff(p::AbstractVector) =
    TrafficNetworks.parameter_vector(p, FOUR_TO_FOUR_TURNING_PARAMETERIZATION; allow_extra=true)
parameter_vector_forwarddiff(p::NamedTuple) =
    TrafficNetworks.parameter_vector(p, FOUR_TO_FOUR_TURNING_PARAMETERIZATION)

function turning_matrix_forwarddiff(p)
    return TrafficNetworks.turning_matrix_from_logits(
        p,
        FOUR_TO_FOUR_TURNING_PARAMETERIZATION;
        validate=false,
        allow_extra=true,
    )
end

four_to_four_state_eltype(p) = promote_type(eltype(parameter_vector_forwarddiff(p)), Float64)

function build_four_to_four_network_forwarddiff(p, setup::FourToFourInferenceSetup)
    state_T = four_to_four_state_eltype(p)
    return TrafficNetworks.build_experiment_network(
        setup.network_spec,
        [TurningFractionRule(turning_matrix_forwarddiff(p))];
        T=setup.T,
        CFL=setup.CFL,
        road_length_values=setup.road_lengths,
        road_profile_values=setup.road_profiles,
        speed_limit_values=setup.speed_limits,
        boundary_inflow_values=setup.inflows,
        state_eltype=state_T,
    )
end

function simulate_history_forwarddiff(p, setup::FourToFourInferenceSetup)
    net = build_four_to_four_network_forwarddiff(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return hist
end

function flatten_observations_forwarddiff(hist::SimulationHistory, setup::FourToFourInferenceSetup)
    return TrafficNetworks.flatten_cell_observations(
        hist,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end

simulator_forwarddiff(p, setup::FourToFourInferenceSetup) =
    flatten_observations_forwarddiff(simulate_history_forwarddiff(p, setup), setup)

function reshape_observations(y::AbstractVector, setup::FourToFourInferenceSetup)
    return TrafficNetworks.reshape_cell_observations(
        y,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end

function posterior_predictive_mean(sol, weights, y_obs::AbstractVector, sigma_model::AbstractVector)
    pred_ens = if sol.alg isa SimulationBasedInference.EnsembleInferenceAlgorithm
        Array(get_observables(sol).y)
    else
        reduce(hcat, map(out -> out.y, getoutputs(sol)))
    end
    return TrafficNetworks.prediction_ensemble_mean(
        pred_ens,
        weights;
        y_obs=y_obs,
        sigma_model=sigma_model,
    )
end

function model_parameter_vector(p)
    if hasproperty(p, :model)
        return model_parameter_vector(getproperty(p, :model))
    elseif p isa NamedTuple
        return [getfield(p, name) for name in LATENT_NAMES]
    else
        return parameter_vector(p)
    end
end

function simulate_posterior_draws(param_samples::AbstractMatrix, setup::FourToFourInferenceSetup)
    y_samples = Matrix{Float64}(undef, observation_length(setup), size(param_samples, 2))
    for j in 1:size(param_samples, 2)
        y_samples[:, j] = simulator(view(param_samples, :, j), setup)
    end
    return y_samples
end

function weighted_residual(p, setup::FourToFourInferenceSetup, y_obs::AbstractVector, sigma_model::AbstractVector)
    return TrafficNetworks.weighted_residual(simulator, p, setup, y_obs, sigma_model)
end

function weighted_residual_forwarddiff(p, setup::FourToFourInferenceSetup, y_obs::AbstractVector, sigma_model::AbstractVector)
    return TrafficNetworks.weighted_residual(simulator_forwarddiff, p, setup, y_obs, sigma_model)
end

function build_weighted_inference_problem(
    setup::FourToFourInferenceSetup,
    y_obs::AbstractVector,
    sigma_model::AbstractVector;
    prior_scale=1.0,
    noise_prior_spread=0.05,
)
    y_target = zeros(length(y_obs))
    obs = SimulatorObservable(:y, state -> state.u, (length(y_target),))
    forward_prob = SimulatorForwardProblem(
        p -> weighted_residual(p, setup, y_obs, sigma_model),
        zeros(12),
        obs,
    )
    model_prior = prior(
        z11=Normal(0.0, prior_scale), z12=Normal(0.0, prior_scale), z13=Normal(0.0, prior_scale),
        z21=Normal(0.0, prior_scale), z22=Normal(0.0, prior_scale), z23=Normal(0.0, prior_scale),
        z31=Normal(0.0, prior_scale), z32=Normal(0.0, prior_scale), z33=Normal(0.0, prior_scale),
        z41=Normal(0.0, prior_scale), z42=Normal(0.0, prior_scale), z43=Normal(0.0, prior_scale),
    )
    noise_prior = prior(sigma=LogNormal(log(1.0), noise_prior_spread))
    lik = SimulatorLikelihood(IsoNormal, obs, y_target, noise_prior, :y)
    return SimulatorInferenceProblem(forward_prob, nothing, model_prior, lik)
end

function build_weighted_inference_problem_forwarddiff(
    setup::FourToFourInferenceSetup,
    y_obs::AbstractVector,
    sigma_model::AbstractVector;
    prior_scale=1.0,
    noise_prior_spread=0.05,
)
    y_target = zeros(length(y_obs))
    obs = SimulatorObservable(:y, state -> state.u, (length(y_target),))
    forward_prob = SimulatorForwardProblem(
        p -> weighted_residual_forwarddiff(p, setup, y_obs, sigma_model),
        zeros(12),
        obs,
    )
    model_prior = prior(
        z11=Normal(0.0, prior_scale), z12=Normal(0.0, prior_scale), z13=Normal(0.0, prior_scale),
        z21=Normal(0.0, prior_scale), z22=Normal(0.0, prior_scale), z23=Normal(0.0, prior_scale),
        z31=Normal(0.0, prior_scale), z32=Normal(0.0, prior_scale), z33=Normal(0.0, prior_scale),
        z41=Normal(0.0, prior_scale), z42=Normal(0.0, prior_scale), z43=Normal(0.0, prior_scale),
    )
    noise_prior = prior(sigma=LogNormal(log(1.0), noise_prior_spread))
    lik = SimulatorLikelihood(IsoNormal, obs, y_target, noise_prior, :y)
    return SimulatorInferenceProblem(forward_prob, nothing, model_prior, lik)
end

struct ForwardDiffGradientLogDensity{L}
    l::L
end

logdensity(wrapper::ForwardDiffGradientLogDensity, x::AbstractVector) = logdensity(wrapper.l, x)
capabilities(::Type{<:ForwardDiffGradientLogDensity}) = LogDensityOrder{1}()
dimension(wrapper::ForwardDiffGradientLogDensity) = dimension(wrapper.l)

function logdensity_and_gradient(wrapper::ForwardDiffGradientLogDensity, x::AbstractVector)
    f = z -> logdensity(wrapper.l, z)
    y = f(x)
    grad = ForwardDiff.gradient(f, x)
    return y, grad
end

function solve_with_nuts(
    inference_prob,
    setup::FourToFourInferenceSetup;
    rng=MersenneTwister(42),
    num_samples=40,
    warmup_stages=DynamicHMC.default_warmup_stages(
        init_steps=20,
        middle_steps=20,
        doubling_stages=2,
        terminating_steps=20,
    ),
)
    b = SimulationBasedInference.bijector(inference_prob)
    b_inv = SimulationBasedInference.inverse(b)
    wrapped_logdensity = ForwardDiffGradientLogDensity(logdensity(inference_prob))
    initialization = (; q=b(sample(rng, inference_prob.prior)))

    solve_seconds = @elapsed begin
        warmup = DynamicHMC.mcmc_keep_warmup(
            rng,
            wrapped_logdensity,
            0;
            initialization=initialization,
            warmup_stages=warmup_stages,
            reporter=DynamicHMC.NoProgressReport(),
        )

        steps = DynamicHMC.mcmc_steps(warmup.sampling_logdensity, warmup.final_warmup_state)
        state = warmup.final_warmup_state.Q
        q_samples = Matrix{Float64}(undef, length(state.q), num_samples)

        for i in 1:num_samples
            state, _ = DynamicHMC.mcmc_next_step(steps, state)
            q_samples[:, i] = state.q
        end

        param_samples = reduce(hcat, map(q -> model_parameter_vector(b_inv(q)), eachcol(q_samples)))
        predictive_samples = simulate_posterior_draws(param_samples, setup)
        solution = (
            warmup=warmup,
            final_state=state,
            q_samples=q_samples,
        )
    end

    weights = fill(1.0 / size(param_samples, 2), size(param_samples, 2))
    return solution, solve_seconds, param_samples, weights, "posterior draws", size(param_samples, 2), predictive_samples
end

function solve_with_method(
    inference_prob;
    method=:enis,
    rng=MersenneTwister(42),
    ensemble_size=2_500,
    eks_maxiters=4,
    esmda_maxiters=6,
    num_samples=40,
    nuts_warmup_stages=DynamicHMC.default_warmup_stages(
        init_steps=20,
        middle_steps=20,
        doubling_stages=2,
        terminating_steps=20,
    ),
    setup=default_setup(),
)
    if method == :enis
        solve_seconds = @elapsed sol = solve(inference_prob, EnIS(); ensemble_size=ensemble_size, rng=rng)
        param_samples = Array(get_transformed_ensemble(sol))
        weights = get_weights(sol)
        weights ./= sum(weights)
        diagnostic_label = "importance ESS"
        diagnostic_value = 1 / sum(weights .^ 2)
        predictive_samples = nothing
    elseif method == :eks
        solve_seconds = @elapsed sol = solve(
            inference_prob,
            EKS(maxiters=eks_maxiters),
            ensemble_size=ensemble_size,
            rng=rng,
            verbose=false,
        )
        param_samples = Array(get_transformed_ensemble(sol))
        weights = fill(1.0 / size(param_samples, 2), size(param_samples, 2))
        diagnostic_label = "ensemble members"
        diagnostic_value = size(param_samples, 2)
        predictive_samples = nothing
    elseif method == :esmda
        solve_seconds = @elapsed sol = solve(
            inference_prob,
            ESMDA(maxiters=esmda_maxiters),
            ensemble_size=ensemble_size,
            rng=rng,
            verbose=false,
        )
        param_samples = Array(get_transformed_ensemble(sol))
        weights = fill(1.0 / size(param_samples, 2), size(param_samples, 2))
        diagnostic_label = "ensemble members"
        diagnostic_value = size(param_samples, 2)
        predictive_samples = nothing
    elseif method == :nuts
        sol, solve_seconds, param_samples, weights, diagnostic_label, diagnostic_value, predictive_samples = solve_with_nuts(
            inference_prob,
            setup;
            rng=rng,
            num_samples=num_samples,
            warmup_stages=nuts_warmup_stages,
        )
    else
        error("Unknown method $(method). Use :enis, :eks, :esmda, or :nuts.")
    end

    return sol, solve_seconds, param_samples, weights, diagnostic_label, diagnostic_value, predictive_samples
end

function run_four_to_four_inference(;
    rng=MersenneTwister(42),
    method=:enis,
    ensemble_size=2_500,
    num_samples=40,
    prior_scale=1.00,
    noise_peak_sigma=0.03,
    noise_prior_spread=0.05,
    sigma_floor_fraction=0.03,
    sigma_floor_absolute=1e-3,
    setup=default_setup(),
    P_true=true_turning_matrix(),
    eks_maxiters=4,
    esmda_maxiters=6,
    nuts_warmup_stages=DynamicHMC.default_warmup_stages(
        init_steps=20,
        middle_steps=20,
        doubling_stages=2,
        terminating_steps=20,
    ),
)
    y_true = simulator(P_true, setup)
    noise = generate_physical_observations(y_true, noise_peak_sigma, rng)
    sigma_model = inference_sigma_from_observation(
        noise.y_obs,
        noise_peak_sigma;
        floor_fraction=sigma_floor_fraction,
        absolute_floor=sigma_floor_absolute,
    )
    inference_prob = if method == :nuts
        build_weighted_inference_problem_forwarddiff(
            setup,
            noise.y_obs,
            sigma_model;
            prior_scale=prior_scale,
            noise_prior_spread=noise_prior_spread,
        )
    else
        build_weighted_inference_problem(
            setup,
            noise.y_obs,
            sigma_model;
            prior_scale=prior_scale,
            noise_prior_spread=noise_prior_spread,
        )
    end

    sol, solve_seconds, param_samples, weights, diagnostic_label, diagnostic_value, predictive_samples = solve_with_method(
        inference_prob;
        method=method,
        rng=rng,
        ensemble_size=ensemble_size,
        num_samples=num_samples,
        eks_maxiters=eks_maxiters,
        esmda_maxiters=esmda_maxiters,
        nuts_warmup_stages=nuts_warmup_stages,
        setup=setup,
    )

    model_param_samples = param_samples[1:12, :]
    fraction_samples = turning_entry_samples(model_param_samples)
    entry_true = turning_entries(P_true)
    entry_post_mean, entry_post_median, entry_ci_05, entry_ci_95 = sample_summary(fraction_samples, weights)
    P_post_mean = reshape(entry_post_mean, 4, 4)'

    return (
        method=method,
        setup=setup,
        prior_scale=prior_scale,
        noise_peak_sigma=noise_peak_sigma,
        noise_prior_spread=noise_prior_spread,
        sigma_floor_fraction=sigma_floor_fraction,
        sigma_floor_absolute=sigma_floor_absolute,
        sigma_true_vec=noise.sigma_true,
        sigma_model=sigma_model,
        sigma_true_mean=mean(noise.sigma_true),
        sigma_model_mean=mean(sigma_model),
        clip_fraction=noise.clip_fraction,
        P_true=validate_turning_matrix(P_true),
        entry_true=entry_true,
        y_true=y_true,
        y_obs=noise.y_obs,
        solution=sol,
        weights=weights,
        param_samples=model_param_samples,
        fraction_samples=fraction_samples,
        entry_post_mean=entry_post_mean,
        entry_post_median=entry_post_median,
        entry_ci_05=entry_ci_05,
        entry_ci_95=entry_ci_95,
        P_post_mean=P_post_mean,
        y_post_mean=if predictive_samples === nothing
            posterior_predictive_mean(sol, weights, noise.y_obs, sigma_model)
        else
            weighted_column_mean(predictive_samples, weights)
        end,
        solve_seconds=solve_seconds,
        diagnostic_label=diagnostic_label,
        diagnostic_value=diagnostic_value,
    )
end

turning_rmse(results) = TrafficNetworks.rmse(results.entry_post_mean, results.entry_true)
predictive_rmse(results) = TrafficNetworks.rmse(results.y_post_mean, results.y_true)
mean_interval_width(results) = TrafficNetworks.mean_interval_width(results.entry_ci_05, results.entry_ci_95)
interval_coverage_mask(results) = TrafficNetworks.interval_coverage_mask(results.entry_true, results.entry_ci_05, results.entry_ci_95)
interval_coverage_rate(results) = TrafficNetworks.interval_coverage_rate(results.entry_true, results.entry_ci_05, results.entry_ci_95)

uniform_turning_matrix() = fill(0.25, 4, 4)

function sensitivity_to_uniform(setup::FourToFourInferenceSetup; P_true=true_turning_matrix())
    y_true = simulator(P_true, setup)
    y_uniform = simulator(uniform_turning_matrix(), setup)
    return (
        rmse=TrafficNetworks.rmse(y_true, y_uniform),
        maxabs=maximum(abs.(y_true .- y_uniform)),
    )
end

function estimate_nuts_cost(;
    rng_seed=42,
    noise_peak_sigma=0.03,
    prior_scale=1.0,
    repeats=3,
)
    setup = default_setup()
    rng = MersenneTwister(rng_seed)
    P_true = true_turning_matrix()
    y_true = simulator(P_true, setup)
    noise = generate_physical_observations(y_true, noise_peak_sigma, rng)
    inference_prob = build_weighted_inference_problem_forwarddiff(
        setup,
        noise.y_obs,
        noise.sigma_model;
        prior_scale=prior_scale,
    )
    wrapped = ForwardDiffGradientLogDensity(logdensity(inference_prob))
    b = SimulationBasedInference.bijector(inference_prob)
    q0 = b(sample(rng, inference_prob.prior))

    logdensity_and_gradient(wrapped, q0)
    grad_times = [@elapsed logdensity_and_gradient(wrapped, q0) for _ in 1:repeats]
    avg_grad_seconds = mean(grad_times)

    estimates = (
        two_hundred_fifty_gradients=250 * avg_grad_seconds,
        five_hundred_gradients=500 * avg_grad_seconds,
        one_thousand_gradients=1_000 * avg_grad_seconds,
    )

    return (
        setup=setup,
        parameter_dimension=length(q0),
        noise_peak_sigma=noise_peak_sigma,
        avg_grad_seconds=avg_grad_seconds,
        estimates=estimates,
    )
end

function print_matrix_side_by_side(label_left, left, label_right, right)
    println(label_left, "            ", label_right)
    for row in 1:size(left, 1)
        @printf("  [%.4f  %.4f  %.4f  %.4f]    [%.4f  %.4f  %.4f  %.4f]\n",
            left[row, 1], left[row, 2], left[row, 3], left[row, 4],
            right[row, 1], right[row, 2], right[row, 3], right[row, 4],
        )
    end
end

function print_turning_entry_summary(results)
    println("Turning-fraction entries")
    inside_mask = interval_coverage_mask(results)
    for i in 1:4
        for j in 1:4
            idx = 4 * (i - 1) + j
            @printf(
                "  P[%d,%d]: true = %.4f, mean = %.4f, 90%% CI = [%.4f, %.4f], inside = %s\n",
                i,
                j,
                results.entry_true[idx],
                results.entry_post_mean[idx],
                results.entry_ci_05[idx],
                results.entry_ci_95[idx],
                inside_mask[idx] ? "yes" : "no",
            )
        end
    end
end

function comparison_row(results)
    return (
        method=String(results.method),
        turning_rmse=turning_rmse(results),
        predictive_rmse=predictive_rmse(results),
        mean_interval_width=mean_interval_width(results),
        interval_coverage_count=sum(interval_coverage_mask(results)),
        interval_coverage_rate=interval_coverage_rate(results),
        runtime_seconds=results.solve_seconds,
    )
end

function print_comparison_table(results_list)
    println("Comparison summary")
    println("------------------")
    println("method    turning_RMSE  predictive_RMSE  mean_90%_width  coverage  runtime_s")
    for results in results_list
        row = comparison_row(results)
        @printf(
            "%-8s  %12.4f  %15.4f  %14.4f  %2d/16 (%.2f)  %9.2f\n",
            row.method,
            row.turning_rmse,
            row.predictive_rmse,
            row.mean_interval_width,
            row.interval_coverage_count,
            row.interval_coverage_rate,
            row.runtime_seconds,
        )
    end
end

function warmup_method_comparison(setup::FourToFourInferenceSetup, trials; rng_seed=42)
    for trial in trials
        run_four_to_four_inference(
            rng=MersenneTwister(rng_seed),
            setup=setup,
            method=trial.method;
            trial.kwargs...,
        )
    end

    estimate_nuts_cost(rng_seed=rng_seed, repeats=1)
    return nothing
end

function print_observation_preview(results)
    setup = results.setup
    Y_true = reshape_observations(results.y_true, setup)
    Y_pred = reshape_observations(results.y_post_mean, setup)

    road_positions = findall(road_id -> road_id >= 5, setup.observed_road_ids)
    if isempty(road_positions)
        road_positions = collect(eachindex(setup.observed_road_ids))
    end

    sensor_positions = unique([1, length(setup.observed_cell_ids)])
    time_idx = length(setup.control_times)

    println("Representative outgoing-road observations at final time")
    for road_pos in road_positions
        road_id = setup.observed_road_ids[road_pos]
        for sensor_pos in sensor_positions
            cell_id = setup.observed_cell_ids[sensor_pos]
            @printf(
                "  road %d, cell %d, t = %.2f: true = %.4f, pred = %.4f, error = %.4f\n",
                road_id,
                cell_id,
                setup.control_times[time_idx],
                Y_true[sensor_pos, time_idx, road_pos],
                Y_pred[sensor_pos, time_idx, road_pos],
                Y_pred[sensor_pos, time_idx, road_pos] - Y_true[sensor_pos, time_idx, road_pos],
            )
        end
    end
end

function print_final_time_road_summary(results)
    setup = results.setup
    Y_true = reshape_observations(results.y_true, setup)
    Y_pred = reshape_observations(results.y_post_mean, setup)

    println("Final-time mean density across observed cells")
    for (k, road_id) in enumerate(setup.observed_road_ids)
        true_mean = mean(Y_true[:, end, k])
        pred_mean = mean(Y_pred[:, end, k])
        @printf(
            "  road %d: true = %.4f, pred = %.4f, error = %.4f\n",
            road_id,
            true_mean,
            pred_mean,
            pred_mean - true_mean,
        )
    end
end

function print_summary(results)
    setup = results.setup

    println("Four-to-four junction turning-fraction inference")
    println("------------------------------------------------")
    println("Method: ", results.method)
    println("Control times: ", setup.control_times)
    println("Observed roads: ", setup.observed_road_ids)
    println("Observed cells: ", setup.observed_cell_ids)
    println("Observation vector length: ", observation_length(setup))
    println("Road lengths (km): ", setup.road_lengths)
    println("Speed limits (km/h): ", setup.speed_limits)
    println("Prior center per row: [0.25, 0.25, 0.25, 0.25]")
    println("Experiment design: physical same-length roads, staggered inflow pulses, outgoing-road observations")
    @printf("Prior logit scale: %.3f\n", results.prior_scale)
    @printf("Physical noise peak sigma: %.4f\n", results.noise_peak_sigma)
    @printf("Noise prior on global scale: LogNormal(log(1.0), %.2f)\n", results.noise_prior_spread)
    @printf("Mean true sigma: %.4f\n", results.sigma_true_mean)
    @printf("Mean model sigma: %.4f\n", results.sigma_model_mean)
    @printf("Observation clip fraction: %.4f\n", results.clip_fraction)
    @printf("solve time: %.2f s", results.solve_seconds)
    if results.solve_seconds > METHOD_TIME_BUDGET_SECONDS
        print("  (over 60 s budget)")
    end
    println()
    println(results.diagnostic_label, ": ", results.diagnostic_value)
    @printf("turning RMSE: %.4f\n", turning_rmse(results))
    @printf("predictive RMSE: %.4f\n", predictive_rmse(results))
    @printf("mean 90%% interval width: %.4f\n", mean_interval_width(results))
    @printf("90%% interval coverage: %d/16 (%.2f)\n", sum(interval_coverage_mask(results)), interval_coverage_rate(results))
    println()

    print_matrix_side_by_side("True turning matrix:", results.P_true, "Posterior mean:", results.P_post_mean)
    println()
    print_turning_entry_summary(results)
    println()
    print_final_time_road_summary(results)
    println()
    print_observation_preview(results)
end

function recommend_method(method_results)
    feasible = filter(r -> r.solve_seconds <= METHOD_TIME_BUDGET_SECONDS, method_results)
    isempty(feasible) && return nothing
    return first(sort(feasible; by=r -> (turning_rmse(r), predictive_rmse(r), r.solve_seconds)))
end

function run_method_comparison(; rng_seed=42, warmup=true)
    setup = default_setup()
    sens = sensitivity_to_uniform(setup)

    trials = [
        MethodTrial(:enis, (; ensemble_size=600)),
        MethodTrial(:eks, (; ensemble_size=144, eks_maxiters=3)),
        MethodTrial(:esmda, (; ensemble_size=96, esmda_maxiters=6)),
    ]

    results = NamedTuple[]

    println("==============================================================")
    println("Identifiability diagnostic")
    @printf("RMSE between true observations and equal-split observations: %.4f\n", sens.rmse)
    @printf("Max absolute observation difference: %.4f\n", sens.maxabs)
    println("Observation noise: physical clipped noise with peak sigma 0.0300")
    println("Comparison target: roughly one minute per method (after a short warmup pass)")
    println()

    if warmup
        println("Warming up inference methods for fairer runtime measurements...")
        warmup_method_comparison(setup, trials; rng_seed=rng_seed)
        println()
    end

    for (i, trial) in enumerate(trials)
        println("==============================================================")
        println("Running method ", i, "/", length(trials), ": ", trial.method)
        result = run_four_to_four_inference(
            rng=MersenneTwister(rng_seed),
            setup=setup,
            method=trial.method;
            trial.kwargs...,
        )
        print_summary(result)
        push!(results, result)
        println()
    end

    best = recommend_method(results)
    println("==============================================================")
    if best === nothing
        println("No method finished within the 60 s budget.")
    else
        println("Recommended method for this 4-to-4 setup: ", best.method)
        @printf("  solve time      = %.2f s\n", best.solve_seconds)
        @printf("  turning RMSE    = %.4f\n", turning_rmse(best))
        @printf("  predictive RMSE = %.4f\n", predictive_rmse(best))
        @printf("  mean 90%% width  = %.4f\n", mean_interval_width(best))
        @printf("  coverage        = %d/16 (%.2f)\n", sum(interval_coverage_mask(best)), interval_coverage_rate(best))
    end
    println()

    nuts_cost = estimate_nuts_cost(rng_seed=rng_seed)
    print_comparison_table(results)
    println()
    println("NUTS AD cost estimate")
    println("--------------------")
    println("AD-based NUTS is available for this problem, but it is not included in the one-minute table because actual DynamicHMC warmup remains far above the target budget.")
    println("parameter dimension: ", nuts_cost.parameter_dimension)
    @printf("average logdensity+gradient call: %.4f s\n", nuts_cost.avg_grad_seconds)
    @printf("250 gradients  ~= %.2f s\n", nuts_cost.estimates.two_hundred_fifty_gradients)
    @printf("500 gradients  ~= %.2f s\n", nuts_cost.estimates.five_hundred_gradients)
    @printf("1000 gradients ~= %.2f s\n", nuts_cost.estimates.one_thousand_gradients)

    return results, best, nuts_cost
end
