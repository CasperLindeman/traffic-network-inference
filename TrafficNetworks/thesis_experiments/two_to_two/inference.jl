import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Random
using Printf
using Distributions
using SimulationBasedInference
using TrafficNetworks
import TrafficNetworks:
    ExperimentNetworkSpec,
    control_times_seconds,
    load_experiment_network_spec
import TrafficNetworks: sample_summary

const TWO_TO_TWO_NETWORK_SPEC_PATH = joinpath(@__DIR__, "..", "network_specs", "two_to_two.toml")
const TWO_TO_TWO_NETWORK_SPEC = load_experiment_network_spec(TWO_TO_TWO_NETWORK_SPEC_PATH)
const ROAD_LABELS = Tuple(TrafficNetworks.road_labels(TWO_TO_TWO_NETWORK_SPEC))

struct TwoToTwoInferenceSetup
    T::Float64
    CFL::Float64
    control_times::Vector{Float64}
    observed_road_ids::Vector{Int}
    observed_cell_ids::Vector{Int}
    network_spec::ExperimentNetworkSpec
end

road_cell_count(setup::TwoToTwoInferenceSetup, road_id::Int) =
    TrafficNetworks.road_cell_count(setup.network_spec, road_id)

road_length_km(setup::TwoToTwoInferenceSetup, road_id::Int) =
    TrafficNetworks.road_length_km(setup.network_spec, road_id)

cell_center_distance_km(setup::TwoToTwoInferenceSetup, road_id::Int, cell_id::Int) =
    TrafficNetworks.cell_center_distance_km(setup.network_spec, road_id, cell_id)

distance_to_junction_km(setup::TwoToTwoInferenceSetup, road_id::Int, cell_id::Int) =
    road_id in (1, 2) ? road_length_km(setup, road_id) - cell_center_distance_km(setup, road_id, cell_id) :
    cell_center_distance_km(setup, road_id, cell_id)

function default_setup(; network_spec=TWO_TO_TWO_NETWORK_SPEC)
    observed_road_ids = TrafficNetworks.observed_road_ids(network_spec)
    observed_cell_ids = TrafficNetworks.observed_cell_ids(network_spec; mode=:paired)
    return TwoToTwoInferenceSetup(
        network_spec.T,
        network_spec.CFL,
        copy(network_spec.control_times),
        observed_road_ids,
        observed_cell_ids,
        network_spec,
    )
end

extract_turning_parameters(p::AbstractVector) = (p1=p[1], p2=p[2])
extract_turning_parameters(p::NamedTuple) = (p1=p.p1, p2=p.p2)

function turning_matrix(p)
    pars = extract_turning_parameters(p)

    @assert 0.0 <= pars.p1 <= 1.0 "p1 must lie in [0, 1]"
    @assert 0.0 <= pars.p2 <= 1.0 "p2 must lie in [0, 1]"

    return [
        pars.p1 1.0 - pars.p1
        pars.p2 1.0 - pars.p2
    ]
end

turning_entries(P::AbstractMatrix) = [P[1, 1], P[1, 2], P[2, 1], P[2, 2]]

function turning_entry_samples(param_samples::AbstractMatrix)
    samples = Matrix{Float64}(undef, 4, size(param_samples, 2))
    for j in 1:size(param_samples, 2)
        samples[:, j] = turning_entries(turning_matrix(view(param_samples, :, j)))
    end
    return samples
end

function build_two_to_two_network(p, setup::TwoToTwoInferenceSetup)
    rule = TurningFractionRule(turning_matrix(p))
    return TrafficNetworks.build_experiment_network(setup.network_spec, [rule]; T=setup.T, CFL=setup.CFL)
end

observation_length(setup::TwoToTwoInferenceSetup) =
    TrafficNetworks.paired_cell_observation_length(setup.observed_road_ids, setup.control_times)

function flatten_observations(hist::SimulationHistory, setup::TwoToTwoInferenceSetup)
    return TrafficNetworks.flatten_paired_cell_observations(
        hist,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end

function simulator(p, setup::TwoToTwoInferenceSetup)
    net = build_two_to_two_network(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return flatten_observations(hist, setup)
end

function simulation_step_count(p, setup::TwoToTwoInferenceSetup)
    net = build_two_to_two_network(p, setup)
    return TrafficNetworks.simulation_step_count(net; times=setup.control_times)
end

function posterior_predictive_mean(sol, weights)
    Y = reshape(Array(get_observables(sol).y), :, length(weights))
    return vec(sum(Y .* reshape(weights, 1, :), dims=2))
end

function run_two_to_two_inference(;
    rng=MersenneTwister(42),
    ensemble_size=1_000,
    p_true=(p1=0.68, p2=0.42),
    sigma_true=0.03,
    setup=default_setup(),
    prior_shape=3.0,
    noise_prior_spread=0.25,
)
    simulator_for_setup = p -> simulator(p, setup)
    pars_true = extract_turning_parameters(p_true)
    param_true = [pars_true.p1, pars_true.p2]
    P_true = turning_matrix(p_true)

    y_true = simulator(p_true, setup)
    y_obs = y_true .+ sigma_true .* randn(rng, length(y_true))

    obs = SimulatorObservable(:y, state -> state.u, (length(y_true),))
    forward_prob = SimulatorForwardProblem(simulator_for_setup, [0.5, 0.5], obs)

    model_prior = prior(
        p1=Beta(prior_shape, prior_shape),
        p2=Beta(prior_shape, prior_shape),
    )
    noise_prior = prior(sigma=LogNormal(log(sigma_true), noise_prior_spread))

    lik = SimulatorLikelihood(IsoNormal, obs, y_obs, noise_prior, :y)
    inference_prob = SimulatorInferenceProblem(forward_prob, nothing, model_prior, lik)

    solve_seconds = @elapsed sol = solve(inference_prob, EnIS(); ensemble_size=ensemble_size, rng=rng)
    param_samples = Array(get_transformed_ensemble(sol))
    weights = get_weights(sol)
    weights ./= sum(weights)
    diagnostic_value = 1 / sum(weights .^ 2)

    param_post_mean, param_post_median, param_ci_05, param_ci_95 = sample_summary(param_samples, weights)

    fraction_samples = turning_entry_samples(param_samples)
    entry_true = turning_entries(P_true)
    entry_post_mean, entry_post_median, entry_ci_05, entry_ci_95 = sample_summary(fraction_samples, weights)

    P_post_mean = [
        entry_post_mean[1] entry_post_mean[2]
        entry_post_mean[3] entry_post_mean[4]
    ]

    return (
        method=:enis,
        setup=setup,
        param_true=param_true,
        p_true=P_true,
        sigma_true=sigma_true,
        y_true=y_true,
        y_obs=y_obs,
        solution=sol,
        weights=weights,
        param_samples=param_samples,
        fraction_samples=fraction_samples,
        param_post_mean=param_post_mean,
        param_post_median=param_post_median,
        param_ci_05=param_ci_05,
        param_ci_95=param_ci_95,
        entry_true=entry_true,
        entry_post_mean=entry_post_mean,
        entry_post_median=entry_post_median,
        entry_ci_05=entry_ci_05,
        entry_ci_95=entry_ci_95,
        P_post_mean=P_post_mean,
        y_post_mean=posterior_predictive_mean(sol, weights),
        ensemble_size=ensemble_size,
        prior_shape=prior_shape,
        noise_prior_spread=noise_prior_spread,
        truth_simulation_steps=simulation_step_count(p_true, setup),
        solve_seconds=solve_seconds,
        diagnostic_label="importance ESS",
        diagnostic_value=diagnostic_value,
    )
end

function print_matrix(label, P)
    println(label)
    @printf("  [%.4f  %.4f]\n", P[1, 1], P[1, 2])
    @printf("  [%.4f  %.4f]\n", P[2, 1], P[2, 2])
end

function print_summary(results)
    setup = results.setup
    param_labels = ("p1", "p2")
    entry_labels = ("P11", "P12", "P21", "P22")
    control_times_s = control_times_seconds(setup)

    println("Two-to-two junction turning-fraction inference")
    println("------------------------------------------------")
    println("Parameterization: row 1 = [p1, 1-p1], row 2 = [p2, 1-p2]")
    println("Inference method: ", results.method)
    @printf("Prior for each parameter: Beta(%.1f, %.1f)\n", results.prior_shape, results.prior_shape)
    @printf("Noise prior: LogNormal(log(%.4f), %.2f)\n", results.sigma_true, results.noise_prior_spread)
    println("Observation times (s): ", control_times_s)
    println("Observed roads: ", setup.observed_road_ids)
    println("Observed cells: ", setup.observed_cell_ids)
    println("Observation vector length: ", observation_length(setup))
    @printf("Reference simulation time steps: %d\n", results.truth_simulation_steps)
    println()

    @printf(
        "Common basis length L = %.2f km with %d cells per basis block\n",
        setup.network_spec.basis_length_km,
        setup.network_spec.cells_per_block,
    )
    println()

    road_blocks = TrafficNetworks.road_blocks(setup.network_spec)
    speed_limits = TrafficNetworks.speed_limits(setup.network_spec)
    for road_id in eachindex(road_blocks)
        @printf(
            "%s: b = %d, length = %.2f km, cells = %d, speed limit = %d km/h\n",
            ROAD_LABELS[road_id],
            road_blocks[road_id],
            road_length_km(setup, road_id),
            road_cell_count(setup, road_id),
            speed_limits[road_id],
        )
    end

    println()

    for (idx, road_id) in enumerate(setup.observed_road_ids)
        sensor_cell = setup.observed_cell_ids[idx]
        sensor_distance = distance_to_junction_km(setup, road_id, sensor_cell)
        direction_text = road_id in (1, 2) ? "upstream of the junction" : "downstream of the junction"
        @printf(
            "Sensor on %s at cell %d (%.3f km %s)\n",
            ROAD_LABELS[road_id],
            sensor_cell,
            sensor_distance,
            direction_text,
        )
    end

    println()

    print_matrix("True turning matrix:", results.p_true)
    print_matrix("Posterior mean turning matrix:", results.P_post_mean)
    println()

    for i in eachindex(param_labels)
        @printf(
            "%s: true = %.4f, mean = %.4f, median = %.4f, 90%% CI = [%.4f, %.4f]\n",
            param_labels[i],
            results.param_true[i],
            results.param_post_mean[i],
            results.param_post_median[i],
            results.param_ci_05[i],
            results.param_ci_95[i],
        )
    end

    println()

    for i in eachindex(entry_labels)
        @printf(
            "%s: true = %.4f, mean = %.4f, median = %.4f, 90%% CI = [%.4f, %.4f]\n",
            entry_labels[i],
            results.entry_true[i],
            results.entry_post_mean[i],
            results.entry_post_median[i],
            results.entry_ci_05[i],
            results.entry_ci_95[i],
        )
    end

    println(results.diagnostic_label, ": ", results.diagnostic_value)
    @printf("solve time: %.2f s\n", results.solve_seconds)
end

if get(ENV, "TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO", "0") != "1"
    results_enis = run_two_to_two_inference()
    print_summary(results_enis)
end
