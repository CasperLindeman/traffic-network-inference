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
import TrafficNetworks: weighted_quantile

const ONE_TO_TWO_OUTPUT_DIR = joinpath(@__DIR__, "figures")
const ONE_TO_TWO_NETWORK_SPEC_PATH = joinpath(@__DIR__, "..", "network_specs", "one_to_two.toml")
const ONE_TO_TWO_NETWORK_SPEC = load_experiment_network_spec(ONE_TO_TWO_NETWORK_SPEC_PATH)

const ROAD_LABELS = Tuple(TrafficNetworks.road_labels(ONE_TO_TWO_NETWORK_SPEC))

struct OneToTwoInferenceSetup
    T::Float64
    CFL::Float64
    control_times::Vector{Float64}
    observed_road_ids::Vector{Int}
    observed_cell_ids::Vector{Int}
    network_spec::ExperimentNetworkSpec
end

road_cell_count(setup::OneToTwoInferenceSetup, road_id::Int) =
    TrafficNetworks.road_cell_count(setup.network_spec, road_id)

road_length_km(setup::OneToTwoInferenceSetup, road_id::Int) =
    TrafficNetworks.road_length_km(setup.network_spec, road_id)

cell_center_distance_km(setup::OneToTwoInferenceSetup, road_id::Int, cell_id::Int) =
    TrafficNetworks.cell_center_distance_km(setup.network_spec, road_id, cell_id)

function default_setup(; network_spec=ONE_TO_TWO_NETWORK_SPEC)
    observed_road_ids = TrafficNetworks.observed_road_ids(network_spec)
    observed_cell_ids = TrafficNetworks.observed_cell_ids(network_spec; reference_road_id=2)
    return OneToTwoInferenceSetup(
        network_spec.T,
        network_spec.CFL,
        copy(network_spec.control_times),
        observed_road_ids,
        observed_cell_ids,
        network_spec,
    )
end

extract_turning_fraction(p::AbstractVector) = only(p)
extract_turning_fraction(p::NamedTuple) = p.a

function build_one_to_two_network(a, setup::OneToTwoInferenceSetup)
    @assert 0.0 <= a <= 1.0 "Turning fraction a must lie in [0, 1]"
    rule = TurningFractionRule(reshape([a, 1.0 - a], 1, 2))

    return TrafficNetworks.build_experiment_network(setup.network_spec, [rule]; T=setup.T, CFL=setup.CFL)
end

observation_length(setup::OneToTwoInferenceSetup) =
    TrafficNetworks.cell_observation_length(
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )

function flatten_observations(hist::SimulationHistory, setup::OneToTwoInferenceSetup)
    return TrafficNetworks.flatten_cell_observations(
        hist,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end

function simulator(p, setup::OneToTwoInferenceSetup)
    a = extract_turning_fraction(p)
    net = build_one_to_two_network(a, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return flatten_observations(hist, setup)
end

function simulation_step_count(p, setup::OneToTwoInferenceSetup)
    a = extract_turning_fraction(p)
    net = build_one_to_two_network(a, setup)
    return TrafficNetworks.simulation_step_count(net; times=setup.control_times)
end

function reshape_observations(y::AbstractVector, setup::OneToTwoInferenceSetup)
    return TrafficNetworks.reshape_cell_observations(
        y,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end

function posterior_predictive_mean(sol, weights, n_obs)
    Y = Array(get_observables(sol).y)
    Y = reshape(Y, n_obs, :)
    return vec(sum(Y .* weights', dims=2))
end

function run_one_to_two_inference(;
    rng=MersenneTwister(42),
    ensemble_size=1_000,
    a_true=0.68,
    sigma_true=0.01,
    setup=default_setup(),
    prior_alpha=3.0,
    prior_beta=3.0,
    noise_prior_spread=0.25,
)
    simulator_for_setup = p -> simulator(p, setup)
    y_true = simulator([a_true], setup)
    y_obs = y_true .+ sigma_true .* randn(rng, length(y_true))

    obs = SimulatorObservable(:y, state -> state.u, (length(y_true),))
    forward_prob = SimulatorForwardProblem(simulator_for_setup, [a_true], obs)

    model_prior = prior(a=Beta(prior_alpha, prior_beta))
    noise_prior = prior(sigma=LogNormal(log(sigma_true), noise_prior_spread))

    lik = SimulatorLikelihood(IsoNormal, obs, y_obs, noise_prior, :y)
    inference_prob = SimulatorInferenceProblem(forward_prob, nothing, model_prior, lik)

    sol = solve(inference_prob, EnIS(); ensemble_size=ensemble_size, rng=rng)

    weights = get_weights(sol)
    weights ./= sum(weights)

    ensemble = get_transformed_ensemble(sol)
    a_samples = vec(ensemble[1, :])

    a_post_mean = sum(a_samples .* weights)
    a_post_median = weighted_quantile(a_samples, weights, 0.5)
    a_ci_05 = weighted_quantile(a_samples, weights, 0.05)
    a_ci_95 = weighted_quantile(a_samples, weights, 0.95)
    ess = 1 / sum(weights .^ 2)

    y_post_mean = posterior_predictive_mean(sol, weights, length(y_true))

    return (
        method=:enis,
        ensemble_size=ensemble_size,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        noise_prior_spread=noise_prior_spread,
        setup=setup,
        a_true=a_true,
        sigma_true=sigma_true,
        y_true=y_true,
        y_obs=y_obs,
        solution=sol,
        weights=weights,
        a_samples=a_samples,
        a_post_mean=a_post_mean,
        a_post_median=a_post_median,
        a_ci_05=a_ci_05,
        a_ci_95=a_ci_95,
        ess=ess,
        y_post_mean=y_post_mean,
        truth_simulation_steps=simulation_step_count([a_true], setup),
    )
end

function print_summary(results)
    setup = results.setup
    control_times_s = control_times_seconds(setup)

    println("One-to-two junction turning-fraction inference")
    println("------------------------------------------------")
    println("Method: Ensemble importance sampling (EnIS)")
    @printf("Prior for a: Beta(%.1f, %.1f)\n", results.prior_alpha, results.prior_beta)
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

    sensor_cell = only(setup.observed_cell_ids)
    for road_id in setup.observed_road_ids
        sensor_distance = cell_center_distance_km(setup, road_id, sensor_cell)
        @printf(
            "Sensor on %s at cell %d (%.3f km downstream of the junction)\n",
            ROAD_LABELS[road_id],
            sensor_cell,
            sensor_distance,
        )
    end

    println()

    @printf("true a                = %.4f\n", results.a_true)
    @printf("posterior mean a      = %.4f\n", results.a_post_mean)
    @printf("posterior median a    = %.4f\n", results.a_post_median)
    @printf("90%% credible interval = [%.4f, %.4f]\n", results.a_ci_05, results.a_ci_95)
    @printf("effective sample size = %.1f / %d\n", results.ess, length(results.weights))
end

if get(ENV, "TRAFFICNETWORKS_SKIP_ONE_TO_TWO_DEMO", "0") != "1"
    results = run_one_to_two_inference()
    print_summary(results)
end
