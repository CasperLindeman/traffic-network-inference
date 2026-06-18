if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using Statistics
using LinearAlgebra
using ForwardDiff
using Optim

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_ESMDA_ADAM_DEMO"] = "1"
if !isdefined(@__MODULE__, :SquareFourToFourSetup)
    include(joinpath(@__DIR__, "..", "base", "esmda_vs_adam.jl"))
end

FORWARDDIFF_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_FORWARDDIFF_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "generated", "square_four_to_four", "square_four_to_four_esmda_vs_adam_forwarddiff"),
)

parameter_vector_forwarddiff(p::AbstractVector) = TrafficNetworks.parameter_vector(p, SQUARE_TURNING_PARAMETERIZATION)
parameter_vector_forwarddiff(p::NamedTuple) = TrafficNetworks.parameter_vector(p, SQUARE_TURNING_PARAMETERIZATION)

stable_row_softmax_forwarddiff(z::AbstractVector) = TrafficNetworks.stable_row_softmax(z)

function turning_matrices_forwarddiff(p)
    return TrafficNetworks.turning_matrices_from_logits(
        p,
        SQUARE_TURNING_PARAMETERIZATION;
        validate=false,
    )
end
function square_state_eltype(p)
    return promote_type(eltype(parameter_vector_forwarddiff(p)), Float64)
end

function build_square_network_forwarddiff(p, setup::SquareFourToFourSetup)
    Ps = turning_matrices_forwarddiff(p)
    state_T = square_state_eltype(p)
    rules = [TurningFractionRule(Ps[junction]) for junction in 1:N_JUNCTIONS]
    return TrafficNetworks.build_experiment_network(
        SQUARE_NETWORK_SPEC,
        rules;
        T=setup.T,
        CFL=setup.CFL,
        road_length_values=setup.road_lengths,
        road_profile_values=setup.road_profiles,
        speed_limit_values=setup.speed_limits,
        boundary_inflow_values=setup.inflows,
        state_eltype=state_T,
    )
end

function simulate_history_forwarddiff(p, setup::SquareFourToFourSetup)
    net = build_square_network_forwarddiff(p, setup)
    hist = simulate!(net; times=setup.control_times)

    @assert length(hist.times) == length(setup.control_times)
    @assert maximum(abs.(hist.times .- setup.control_times)) <= 1e-12

    return hist
end

function flatten_observations_forwarddiff(hist::SimulationHistory, setup::SquareFourToFourSetup)
    return TrafficNetworks.flatten_cell_observations(
        hist,
        setup.observed_road_ids,
        setup.observed_cell_ids,
        setup.control_times,
    )
end
simulator_forwarddiff(p, setup::SquareFourToFourSetup) =
    flatten_observations_forwarddiff(simulate_history_forwarddiff(p, setup), setup)

function map_loss_forwarddiff(z, y_obs, setup::SquareFourToFourSetup; prior_scale=1.0, sigma_obs=0.003)
    y_pred = simulator_forwarddiff(z, setup)
    residual = (y_pred .- y_obs) ./ sigma_obs
    prior_term = z ./ prior_scale
    return 0.5 * sum(abs2, residual) + 0.5 * sum(abs2, prior_term)
end

function run_adam_map_forwarddiff(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.0,
    sigma_obs=0.003,
    z0=zeros(N_PARAMS),
    learning_rate=0.05,
    maxiters=120,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    grad_clip=300.0,
)
    loss_fn = z -> map_loss_forwarddiff(z, y_obs, setup; prior_scale=prior_scale, sigma_obs=sigma_obs)
    z = Float64.(collect(z0))
    best_z = copy(z)
    best_loss = Inf

    m = zeros(length(z))
    v = zeros(length(z))
    grad = zeros(length(z))
    losses = Float64[]
    grad_norms = Float64[]
    raw_grad_norms = Float64[]
    cfg = ForwardDiff.GradientConfig(loss_fn, z)

    solve_seconds = @elapsed begin
        for iter in 1:maxiters
            loss = loss_fn(z)
            ForwardDiff.gradient!(grad, loss_fn, z, cfg)
            raw_grad_norm = norm(grad)
            grad_norm = raw_grad_norm

            if grad_norm > grad_clip
                grad .*= grad_clip / grad_norm
                grad_norm = grad_clip
            end

            push!(losses, loss)
            push!(raw_grad_norms, raw_grad_norm)
            push!(grad_norms, grad_norm)

            if loss < best_loss
                best_loss = loss
                best_z .= z
            end

            m .= beta1 .* m .+ (1 - beta1) .* grad
            v .= beta2 .* v .+ (1 - beta2) .* (grad .^ 2)

            m_hat = m ./ (1 - beta1^iter)
            v_hat = v ./ (1 - beta2^iter)
            z .-= learning_rate .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
        end
    end

    final_loss = loss_fn(z)
    if final_loss < best_loss
        best_loss = final_loss
        best_z .= z
    end

    P_est = turning_matrices(best_z)
    y_est = simulator(best_z, setup)

    return (
        z=copy(z),
        z_best=best_z,
        P_est=P_est,
        y_est=y_est,
        best_loss=best_loss,
        losses=losses,
        raw_grad_norms=raw_grad_norms,
        grad_norms=grad_norms,
        solve_seconds=solve_seconds,
        iterations=maxiters,
    )
end

function run_lbfgs_map_forwarddiff(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.0,
    sigma_obs=0.003,
    z0=zeros(N_PARAMS),
    maxiters=80,
)
    loss_fn = z -> map_loss_forwarddiff(z, y_obs, setup; prior_scale=prior_scale, sigma_obs=sigma_obs)

    function fg!(F, G, z)
        if G !== nothing
            ForwardDiff.gradient!(G, loss_fn, z)
        end
        if F !== nothing
            return loss_fn(z)
        end
        return nothing
    end

    options = Optim.Options(iterations=maxiters, store_trace=true, show_trace=false)
    solve_seconds = @elapsed result = optimize(
        Optim.only_fg!(fg!),
        Float64.(collect(z0)),
        LBFGS(),
        options,
    )

    z_best = Optim.minimizer(result)
    P_est = turning_matrices(z_best)
    y_est = simulator(z_best, setup)

    return (
        z=copy(z_best),
        z_best=copy(z_best),
        P_est=P_est,
        y_est=y_est,
        best_loss=Optim.minimum(result),
        losses=[tr.value for tr in Optim.trace(result) if tr.value !== nothing],
        raw_grad_norms=Float64[],
        grad_norms=Float64[],
        solve_seconds=solve_seconds,
        iterations=Optim.iterations(result),
        result=result,
    )
end

function run_adam_restart_diagnostics_forwarddiff(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.0,
    sigma_obs=0.003,
    seeds=0:3,
    init_scale=0.8,
    learning_rate=0.05,
    maxiters=120,
    base_result=nothing,
)
    diagnostics = NamedTuple[]

    for seed in seeds
        label = seed == 0 ? "prior mean" : "random $(seed)"
        z0 = seed == 0 ? zeros(N_PARAMS) : init_scale .* randn(MersenneTwister(100 + seed), N_PARAMS)
        adam_result = if seed == 0 && base_result !== nothing
            base_result
        else
            run_adam_map_forwarddiff(
                y_obs,
                setup;
                prior_scale=prior_scale,
                sigma_obs=sigma_obs,
                z0=z0,
                learning_rate=learning_rate,
                maxiters=maxiters,
            )
        end

        push!(
            diagnostics,
            (
                seed=seed,
                label=label,
                z0_norm=norm(z0),
                adam=adam_result,
            ),
        )
    end

    return diagnostics
end

function forwarddiff_gradient_smoke_test(; z_value=fill(0.0, N_PARAMS))
    setup = square_stress_setup()
    y_true = simulator(true_turning_matrices(), setup)
    loss_fn = z -> map_loss_forwarddiff(z, y_true, setup; prior_scale=1.25, sigma_obs=setup.likelihood_sigma)
    grad = ForwardDiff.gradient(loss_fn, Float64.(collect(z_value)))
    return (
        loss=loss_fn(Float64.(collect(z_value))),
        gradient_norm=norm(grad),
        maxabs_gradient=maximum(abs.(grad)),
    )
end

function run_square_four_to_four_esmda_vs_adam_forwarddiff(;
    rng_seed=42,
    credible_level=0.68,
    prior_scale=1.25,
    ensemble_size=192,
    esmda_maxiters=10,
    prior_draws=96,
    adam_learning_rate=0.04,
    adam_maxiters=300,
    output_dir=FORWARDDIFF_OUTPUT_DIR,
)
    setup = square_stress_setup()
    P_true = true_turning_matrices()
    checks = run_experiment_checks(setup, P_true)
    gradient_smoke = forwarddiff_gradient_smoke_test()

    esmda_results = run_square_four_to_four_esmda(
        setup;
        rng=MersenneTwister(rng_seed),
        prior_scale=prior_scale,
        ensemble_size=ensemble_size,
        esmda_maxiters=esmda_maxiters,
        P_true=P_true,
    )

    prior_param_samples = prior_scale .* randn(MersenneTwister(rng_seed + 1), N_PARAMS, prior_draws)
    prior_weights = fill(1.0 / prior_draws, prior_draws)

    true_state = final_state_snapshot(P_true, setup)
    prior_summary = summarize_final_states(prior_param_samples, setup, prior_weights; level=credible_level)
    posterior_summary = summarize_final_states(esmda_results.param_samples, setup, esmda_results.weights; level=credible_level)
    esmda_state = posterior_summary.mean
    esmda_lower_turning = entry_vector_to_turning_matrices(esmda_results.entry_ci_05; validate=false)
    esmda_upper_turning = entry_vector_to_turning_matrices(esmda_results.entry_ci_95; validate=false)

    adam_results = run_adam_map_forwarddiff(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        learning_rate=adam_learning_rate,
        maxiters=adam_maxiters,
    )
    adam_state = final_state_snapshot(adam_results.z_best, setup)
    adam_restarts = run_adam_restart_diagnostics_forwarddiff(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        learning_rate=adam_learning_rate,
        maxiters=adam_maxiters,
        base_result=adam_results,
    )

    transient_index = max(2, fld(length(setup.control_times), 2))
    transient_time_minutes = 60.0 * setup.control_times[transient_index]
    true_transient_state = state_snapshot(P_true, setup, transient_index)
    esmda_transient_state = state_snapshot(esmda_results.P_post_mean, setup, transient_index)
    adam_transient_state = state_snapshot(adam_results.z_best, setup, transient_index)

    initial_plot_path = joinpath(output_dir, "square_network_external_initial_profiles.png")
    turning_bars_plot_path = joinpath(output_dir, "square_network_turning_fraction_bars.png")
    turning_uncertainty_plot_path = joinpath(output_dir, "square_network_turning_fraction_uncertainty.png")
    final_state_plot_path = joinpath(output_dir, "square_network_final_state_comparison_all_roads.png")
    transient_state_plot_path = joinpath(output_dir, "square_network_transient_state_comparison_all_roads.png")
    adam_plot_path = joinpath(output_dir, "square_network_adam_forwarddiff_diagnostics.png")
    adam_restarts_plot_path = joinpath(output_dir, "square_network_adam_forwarddiff_restart_summary.png")

    initial_plot = plot_external_initial_profiles(setup; output_path=initial_plot_path)
    turning_bars_plot = plot_turning_fraction_bars(
        P_true,
        esmda_results.P_post_mean,
        esmda_lower_turning,
        esmda_upper_turning,
        adam_results.P_est;
        output_path=turning_bars_plot_path,
    )
    turning_uncertainty_plot = plot_turning_fraction_uncertainty_by_junction(
        esmda_results.fraction_samples,
        esmda_results.weights,
        esmda_results.entry_true,
        turning_entries(adam_results.P_est);
        output_path=turning_uncertainty_plot_path,
    )
    final_state_plot = plot_final_state_comparison(
        true_state,
        setup;
        road_ids=collect(1:length(ROAD_LABELS)),
        prior_summary=prior_summary,
        posterior_summary=posterior_summary,
        adam_state=adam_state,
        interval_level=credible_level,
        output_path=final_state_plot_path,
    )
    transient_state_plot = plot_state_snapshot_comparison(
        true_transient_state,
        esmda_transient_state,
        adam_transient_state,
        setup;
        time_label="$(round(transient_time_minutes; digits=2)) min",
        output_path=transient_state_plot_path,
    )
    adam_plot = plot_adam_diagnostics(adam_results; output_path=adam_plot_path)
    adam_restarts_plot = plot_adam_restart_summary(adam_restarts, P_true, esmda_results.y_true; output_path=adam_restarts_plot_path)

    print_setup_summary(setup)
    print_variation_summary(setup, checks.variation)
    print_sensitivity_summary(checks.sensitivity)
    println("ForwardDiff gradient smoke test")
    println("-------------------------------")
    @printf("loss at z = 0: %.4f\n", gradient_smoke.loss)
    @printf("gradient norm at z = 0: %.4f\n", gradient_smoke.gradient_norm)
    @printf("max |gradient| at z = 0: %.4f\n", gradient_smoke.maxabs_gradient)
    println()
    print_method_metrics("ESMDA posterior summary", esmda_results.P_post_mean, esmda_results.y_post_mean, P_true, esmda_results.y_true, esmda_results.y_obs, true_state, esmda_state, esmda_results.solve_seconds, setup.observed_road_ids)
    print_method_metrics("ADAM MAP baseline (ForwardDiff gradient)", adam_results.P_est, adam_results.y_est, P_true, esmda_results.y_true, esmda_results.y_obs, true_state, adam_state, adam_results.solve_seconds, setup.observed_road_ids)
    print_adam_restart_summary(adam_restarts, P_true, esmda_results.y_true)
    print_turning_matrix_comparison(P_true, esmda_results.P_post_mean, adam_results.P_est)
    println("Saved plots:")
    println("  ", initial_plot_path)
    println("  ", turning_bars_plot_path)
    println("  ", turning_uncertainty_plot_path)
    println("  ", final_state_plot_path)
    println("  ", transient_state_plot_path)
    println("  ", adam_plot_path)
    println("  ", adam_restarts_plot_path)

    return (
        setup=setup,
        checks=checks,
        gradient_smoke=gradient_smoke,
        esmda=esmda_results,
        prior_summary=prior_summary,
        posterior_summary=posterior_summary,
        true_state=true_state,
        adam=adam_results,
        adam_restarts=adam_restarts,
        adam_state=adam_state,
        initial_plot=initial_plot,
        turning_bars_plot=turning_bars_plot,
        turning_uncertainty_plot=turning_uncertainty_plot,
        final_state_plot=final_state_plot,
        transient_state_plot=transient_state_plot,
        adam_plot=adam_plot,
        adam_restarts_plot=adam_restarts_plot,
        initial_plot_path=initial_plot_path,
        turning_bars_plot_path=turning_bars_plot_path,
        turning_uncertainty_plot_path=turning_uncertainty_plot_path,
        final_state_plot_path=final_state_plot_path,
        transient_state_plot_path=transient_state_plot_path,
        adam_plot_path=adam_plot_path,
        adam_restarts_plot_path=adam_restarts_plot_path,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_ESMDA_ADAM_FORWARDDIFF_DEMO", "0") != "1"
    square_forwarddiff_results = run_square_four_to_four_esmda_vs_adam_forwarddiff()
end
