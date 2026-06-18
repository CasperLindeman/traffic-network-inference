function run_square_four_to_four_esmda_vs_adam_core(;
    rng_seed=42,
    credible_level=0.68,
    prior_scale=1.25,
    ensemble_size=192,
    esmda_maxiters=10,
    prior_draws=96,
    adam_learning_rate=0.04,
    adam_maxiters=300,
    adam_relstep=1e-2,
    adam_absstep=1e-3,
)
    setup = square_stress_setup()
    P_true = true_turning_matrices()
    checks = run_experiment_checks(setup, P_true)

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

    adam_results = run_adam_map(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        learning_rate=adam_learning_rate,
        maxiters=adam_maxiters,
        relstep=adam_relstep,
        absstep=adam_absstep,
    )
    adam_state = final_state_snapshot(adam_results.z_best, setup)
    adam_restarts = run_adam_restart_diagnostics(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        learning_rate=adam_learning_rate,
        maxiters=adam_maxiters,
        relstep=adam_relstep,
        absstep=adam_absstep,
        base_result=adam_results,
    )

    transient_index = max(2, fld(length(setup.control_times), 2))
    transient_time_minutes = 60.0 * setup.control_times[transient_index]
    true_transient_state = state_snapshot(P_true, setup, transient_index)
    esmda_transient_state = state_snapshot(esmda_results.P_post_mean, setup, transient_index)
    adam_transient_state = state_snapshot(adam_results.z_best, setup, transient_index)

    return (
        setup=setup,
        P_true=P_true,
        checks=checks,
        esmda=esmda_results,
        prior_summary=prior_summary,
        posterior_summary=posterior_summary,
        true_state=true_state,
        esmda_state=esmda_state,
        esmda_lower_turning=esmda_lower_turning,
        esmda_upper_turning=esmda_upper_turning,
        adam=adam_results,
        adam_restarts=adam_restarts,
        adam_state=adam_state,
        transient_index=transient_index,
        transient_time_minutes=transient_time_minutes,
        true_transient_state=true_transient_state,
        esmda_transient_state=esmda_transient_state,
        adam_transient_state=adam_transient_state,
        rng_seed=rng_seed,
        credible_level=credible_level,
        prior_scale=prior_scale,
        ensemble_size=ensemble_size,
        esmda_maxiters=esmda_maxiters,
        prior_draws=prior_draws,
        adam_learning_rate=adam_learning_rate,
        adam_maxiters=adam_maxiters,
        adam_relstep=adam_relstep,
        adam_absstep=adam_absstep,
    )
end

function write_square_four_to_four_esmda_vs_adam_figures(core; output_dir=DEFAULT_OUTPUT_DIR)
    setup = core.setup
    P_true = core.P_true
    esmda_results = core.esmda
    adam_results = core.adam

    initial_plot_path = joinpath(output_dir, "square_network_external_initial_profiles.png")
    turning_bars_plot_path = joinpath(output_dir, "square_network_turning_fraction_bars.png")
    turning_uncertainty_plot_path = joinpath(output_dir, "square_network_turning_fraction_uncertainty.png")
    matrices_plot_path = joinpath(output_dir, "square_network_turning_matrices_heatmap.png")
    final_state_plot_path = joinpath(output_dir, "square_network_final_state_comparison_all_roads.png")
    transient_state_plot_path = joinpath(output_dir, "square_network_transient_state_comparison_all_roads.png")
    adam_plot_path = joinpath(output_dir, "square_network_adam_diagnostics.png")
    adam_restarts_plot_path = joinpath(output_dir, "square_network_adam_restart_summary.png")

    initial_plot = plot_external_initial_profiles(setup; output_path=initial_plot_path)
    turning_bars_plot = plot_turning_fraction_bars(
        P_true,
        esmda_results.P_post_mean,
        core.esmda_lower_turning,
        core.esmda_upper_turning,
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
    matrices_plot = plot_turning_matrix_heatmaps(P_true, esmda_results.P_post_mean, adam_results.P_est; output_path=matrices_plot_path)
    final_state_plot = plot_final_state_comparison(
        core.true_state,
        setup;
        road_ids=collect(1:length(ROAD_LABELS)),
        prior_summary=core.prior_summary,
        posterior_summary=core.posterior_summary,
        adam_state=core.adam_state,
        interval_level=core.credible_level,
        output_path=final_state_plot_path,
    )
    transient_state_plot = plot_state_snapshot_comparison(
        core.true_transient_state,
        core.esmda_transient_state,
        core.adam_transient_state,
        setup;
        time_label="$(round(core.transient_time_minutes; digits=2)) min",
        output_path=transient_state_plot_path,
    )
    adam_plot = plot_adam_diagnostics(adam_results; output_path=adam_plot_path)
    adam_restarts_plot = plot_adam_restart_summary(core.adam_restarts, P_true, esmda_results.y_true; output_path=adam_restarts_plot_path)

    return (
        initial_plot=initial_plot,
        turning_bars_plot=turning_bars_plot,
        turning_uncertainty_plot=turning_uncertainty_plot,
        matrices_plot=matrices_plot,
        final_state_plot=final_state_plot,
        transient_state_plot=transient_state_plot,
        adam_plot=adam_plot,
        adam_restarts_plot=adam_restarts_plot,
        initial_plot_path=initial_plot_path,
        turning_bars_plot_path=turning_bars_plot_path,
        turning_uncertainty_plot_path=turning_uncertainty_plot_path,
        matrices_plot_path=matrices_plot_path,
        final_state_plot_path=final_state_plot_path,
        transient_state_plot_path=transient_state_plot_path,
        adam_plot_path=adam_plot_path,
        adam_restarts_plot_path=adam_restarts_plot_path,
    )
end

function print_square_four_to_four_esmda_vs_adam_report(core, figures)
    setup = core.setup
    esmda_results = core.esmda
    adam_results = core.adam

    print_setup_summary(setup)
    print_variation_summary(setup, core.checks.variation)
    print_sensitivity_summary(core.checks.sensitivity)
    print_method_metrics("ESMDA posterior summary", esmda_results.P_post_mean, esmda_results.y_post_mean, core.P_true, esmda_results.y_true, esmda_results.y_obs, core.true_state, core.esmda_state, esmda_results.solve_seconds, setup.observed_road_ids)
    print_method_metrics("ADAM MAP baseline", adam_results.P_est, adam_results.y_est, core.P_true, esmda_results.y_true, esmda_results.y_obs, core.true_state, core.adam_state, adam_results.solve_seconds, setup.observed_road_ids)
    print_adam_restart_summary(core.adam_restarts, core.P_true, esmda_results.y_true)
    print_turning_matrix_comparison(core.P_true, esmda_results.P_post_mean, adam_results.P_est)
    println("Saved plots:")
    println("  ", figures.initial_plot_path)
    println("  ", figures.turning_bars_plot_path)
    println("  ", figures.turning_uncertainty_plot_path)
    println("  ", figures.matrices_plot_path)
    println("  ", figures.final_state_plot_path)
    println("  ", figures.transient_state_plot_path)
    println("  ", figures.adam_plot_path)
    println("  ", figures.adam_restarts_plot_path)
end

function run_square_four_to_four_esmda_vs_adam(; output_dir=DEFAULT_OUTPUT_DIR, kwargs...)
    core = run_square_four_to_four_esmda_vs_adam_core(; kwargs...)
    figures = write_square_four_to_four_esmda_vs_adam_figures(core; output_dir=output_dir)
    print_square_four_to_four_esmda_vs_adam_report(core, figures)
    return merge(core, figures)
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_ESMDA_ADAM_DEMO", "0") != "1"
    square_comparison_results = run_square_four_to_four_esmda_vs_adam()
end
