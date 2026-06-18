import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Random

include(joinpath(@__DIR__, "..", "common", "four_to_four_method_core.jl"))
include(joinpath(@__DIR__, "..", "common", "four_to_four_state_summaries.jl"))
include(joinpath(@__DIR__, "..", "common", "four_to_four_adam.jl"))
include(joinpath(@__DIR__, "plots.jl"))
include(joinpath(@__DIR__, "..", "common", "four_to_four_reporting.jl"))

const DEFAULT_FIGURE_DIR = joinpath(@__DIR__, "figures")

function run_four_to_four_esmda_vs_adam(;
    rng_seed=42,
    credible_level=0.90,
    ensemble_size=192,
    esmda_maxiters=8,
    adam_learning_rate=0.08,
    adam_maxiters=500,
    figure_dir=DEFAULT_FIGURE_DIR,
)
    apply_four_to_four_esmda_adam_plot_style!()

    setup = default_setup()
    esmda_results = run_four_to_four_inference(
        rng=MersenneTwister(rng_seed),
        setup=setup,
        method=:esmda,
        ensemble_size=ensemble_size,
        esmda_maxiters=esmda_maxiters,
    )

    prior_param_samples = esmda_results.prior_scale .* randn(MersenneTwister(rng_seed + 1), 12, ensemble_size)
    prior_weights = fill(1.0 / ensemble_size, ensemble_size)

    true_state = final_state_snapshot(esmda_results.P_true, setup)
    prior_summary = summarize_final_states(prior_param_samples, setup, prior_weights; level=credible_level)
    posterior_summary = summarize_final_states(esmda_results.param_samples, setup, esmda_results.weights; level=credible_level)
    esmda_state = posterior_summary.mean

    adam_results = run_adam_map_forwarddiff(
        esmda_results.y_obs,
        setup;
        prior_scale=esmda_results.prior_scale,
        sigma_obs=esmda_results.sigma_model,
        learning_rate=adam_learning_rate,
        maxiters=adam_maxiters,
    )
    adam_state = final_state_snapshot(adam_results.z_best, setup)

    esmda_adam_plot_path = joinpath(figure_dir, "four_to_four_esmda_vs_adam_final_state.png")
    coefficient_plot_path = joinpath(figure_dir, "four_to_four_turning_fraction_uncertainty.png")

    esmda_adam_plot = plot_final_state_overview(
        true_state,
        setup;
        prior_summary=prior_summary,
        posterior_summary=posterior_summary,
        adam_state=adam_state,
        interval_level=credible_level,
        output_path=esmda_adam_plot_path,
    )
    coefficient_plot = plot_turning_fraction_uncertainty(
        esmda_results.fraction_samples,
        esmda_results.weights,
        esmda_results.entry_true,
        turning_entries(adam_results.P_est);
        output_path=coefficient_plot_path,
    )

    print_summary(esmda_results)
    println()
    print_variation_summary(setup, esmda_results.y_true)
    println()
    print_adam_summary(adam_results, esmda_results, true_state, esmda_state, adam_state)
    println()
    println("Saved thesis figures:")
    println("  ", esmda_adam_plot_path)
    println("  ", coefficient_plot_path)

    return (
        setup=setup,
        credible_level=credible_level,
        ensemble_size=ensemble_size,
        esmda=esmda_results,
        prior_summary=prior_summary,
        posterior_summary=posterior_summary,
        true_state=true_state,
        adam=adam_results,
        adam_state=adam_state,
        esmda_adam_plot=esmda_adam_plot,
        coefficient_plot=coefficient_plot,
        esmda_adam_plot_path=esmda_adam_plot_path,
        coefficient_plot_path=coefficient_plot_path,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_FOUR_TO_FOUR_ESMDA_ADAM_DEMO", "0") != "1"
    comparison_results = run_four_to_four_esmda_vs_adam()
end
