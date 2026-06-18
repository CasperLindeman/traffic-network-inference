if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using Statistics
using LinearAlgebra
using Plots
using ForwardDiff
using DynamicHMC

import SimulationBasedInference.LogDensityProblems: logdensity, logdensity_and_gradient, capabilities, dimension, LogDensityOrder

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_ESMDA_ADAM_DEMO"] = "1"
ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_ESMDA_ADAM_FORWARDDIFF_DEMO"] = "1"
if !isdefined(@__MODULE__, :simulator_forwarddiff)
    include(joinpath(@__DIR__, "forwarddiff.jl"))
end

LONGRUN_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_LONGRUN_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "generated", "square_four_to_four", "square_four_to_four_longrun_comparison"),
)
const LONGRUN_NOISE_LEVELS = [0.200, 0.100, 0.050, 0.010, 0.005, 0.001]
const LONGRUN_SEEDS = [1, 2]
const LONGRUN_METHODS = (:esmda, :adam_fd, :adam_forwarddiff, :lbfgs_forwarddiff)
const LONGRUN_COLORS = Dict(
    :esmda => :steelblue,
    :adam_fd => :firebrick,
    :adam_forwarddiff => :seagreen4,
    :lbfgs_forwarddiff => :darkorange2,
    :nuts_forwarddiff => :mediumpurple3,
)
const LONGRUN_LABELS = Dict(
    :esmda => "ESMDA",
    :adam_fd => "ADAM finite diff",
    :adam_forwarddiff => "ADAM ForwardDiff",
    :lbfgs_forwarddiff => "L-BFGS ForwardDiff",
    :nuts_forwarddiff => "NUTS ForwardDiff",
)

function setup_with_matched_noise(base_setup::SquareFourToFourSetup, noise_sigma::Real)
    sigma = Float64(noise_sigma)
    @assert sigma > 0.0

    return SquareFourToFourSetup(
        base_setup.T,
        base_setup.CFL,
        base_setup.n_cells,
        copy(base_setup.control_times),
        copy(base_setup.observed_road_ids),
        copy(base_setup.observed_cell_ids),
        copy(base_setup.boundary_road_ids),
        copy(base_setup.inflows),
        copy(base_setup.road_profiles),
        copy(base_setup.road_lengths),
        copy(base_setup.speed_limits),
        sigma,
        sigma,
    )
end

all_square_road_ids() = collect(1:length(ROAD_LABELS))
unobserved_square_road_ids(setup::SquareFourToFourSetup) = setdiff(all_square_road_ids(), setup.observed_road_ids)
latent_slice(z::AbstractVector) = @view z[1:N_PARAMS]

function final_state_interval_width(summary::FinalStateSummary, road_ids::AbstractVector{Int})
    return mean(summary.upper[:, road_ids] .- summary.lower[:, road_ids])
end

function final_state_interval_coverage(summary::FinalStateSummary, true_state::AbstractMatrix, road_ids::AbstractVector{Int})
    mask = (true_state[:, road_ids] .>= summary.lower[:, road_ids]) .& (true_state[:, road_ids] .<= summary.upper[:, road_ids])
    return mean(mask)
end

turning_interval_width(lower::AbstractVector, upper::AbstractVector) = mean(upper .- lower)

function turning_interval_coverage(true_values::AbstractVector, lower::AbstractVector, upper::AbstractVector)
    return mean((true_values .>= lower) .& (true_values .<= upper))
end

function write_namedtuple_table(rows, output_path)
    isempty(rows) && return nothing
    header = collect(keys(first(rows)))
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(io, join(string.(header), '\t'))
        for row in rows
            vals = [
                value isa AbstractFloat ? @sprintf("%.8f", value) : string(value)
                for value in (getproperty(row, key) for key in header)
            ]
            println(io, join(vals, '\t'))
        end
    end

    return output_path
end

function metric_row(
    method::Symbol,
    noise_sigma::Real,
    seed::Int,
    P_est,
    y_est,
    est_state,
    true_state,
    y_true,
    y_obs,
    setup::SquareFourToFourSetup,
    solve_seconds::Real;
    best_loss=NaN,
    turning_interval_width_val=NaN,
    turning_interval_coverage_val=NaN,
    final_state_interval_width_all=NaN,
    final_state_interval_coverage_all=NaN,
    final_state_interval_width_observed=NaN,
    final_state_interval_coverage_observed=NaN,
    final_state_interval_width_unobserved=NaN,
    final_state_interval_coverage_unobserved=NaN,
)
    observed_ids = setup.observed_road_ids
    unobserved_ids = unobserved_square_road_ids(setup)

    return (
        noise_sigma=Float64(noise_sigma),
        seed=seed,
        method=String(method),
        turning_rmse=overall_turning_rmse(P_est, true_turning_matrices()),
        predictive_rmse=predictive_rmse(y_est, y_true),
        fit_rmse=predictive_rmse(y_est, y_obs),
        final_state_rmse_all=final_state_rmse(est_state, true_state),
        final_state_rmse_observed=observed_road_state_rmse(est_state, true_state, observed_ids),
        final_state_rmse_unobserved=observed_road_state_rmse(est_state, true_state, unobserved_ids),
        turning_interval_width=turning_interval_width_val,
        turning_interval_coverage=turning_interval_coverage_val,
        final_state_interval_width_all=final_state_interval_width_all,
        final_state_interval_coverage_all=final_state_interval_coverage_all,
        final_state_interval_width_observed=final_state_interval_width_observed,
        final_state_interval_coverage_observed=final_state_interval_coverage_observed,
        final_state_interval_width_unobserved=final_state_interval_width_unobserved,
        final_state_interval_coverage_unobserved=final_state_interval_coverage_unobserved,
        solve_seconds=Float64(solve_seconds),
        best_loss=Float64(best_loss),
    )
end

function run_square_noise_case_long(
    noise_sigma::Real,
    seed::Int;
    credible_level=0.90,
    prior_scale=1.25,
    ensemble_size=192,
    esmda_maxiters=18,
    adam_fd_learning_rate=0.04,
    adam_fd_maxiters=600,
    adam_fd_relstep=1e-2,
    adam_fd_absstep=1e-3,
    adam_forwarddiff_learning_rate=0.04,
    adam_forwarddiff_maxiters=600,
    lbfgs_forwarddiff_maxiters=220,
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    P_true = true_turning_matrices()

    esmda_results = run_square_four_to_four_esmda(
        setup;
        rng=MersenneTwister(seed),
        prior_scale=prior_scale,
        ensemble_size=ensemble_size,
        esmda_maxiters=esmda_maxiters,
        P_true=P_true,
    )

    true_state = final_state_snapshot(P_true, setup)
    posterior_summary = summarize_final_states(esmda_results.param_samples, setup, esmda_results.weights; level=credible_level)
    esmda_state = posterior_summary.mean

    rows = NamedTuple[]
    push!(
        rows,
        metric_row(
            :esmda,
            noise_sigma,
            seed,
            esmda_results.P_post_mean,
            esmda_results.y_post_mean,
            esmda_state,
            true_state,
            esmda_results.y_true,
            esmda_results.y_obs,
            setup,
            esmda_results.solve_seconds;
            turning_interval_width_val=turning_interval_width(esmda_results.entry_ci_05, esmda_results.entry_ci_95),
            turning_interval_coverage_val=turning_interval_coverage(esmda_results.entry_true, esmda_results.entry_ci_05, esmda_results.entry_ci_95),
            final_state_interval_width_all=final_state_interval_width(posterior_summary, all_square_road_ids()),
            final_state_interval_coverage_all=final_state_interval_coverage(posterior_summary, true_state, all_square_road_ids()),
            final_state_interval_width_observed=final_state_interval_width(posterior_summary, setup.observed_road_ids),
            final_state_interval_coverage_observed=final_state_interval_coverage(posterior_summary, true_state, setup.observed_road_ids),
            final_state_interval_width_unobserved=final_state_interval_width(posterior_summary, unobserved_square_road_ids(setup)),
            final_state_interval_coverage_unobserved=final_state_interval_coverage(posterior_summary, true_state, unobserved_square_road_ids(setup)),
        ),
    )

    adam_fd = run_adam_map(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        learning_rate=adam_fd_learning_rate,
        maxiters=adam_fd_maxiters,
        relstep=adam_fd_relstep,
        absstep=adam_fd_absstep,
    )
    adam_fd_state = final_state_snapshot(adam_fd.z_best, setup)
    push!(
        rows,
        metric_row(
            :adam_fd,
            noise_sigma,
            seed,
            adam_fd.P_est,
            adam_fd.y_est,
            adam_fd_state,
            true_state,
            esmda_results.y_true,
            esmda_results.y_obs,
            setup,
            adam_fd.solve_seconds;
            best_loss=adam_fd.best_loss,
        ),
    )

    adam_forwarddiff = run_adam_map_forwarddiff(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        learning_rate=adam_forwarddiff_learning_rate,
        maxiters=adam_forwarddiff_maxiters,
    )
    adam_forwarddiff_state = final_state_snapshot(adam_forwarddiff.z_best, setup)
    push!(
        rows,
        metric_row(
            :adam_forwarddiff,
            noise_sigma,
            seed,
            adam_forwarddiff.P_est,
            adam_forwarddiff.y_est,
            adam_forwarddiff_state,
            true_state,
            esmda_results.y_true,
            esmda_results.y_obs,
            setup,
            adam_forwarddiff.solve_seconds;
            best_loss=adam_forwarddiff.best_loss,
        ),
    )

    lbfgs_forwarddiff = run_lbfgs_map_forwarddiff(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        sigma_obs=setup.likelihood_sigma,
        maxiters=lbfgs_forwarddiff_maxiters,
    )
    lbfgs_forwarddiff_state = final_state_snapshot(lbfgs_forwarddiff.z_best, setup)
    push!(
        rows,
        metric_row(
            :lbfgs_forwarddiff,
            noise_sigma,
            seed,
            lbfgs_forwarddiff.P_est,
            lbfgs_forwarddiff.y_est,
            lbfgs_forwarddiff_state,
            true_state,
            esmda_results.y_true,
            esmda_results.y_obs,
            setup,
            lbfgs_forwarddiff.solve_seconds;
            best_loss=lbfgs_forwarddiff.best_loss,
        ),
    )

    details = (
        setup=setup,
        P_true=P_true,
        true_state=true_state,
        esmda=esmda_results,
        posterior_summary=posterior_summary,
        adam_fd=adam_fd,
        adam_forwarddiff=adam_forwarddiff,
        lbfgs_forwarddiff=lbfgs_forwarddiff,
    )

    return rows, details
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

function build_square_inference_problem_forwarddiff(setup::SquareFourToFourSetup, y_obs::AbstractVector; prior_scale=1.25)
    y_template = simulator(true_turning_matrices(), setup)
    obs = SimulatorObservable(:y, state -> state.u, (length(y_template),))
    forward_prob = SimulatorForwardProblem(p -> simulator_forwarddiff(p, setup), zeros(N_PARAMS), obs)
    model_prior = build_model_prior(prior_scale)
    noise_prior = prior(sigma=LogNormal(log(setup.likelihood_sigma), 0.05))
    lik = SimulatorLikelihood(IsoNormal, obs, y_obs, noise_prior, :y)
    return SimulatorInferenceProblem(forward_prob, nothing, model_prior, lik)
end

function simulate_posterior_draws_square(param_samples::AbstractMatrix, setup::SquareFourToFourSetup)
    y_samples = Matrix{Float64}(undef, observation_length(setup), size(param_samples, 2))
    for j in 1:size(param_samples, 2)
        y_samples[:, j] = simulator(view(param_samples, :, j), setup)
    end
    return y_samples
end

function run_square_nuts_forwarddiff(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.25,
    rng=MersenneTwister(42),
    num_samples=40,
    warmup_stages=DynamicHMC.default_warmup_stages(
        init_steps=20,
        middle_steps=25,
        doubling_stages=3,
        terminating_steps=25,
    ),
)
    inference_prob = build_square_inference_problem_forwarddiff(setup, y_obs; prior_scale=prior_scale)
    b = SimulationBasedInference.bijector(inference_prob)
    b_inv = SimulationBasedInference.inverse(b)
    wrapped = ForwardDiffGradientLogDensity(logdensity(inference_prob))
    initialization = (; q=b(rand(rng, inference_prob.prior)))

    solve_seconds = @elapsed begin
        warmup = DynamicHMC.mcmc_keep_warmup(
            rng,
            wrapped,
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

        full_samples = reduce(hcat, map(q -> collect(b_inv(q)), eachcol(q_samples)))
    end

    latent_samples = full_samples[1:N_PARAMS, :]
    weights = fill(1.0 / size(latent_samples, 2), size(latent_samples, 2))
    fraction_samples = turning_entry_samples(latent_samples)
    entry_true = turning_entries(true_turning_matrices())
    entry_post_mean, entry_post_median, entry_ci_05, entry_ci_95 = sample_summary(fraction_samples, weights)
    P_post_mean = entry_vector_to_turning_matrices(entry_post_mean)
    y_samples = simulate_posterior_draws_square(latent_samples, setup)
    y_post_mean = weighted_column_mean(y_samples, weights)
    posterior_summary = summarize_final_states(latent_samples, setup, weights; level=0.90)

    return (
        solution=warmup,
        param_samples=latent_samples,
        full_samples=full_samples,
        weights=weights,
        fraction_samples=fraction_samples,
        entry_true=entry_true,
        entry_post_mean=entry_post_mean,
        entry_post_median=entry_post_median,
        entry_ci_05=entry_ci_05,
        entry_ci_95=entry_ci_95,
        P_post_mean=P_post_mean,
        y_post_mean=y_post_mean,
        posterior_summary=posterior_summary,
        solve_seconds=solve_seconds,
        num_samples=num_samples,
        posterior_sigma_mean=mean(full_samples[end, :]),
        posterior_sigma_std=std(full_samples[end, :]),
    )
end

function metric_values(case_metrics, noise_sigma, method::Symbol, key::Symbol)
    method_name = String(method)
    return [
        Float64(getproperty(row, key))
        for row in case_metrics
        if isapprox(row.noise_sigma, noise_sigma; atol=1e-12) && row.method == method_name
    ]
end

function aggregate_metric(case_metrics, noise_levels, method::Symbol, key::Symbol)
    means = Float64[]
    lowers = Float64[]
    uppers = Float64[]

    for sigma in noise_levels
        vals = metric_values(case_metrics, sigma, method, key)
        push!(means, mean(vals))
        push!(lowers, minimum(vals))
        push!(uppers, maximum(vals))
    end

    return (mean=means, lower=lowers, upper=uppers)
end

function plot_longrun_method_comparison(case_metrics, noise_levels; output_path=nothing)
    specs = [
        (key=:turning_rmse, title="Turning-fraction RMSE", ylabel="RMSE"),
        (key=:final_state_rmse_all, title="Final-state RMSE (all roads)", ylabel="RMSE"),
        (key=:final_state_rmse_unobserved, title="Final-state RMSE (unobserved roads)", ylabel="RMSE"),
        (key=:solve_seconds, title="Solve time", ylabel="Seconds"),
    ]
    x = collect(1:length(noise_levels))
    xtick_labels = [@sprintf("%.3f", sigma) for sigma in noise_levels]
    plt = plot(layout=(2, 2), size=(1320, 980), dpi=180, legend=:top, left_margin=10Plots.mm, bottom_margin=8Plots.mm)

    for (subplot_id, spec) in enumerate(specs)
        for method in LONGRUN_METHODS
            agg = aggregate_metric(case_metrics, noise_levels, method, spec.key)
            plot!(
                plt,
                x,
                agg.mean;
                ribbon=(agg.mean .- agg.lower, agg.upper .- agg.mean),
                color=LONGRUN_COLORS[method],
                fillalpha=0.12,
                linewidth=2.3,
                markershape=:circle,
                markersize=5,
                label=subplot_id == 1 ? LONGRUN_LABELS[method] : "",
                subplot=subplot_id,
            )
        end

        plot!(
            plt;
            xlabel="Matched observation noise sigma",
            ylabel=spec.ylabel,
            title=spec.title,
            xticks=(x, xtick_labels),
            subplot=subplot_id,
        )
    end

    plot!(plt; plot_title="Square-system long-run method comparison")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function summarize_method_level(case_metrics, noise_levels)
    rows = NamedTuple[]
    for sigma in noise_levels, method in LONGRUN_METHODS
        vals_turning = metric_values(case_metrics, sigma, method, :turning_rmse)
        vals_all = metric_values(case_metrics, sigma, method, :final_state_rmse_all)
        vals_unobs = metric_values(case_metrics, sigma, method, :final_state_rmse_unobserved)
        vals_seconds = metric_values(case_metrics, sigma, method, :solve_seconds)
        push!(
            rows,
            (
                noise_sigma=Float64(sigma),
                method=String(method),
                n_cases=length(vals_turning),
                turning_rmse_mean=mean(vals_turning),
                turning_rmse_min=minimum(vals_turning),
                turning_rmse_max=maximum(vals_turning),
                final_state_rmse_all_mean=mean(vals_all),
                final_state_rmse_all_min=minimum(vals_all),
                final_state_rmse_all_max=maximum(vals_all),
                final_state_rmse_unobserved_mean=mean(vals_unobs),
                final_state_rmse_unobserved_min=minimum(vals_unobs),
                final_state_rmse_unobserved_max=maximum(vals_unobs),
                solve_seconds_mean=mean(vals_seconds),
                solve_seconds_min=minimum(vals_seconds),
                solve_seconds_max=maximum(vals_seconds),
            ),
        )
    end
    return rows
end

function best_method_rows(summary_rows, noise_levels)
    rows = NamedTuple[]
    for sigma in noise_levels
        sigma_rows = filter(row -> isapprox(row.noise_sigma, sigma; atol=1e-12), summary_rows)
        best_turning = sigma_rows[argmin(getproperty.(sigma_rows, :turning_rmse_mean))].method
        best_all = sigma_rows[argmin(getproperty.(sigma_rows, :final_state_rmse_all_mean))].method
        best_unobs = sigma_rows[argmin(getproperty.(sigma_rows, :final_state_rmse_unobserved_mean))].method
        fastest = sigma_rows[argmin(getproperty.(sigma_rows, :solve_seconds_mean))].method
        push!(
            rows,
            (
                noise_sigma=Float64(sigma),
                best_turning_rmse=best_turning,
                best_full_state_rmse=best_all,
                best_unobserved_state_rmse=best_unobs,
                fastest_method=fastest,
            ),
        )
    end
    return rows
end

function print_longrun_summary(summary_rows, best_rows)
    println("Square-system long-run comparison")
    println("---------------------------------")
    for best_row in best_rows
        sigma = best_row.noise_sigma
        sigma_rows = filter(row -> isapprox(row.noise_sigma, sigma; atol=1e-12), summary_rows)
        println(@sprintf("sigma = %.4f", sigma))
        for row in sigma_rows
            println(
                @sprintf(
                    "  %-20s turning %.4f | full state %.4f | unobserved %.4f | time %.2fs",
                    LONGRUN_LABELS[Symbol(row.method)],
                    row.turning_rmse_mean,
                    row.final_state_rmse_all_mean,
                    row.final_state_rmse_unobserved_mean,
                    row.solve_seconds_mean,
                ),
            )
        end
        println("  best turning     = ", best_row.best_turning_rmse)
        println("  best full state  = ", best_row.best_full_state_rmse)
        println("  best unobserved  = ", best_row.best_unobserved_state_rmse)
        println("  fastest          = ", best_row.fastest_method)
        println()
    end
end

function run_square_four_to_four_longrun_comparison(;
    noise_levels=LONGRUN_NOISE_LEVELS,
    seeds=LONGRUN_SEEDS,
    credible_level=0.90,
    prior_scale=1.25,
    ensemble_size=192,
    esmda_maxiters=18,
    adam_fd_learning_rate=0.04,
    adam_fd_maxiters=600,
    adam_fd_relstep=1e-2,
    adam_fd_absstep=1e-3,
    adam_forwarddiff_learning_rate=0.04,
    adam_forwarddiff_maxiters=600,
    lbfgs_forwarddiff_maxiters=220,
    output_dir=LONGRUN_OUTPUT_DIR,
)
    case_metrics = NamedTuple[]
    case_details = Dict{Tuple{Float64, Int}, NamedTuple}()

    for sigma in noise_levels
        for seed in seeds
            println(@sprintf("Running square long-run case: sigma = %.4f, seed = %d", sigma, seed))
            rows, details = run_square_noise_case_long(
                sigma,
                seed;
                credible_level=credible_level,
                prior_scale=prior_scale,
                ensemble_size=ensemble_size,
                esmda_maxiters=esmda_maxiters,
                adam_fd_learning_rate=adam_fd_learning_rate,
                adam_fd_maxiters=adam_fd_maxiters,
                adam_fd_relstep=adam_fd_relstep,
                adam_fd_absstep=adam_fd_absstep,
                adam_forwarddiff_learning_rate=adam_forwarddiff_learning_rate,
                adam_forwarddiff_maxiters=adam_forwarddiff_maxiters,
                lbfgs_forwarddiff_maxiters=lbfgs_forwarddiff_maxiters,
            )
            append!(case_metrics, rows)
            case_details[(Float64(sigma), seed)] = details
        end
    end

    summary_rows = summarize_method_level(case_metrics, noise_levels)
    best_rows = best_method_rows(summary_rows, noise_levels)

    case_metrics_path = write_namedtuple_table(case_metrics, joinpath(output_dir, "square_longrun_case_metrics.tsv"))
    summary_path = write_namedtuple_table(summary_rows, joinpath(output_dir, "square_longrun_summary_metrics.tsv"))
    best_path = write_namedtuple_table(best_rows, joinpath(output_dir, "square_longrun_best_methods.tsv"))
    comparison_plot = plot_longrun_method_comparison(case_metrics, noise_levels; output_path=joinpath(output_dir, "square_longrun_method_comparison.png"))

    print_longrun_summary(summary_rows, best_rows)
    println("Saved tables and plots:")
    println("  ", case_metrics_path)
    println("  ", summary_path)
    println("  ", best_path)
    println("  ", joinpath(output_dir, "square_longrun_method_comparison.png"))

    return (
        case_metrics=case_metrics,
        case_details=case_details,
        summary_rows=summary_rows,
        best_rows=best_rows,
        comparison_plot=comparison_plot,
        case_metrics_path=case_metrics_path,
        summary_path=summary_path,
        best_path=best_path,
    )
end

function run_square_four_to_four_nuts_viability(;
    noise_sigma=0.010,
    seed=1,
    prior_scale=1.25,
    num_samples=30,
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    esmda_results = run_square_four_to_four_esmda(
        setup;
        rng=MersenneTwister(seed),
        prior_scale=prior_scale,
        ensemble_size=192,
        esmda_maxiters=18,
        P_true=true_turning_matrices(),
    )
    nuts_results = run_square_nuts_forwarddiff(
        esmda_results.y_obs,
        setup;
        prior_scale=prior_scale,
        rng=MersenneTwister(10_000 + seed),
        num_samples=num_samples,
    )

    true_state = final_state_snapshot(true_turning_matrices(), setup)
    nuts_state = nuts_results.posterior_summary.mean
    metrics = metric_row(
        :nuts_forwarddiff,
        noise_sigma,
        seed,
        nuts_results.P_post_mean,
        nuts_results.y_post_mean,
        nuts_state,
        true_state,
        esmda_results.y_true,
        esmda_results.y_obs,
        setup,
        nuts_results.solve_seconds;
        turning_interval_width_val=turning_interval_width(nuts_results.entry_ci_05, nuts_results.entry_ci_95),
        turning_interval_coverage_val=turning_interval_coverage(nuts_results.entry_true, nuts_results.entry_ci_05, nuts_results.entry_ci_95),
        final_state_interval_width_all=final_state_interval_width(nuts_results.posterior_summary, all_square_road_ids()),
        final_state_interval_coverage_all=final_state_interval_coverage(nuts_results.posterior_summary, true_state, all_square_road_ids()),
        final_state_interval_width_observed=final_state_interval_width(nuts_results.posterior_summary, setup.observed_road_ids),
        final_state_interval_coverage_observed=final_state_interval_coverage(nuts_results.posterior_summary, true_state, setup.observed_road_ids),
        final_state_interval_width_unobserved=final_state_interval_width(nuts_results.posterior_summary, unobserved_square_road_ids(setup)),
        final_state_interval_coverage_unobserved=final_state_interval_coverage(nuts_results.posterior_summary, true_state, unobserved_square_road_ids(setup)),
    )

    println("Square-system NUTS viability")
    println("----------------------------")
    println(@sprintf("noise sigma: %.4f", noise_sigma))
    println(@sprintf("solve time: %.2f s", metrics.solve_seconds))
    println(@sprintf("turning RMSE: %.4f", metrics.turning_rmse))
    println(@sprintf("predictive RMSE: %.4f", metrics.predictive_rmse))
    println(@sprintf("full final-state RMSE: %.4f", metrics.final_state_rmse_all))
    println(@sprintf("unobserved final-state RMSE: %.4f", metrics.final_state_rmse_unobserved))
    println(@sprintf("posterior sigma mean: %.5f", nuts_results.posterior_sigma_mean))
    println(@sprintf("posterior sigma std: %.5f", nuts_results.posterior_sigma_std))
    println("posterior draws: ", nuts_results.num_samples)
    println()

    return (
        setup=setup,
        esmda=esmda_results,
        nuts=nuts_results,
        metrics=metrics,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_LONGRUN_DEMO", "0") != "1"
    longrun_results = run_square_four_to_four_longrun_comparison()
end
