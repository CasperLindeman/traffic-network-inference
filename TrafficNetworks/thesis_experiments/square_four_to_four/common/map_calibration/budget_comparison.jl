if !isdefined(@__MODULE__, :square_four_to_four_config)
    square_four_to_four_config(name::Symbol, fallback) =
        isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : fallback
end
using Random
using Printf
using LinearAlgebra
using ForwardDiff
using Optim
using Plots

ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_LONGRUN_DEMO"] = "1"
if !isdefined(@__MODULE__, :LONGRUN_OUTPUT_DIR)
    include(joinpath(@__DIR__, "longrun_comparison.jl"))
end

BUDGET_OUTPUT_DIR = square_four_to_four_config(
    :SQUARE_FOUR_TO_FOUR_BUDGET_OUTPUT_DIR,
    joinpath(@__DIR__, "..", "..", "generated", "square_four_to_four", "square_four_to_four_budget_comparison"),
)

function run_adam_map_timed(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.0,
    sigma_obs=0.003,
    z0=zeros(N_PARAMS),
    learning_rate=0.05,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    grad_clip=300.0,
    relstep=1e-2,
    absstep=1e-3,
    time_limit_seconds=30.0,
)
    loss_fn = z -> map_loss(z, y_obs, setup; prior_scale=prior_scale, sigma_obs=sigma_obs)
    z = Float64.(collect(z0))
    best_z = copy(z)
    best_loss = Inf

    m = zeros(length(z))
    v = zeros(length(z))
    losses = Float64[]
    grad_norms = Float64[]
    raw_grad_norms = Float64[]
    iter = 0

    t0 = time()
    while true
        iter += 1
        loss, grad = finite_difference_gradient(loss_fn, z; relstep=relstep, absstep=absstep)
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

        if time() - t0 >= time_limit_seconds
            break
        end
    end

    final_loss = loss_fn(z)
    if final_loss < best_loss
        best_loss = final_loss
        best_z .= z
    end

    return (
        z=copy(z),
        z_best=copy(best_z),
        P_est=turning_matrices(best_z),
        y_est=simulator(best_z, setup),
        best_loss=best_loss,
        losses=losses,
        raw_grad_norms=raw_grad_norms,
        grad_norms=grad_norms,
        solve_seconds=time() - t0,
        iterations=iter,
    )
end

function run_adam_map_forwarddiff_timed(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.0,
    sigma_obs=0.003,
    z0=zeros(N_PARAMS),
    learning_rate=0.05,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    grad_clip=300.0,
    time_limit_seconds=30.0,
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
    iter = 0

    t0 = time()
    while true
        iter += 1
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

        if time() - t0 >= time_limit_seconds
            break
        end
    end

    final_loss = loss_fn(z)
    if final_loss < best_loss
        best_loss = final_loss
        best_z .= z
    end

    return (
        z=copy(z),
        z_best=copy(best_z),
        P_est=turning_matrices(best_z),
        y_est=simulator(best_z, setup),
        best_loss=best_loss,
        losses=losses,
        raw_grad_norms=raw_grad_norms,
        grad_norms=grad_norms,
        solve_seconds=time() - t0,
        iterations=iter,
    )
end

function run_lbfgs_map_forwarddiff_timed(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.0,
    sigma_obs=0.003,
    z0=zeros(N_PARAMS),
    time_limit_seconds=30.0,
    maxiters=10_000,
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

    options = Optim.Options(iterations=maxiters, store_trace=true, show_trace=false, time_limit=time_limit_seconds)
    result = optimize(
        Optim.only_fg!(fg!),
        Float64.(collect(z0)),
        LBFGS(),
        options,
    )

    z_best = Optim.minimizer(result)
    return (
        z=copy(z_best),
        z_best=copy(z_best),
        P_est=turning_matrices(z_best),
        y_est=simulator(z_best, setup),
        best_loss=Optim.minimum(result),
        losses=[tr.value for tr in Optim.trace(result) if tr.value !== nothing],
        solve_seconds=Optim.time_run(result),
        iterations=Optim.iterations(result),
        result=result,
    )
end

function run_square_four_to_four_esmda_fixed_budget(
    setup::SquareFourToFourSetup,
    seed::Int;
    budget_seconds=30.0,
    prior_scale=1.25,
    ensemble_size=192,
    pilot_iters=4,
)
    pilot = run_square_four_to_four_esmda(
        setup;
        rng=MersenneTwister(seed),
        prior_scale=prior_scale,
        ensemble_size=ensemble_size,
        esmda_maxiters=pilot_iters,
        P_true=true_turning_matrices(),
    )
    iter_seconds = pilot.solve_seconds / pilot_iters
    target_iters = max(1, round(Int, budget_seconds / max(iter_seconds, 1e-9)))

    result = run_square_four_to_four_esmda(
        setup;
        rng=MersenneTwister(seed),
        prior_scale=prior_scale,
        ensemble_size=ensemble_size,
        esmda_maxiters=target_iters,
        P_true=true_turning_matrices(),
    )

    return result, target_iters, iter_seconds
end

function budget_metric_row(method::Symbol, budget_seconds::Real, solve_seconds::Real, iterations::Int, P_est, y_est, est_state, true_state, y_true, y_obs, setup; best_loss=NaN)
    observed_ids = setup.observed_road_ids
    unobserved_ids = unobserved_square_road_ids(setup)
    return (
        budget_seconds=Float64(budget_seconds),
        method=String(method),
        solve_seconds=Float64(solve_seconds),
        iterations=iterations,
        turning_rmse=overall_turning_rmse(P_est, true_turning_matrices()),
        predictive_rmse=predictive_rmse(y_est, y_true),
        fit_rmse=predictive_rmse(y_est, y_obs),
        final_state_rmse_all=final_state_rmse(est_state, true_state),
        final_state_rmse_observed=observed_road_state_rmse(est_state, true_state, observed_ids),
        final_state_rmse_unobserved=observed_road_state_rmse(est_state, true_state, unobserved_ids),
        best_loss=Float64(best_loss),
    )
end

function plot_budget_comparison(rows; output_path=nothing)
    budgets = unique(getproperty.(rows, :budget_seconds))
    sort!(budgets)
    methods = [:esmda, :adam_fd, :adam_forwarddiff, :lbfgs_forwarddiff]
    specs = [
        (key=:turning_rmse, title="Turning-fraction RMSE", ylabel="RMSE"),
        (key=:final_state_rmse_all, title="Final-state RMSE (all roads)", ylabel="RMSE"),
        (key=:final_state_rmse_unobserved, title="Final-state RMSE (unobserved roads)", ylabel="RMSE"),
        (key=:solve_seconds, title="Actual solve time", ylabel="Seconds"),
    ]

    plt = plot(layout=(2, 2), size=(1320, 980), dpi=180, legend=:top, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
    x = collect(1:length(budgets))
    xtick_labels = ["$(Int(round(b))) s" for b in budgets]

    for (subplot_id, spec) in enumerate(specs)
        for method in methods
            ys = [
                first(getproperty.(filter(row -> row.budget_seconds == b && row.method == String(method), rows), spec.key))
                for b in budgets
            ]
            plot!(
                plt,
                x,
                ys;
                color=LONGRUN_COLORS[method],
                linewidth=2.4,
                markershape=:circle,
                markersize=6,
                label=subplot_id == 1 ? LONGRUN_LABELS[method] : "",
                subplot=subplot_id,
            )
        end

        plot!(
            plt;
            xlabel="Budget",
            ylabel=spec.ylabel,
            title=spec.title,
            xticks=(x, xtick_labels),
            subplot=subplot_id,
        )
    end

    plot!(plt; plot_title="Square-system budget comparison at one noisy case")

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function run_square_four_to_four_budget_comparison(;
    noise_sigma=0.100,
    seed=1,
    budgets_seconds=(30.0, 120.0),
    prior_scale=1.25,
    ensemble_size=192,
    output_dir=BUDGET_OUTPUT_DIR,
)
    setup = setup_with_matched_noise(square_stress_setup(), noise_sigma)
    P_true = true_turning_matrices()
    true_state = final_state_snapshot(P_true, setup)

    println("Warming up methods on the chosen square case")
    println("--------------------------------------------")
    warm_esmda, _, _ = run_square_four_to_four_esmda_fixed_budget(setup, seed; budget_seconds=5.0, prior_scale=prior_scale, ensemble_size=16, pilot_iters=1)
    warm_y_obs = warm_esmda.y_obs
    run_adam_map_timed(warm_y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    run_adam_map_forwarddiff_timed(warm_y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    run_lbfgs_map_forwarddiff_timed(warm_y_obs, setup; prior_scale=prior_scale, sigma_obs=setup.likelihood_sigma, time_limit_seconds=2.0)
    println()

    rows = NamedTuple[]

    for budget in budgets_seconds
        println(@sprintf("Running budgeted comparison with %.0f-second target", budget))
        esmda, esmda_iters, esmda_iter_seconds = run_square_four_to_four_esmda_fixed_budget(
            setup,
            seed;
            budget_seconds=budget,
            prior_scale=prior_scale,
            ensemble_size=ensemble_size,
        )
        y_obs = esmda.y_obs
        esmda_summary = summarize_final_states(esmda.param_samples, setup, esmda.weights; level=0.90)
        esmda_state = esmda_summary.mean
        push!(
            rows,
            budget_metric_row(
                :esmda,
                budget,
                esmda.solve_seconds,
                esmda_iters,
                esmda.P_post_mean,
                esmda.y_post_mean,
                esmda_state,
                true_state,
                esmda.y_true,
                y_obs,
                setup;
                best_loss=NaN,
            ),
        )
        println(@sprintf("  ESMDA used %d iterations (pilot %.3fs/iter), actual %.2fs", esmda_iters, esmda_iter_seconds, esmda.solve_seconds))

        adam_fd = run_adam_map_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        adam_fd_state = final_state_snapshot(adam_fd.z_best, setup)
        push!(
            rows,
            budget_metric_row(
                :adam_fd,
                budget,
                adam_fd.solve_seconds,
                adam_fd.iterations,
                adam_fd.P_est,
                adam_fd.y_est,
                adam_fd_state,
                true_state,
                esmda.y_true,
                y_obs,
                setup;
                best_loss=adam_fd.best_loss,
            ),
        )
        println(@sprintf("  ADAM finite diff did %d iterations in %.2fs", adam_fd.iterations, adam_fd.solve_seconds))

        adam_forwarddiff = run_adam_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        adam_forwarddiff_state = final_state_snapshot(adam_forwarddiff.z_best, setup)
        push!(
            rows,
            budget_metric_row(
                :adam_forwarddiff,
                budget,
                adam_forwarddiff.solve_seconds,
                adam_forwarddiff.iterations,
                adam_forwarddiff.P_est,
                adam_forwarddiff.y_est,
                adam_forwarddiff_state,
                true_state,
                esmda.y_true,
                y_obs,
                setup;
                best_loss=adam_forwarddiff.best_loss,
            ),
        )
        println(@sprintf("  ADAM ForwardDiff did %d iterations in %.2fs", adam_forwarddiff.iterations, adam_forwarddiff.solve_seconds))

        lbfgs_forwarddiff = run_lbfgs_map_forwarddiff_timed(
            y_obs,
            setup;
            prior_scale=prior_scale,
            sigma_obs=setup.likelihood_sigma,
            time_limit_seconds=budget,
        )
        lbfgs_forwarddiff_state = final_state_snapshot(lbfgs_forwarddiff.z_best, setup)
        push!(
            rows,
            budget_metric_row(
                :lbfgs_forwarddiff,
                budget,
                lbfgs_forwarddiff.solve_seconds,
                lbfgs_forwarddiff.iterations,
                lbfgs_forwarddiff.P_est,
                lbfgs_forwarddiff.y_est,
                lbfgs_forwarddiff_state,
                true_state,
                esmda.y_true,
                y_obs,
                setup;
                best_loss=lbfgs_forwarddiff.best_loss,
            ),
        )
        println(@sprintf("  L-BFGS ForwardDiff did %d iterations in %.2fs", lbfgs_forwarddiff.iterations, lbfgs_forwarddiff.solve_seconds))
        println()
    end

    metrics_path = write_namedtuple_table(rows, joinpath(output_dir, "square_budget_comparison_metrics.tsv"))
    plot_path = joinpath(output_dir, "square_budget_comparison.png")
    comparison_plot = plot_budget_comparison(rows; output_path=plot_path)

    println("Budget comparison summary")
    println("-------------------------")
    for budget in budgets_seconds
        println(@sprintf("Budget %.0fs", budget))
        for row in filter(r -> r.budget_seconds == budget, rows)
            println(
                @sprintf(
                    "  %-20s turning %.4f | full state %.4f | unobserved %.4f | actual %.2fs | iters %d",
                    LONGRUN_LABELS[Symbol(row.method)],
                    row.turning_rmse,
                    row.final_state_rmse_all,
                    row.final_state_rmse_unobserved,
                    row.solve_seconds,
                    row.iterations,
                ),
            )
        end
        println()
    end

    println("Saved outputs:")
    println("  ", metrics_path)
    println("  ", plot_path)

    return (
        setup=setup,
        rows=rows,
        metrics_path=metrics_path,
        plot_path=plot_path,
        comparison_plot=comparison_plot,
    )
end

if get(ENV, "TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_BUDGET_DEMO", "0") != "1"
    budget_results = run_square_four_to_four_budget_comparison()
end
