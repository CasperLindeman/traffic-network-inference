function finite_difference_gradient(loss_fn, z; relstep=1e-2, absstep=1e-3)
    base_loss = loss_fn(z)
    grad = Vector{Float64}(undef, length(z))
    z_plus = collect(z)

    for i in eachindex(z)
        zi = z[i]
        h = max(absstep, relstep * max(abs(zi), 1.0))
        z_plus[i] = zi + h
        grad[i] = (loss_fn(z_plus) - base_loss) / h
        z_plus[i] = zi
    end

    return base_loss, grad
end

function build_model_prior(prior_scale)
    dists = ntuple(_ -> Normal(0.0, prior_scale), length(LATENT_NAMES))
    kwargs = NamedTuple{LATENT_NAMES}(dists)
    return prior(; kwargs...)
end

function map_loss(z, y_obs, setup::SquareFourToFourSetup; prior_scale=1.0, sigma_obs=0.003)
    y_pred = simulator(z, setup)
    residual = (y_pred .- y_obs) ./ sigma_obs
    prior_term = z ./ prior_scale
    return 0.5 * sum(residual .^ 2) + 0.5 * sum(prior_term .^ 2)
end

function run_adam_map(
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
    relstep=1e-2,
    absstep=1e-3,
    grad_clip=300.0,
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

    solve_seconds = @elapsed begin
        for iter in 1:maxiters
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

function run_adam_restart_diagnostics(
    y_obs::AbstractVector,
    setup::SquareFourToFourSetup;
    prior_scale=1.0,
    sigma_obs=0.003,
    seeds=0:3,
    init_scale=0.8,
    learning_rate=0.05,
    maxiters=120,
    relstep=1e-2,
    absstep=1e-3,
    base_result=nothing,
)
    diagnostics = NamedTuple[]

    for seed in seeds
        label = seed == 0 ? "prior mean" : "random $(seed)"
        z0 = seed == 0 ? zeros(N_PARAMS) : init_scale .* randn(MersenneTwister(100 + seed), N_PARAMS)
        adam_result = if seed == 0 && base_result !== nothing
            base_result
        else
            run_adam_map(
                y_obs,
                setup;
                prior_scale=prior_scale,
                sigma_obs=sigma_obs,
                z0=z0,
                learning_rate=learning_rate,
                maxiters=maxiters,
                relstep=relstep,
                absstep=absstep,
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

function run_square_four_to_four_esmda(
    setup::SquareFourToFourSetup;
    rng=MersenneTwister(42),
    prior_scale=1.25,
    ensemble_size=192,
    esmda_maxiters=10,
    P_true=true_turning_matrices(),
)
    y_true = simulator(P_true, setup)
    y_obs = if setup.generated_noise_sigma > 0
        y_true .+ setup.generated_noise_sigma .* randn(rng, length(y_true))
    else
        copy(y_true)
    end

    obs = SimulatorObservable(:y, state -> state.u, (length(y_true),))
    forward_prob = SimulatorForwardProblem(p -> simulator(p, setup), zeros(N_PARAMS), obs)
    model_prior = build_model_prior(prior_scale)
    noise_prior = prior(sigma=LogNormal(log(setup.likelihood_sigma), 0.05))
    lik = SimulatorLikelihood(IsoNormal, obs, y_obs, noise_prior, :y)
    inference_prob = SimulatorInferenceProblem(forward_prob, nothing, model_prior, lik)

    solve_seconds = @elapsed sol = solve(
        inference_prob,
        ESMDA(maxiters=esmda_maxiters),
        ensemble_size=ensemble_size,
        rng=rng,
        verbose=false,
    )

    param_samples = Array(get_transformed_ensemble(sol))
    weights = fill(1.0 / size(param_samples, 2), size(param_samples, 2))
    fraction_samples = turning_entry_samples(param_samples)
    entry_true = turning_entries(P_true)
    entry_post_mean, entry_post_median, entry_ci_05, entry_ci_95 = sample_summary(fraction_samples, weights)
    P_post_mean = entry_vector_to_turning_matrices(entry_post_mean)

    return (
        setup=setup,
        prior_scale=prior_scale,
        P_true=turning_matrices(P_true),
        entry_true=entry_true,
        y_true=y_true,
        y_obs=y_obs,
        solution=sol,
        weights=weights,
        param_samples=param_samples,
        fraction_samples=fraction_samples,
        entry_post_mean=entry_post_mean,
        entry_post_median=entry_post_median,
        entry_ci_05=entry_ci_05,
        entry_ci_95=entry_ci_95,
        P_post_mean=P_post_mean,
        y_post_mean=posterior_predictive_mean(sol, weights),
        solve_seconds=solve_seconds,
    )
end

overall_turning_rmse(P_est, P_true) = sqrt(mean((turning_entries(P_est) .- turning_entries(P_true)) .^ 2))
