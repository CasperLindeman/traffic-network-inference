# Shared helpers for square-four-to-four single-scenario experiments.

function map_loss_forwarddiff(
    z,
    y_obs::AbstractVector,
    setup::SquareSingleScenarioSetup,
    sigma_model::AbstractVector;
    prior_scale=DEFAULT_PRIOR_SCALE,
)
    residual = (simulator_forwarddiff(z, setup) .- y_obs) ./ sigma_model
    prior_term = z ./ prior_scale
    return 0.5 * sum(abs2, residual) + 0.5 * sum(abs2, prior_term)
end

function mse_loss_forwarddiff(z, y_obs::AbstractVector, setup::SquareSingleScenarioSetup)
    residual = simulator_forwarddiff(z, setup) .- y_obs
    return 0.5 * sum(abs2, residual)
end

function build_single_scenario_loss(
    loss_kind::Symbol,
    y_obs::AbstractVector,
    setup::SquareSingleScenarioSetup;
    sigma_model=nothing,
    prior_scale=DEFAULT_PRIOR_SCALE,
)
    if loss_kind == :map
        @assert sigma_model !== nothing "MAP loss requires sigma_model."
        return z -> map_loss_forwarddiff(z, y_obs, setup, sigma_model; prior_scale=prior_scale)
    elseif loss_kind == :mse
        return z -> mse_loss_forwarddiff(z, y_obs, setup)
    end

    error("Unknown loss kind $(loss_kind).")
end

function cosine_decay_learning_rate(iteration::Int, maxiters::Int, initial_lr::Real; decay_start=0.65, final_lr_scale=0.02)
    if maxiters <= 1 || decay_start >= 1.0
        return Float64(initial_lr)
    end

    progress = (iteration - 1) / (maxiters - 1)
    if progress <= decay_start
        return Float64(initial_lr)
    end

    decay_progress = (progress - decay_start) / (1.0 - decay_start)
    scale = final_lr_scale + 0.5 * (1.0 - final_lr_scale) * (1.0 + cos(pi * decay_progress))
    return Float64(initial_lr) * scale
end

function tail_relative_span(values::AbstractVector; window=40)
    isempty(values) && return NaN
    tail = values[max(1, end - window + 1):end]
    denom = max(abs(mean(tail)), 1e-12)
    return (maximum(tail) - minimum(tail)) / denom
end

function run_adam_forwarddiff(
    loss_fn::Function;
    z0=zeros(N_PARAMS),
    learning_rate=0.02,
    maxiters=320,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    grad_clip=Inf,
    decay_start=0.65,
    final_lr_scale=0.02,
)
    z = Float64.(collect(z0))
    best_z = copy(z)
    best_loss = Inf
    best_iter = 0

    m = zeros(length(z))
    v = zeros(length(z))
    grad = zeros(length(z))

    losses = Float64[]
    raw_grad_norms = Float64[]
    grad_norms = Float64[]
    learning_rates = Float64[]
    clipped_flags = Bool[]

    cfg = ForwardDiff.GradientConfig(loss_fn, z)
    terminated_early = false

    solve_seconds = @elapsed begin
        for iter in 1:maxiters
            loss = loss_fn(z)
            ForwardDiff.gradient!(grad, loss_fn, z, cfg)

            if !isfinite(loss) || !all(isfinite, grad)
                terminated_early = true
                break
            end

            raw_grad_norm = norm(grad)
            grad_norm = raw_grad_norm
            clipped = isfinite(grad_clip) && grad_norm > grad_clip

            if clipped
                grad .*= grad_clip / grad_norm
                grad_norm = grad_clip
            end

            push!(losses, loss)
            push!(raw_grad_norms, raw_grad_norm)
            push!(grad_norms, grad_norm)
            push!(clipped_flags, clipped)

            if loss < best_loss
                best_loss = loss
                best_z .= z
                best_iter = iter
            end

            lr = cosine_decay_learning_rate(iter, maxiters, learning_rate; decay_start=decay_start, final_lr_scale=final_lr_scale)
            push!(learning_rates, lr)

            m .= beta1 .* m .+ (1.0 - beta1) .* grad
            v .= beta2 .* v .+ (1.0 - beta2) .* (grad .^ 2)

            m_hat = m ./ (1.0 - beta1^iter)
            v_hat = v ./ (1.0 - beta2^iter)
            z .-= lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
        end
    end

    final_loss = loss_fn(z)
    final_grad = ForwardDiff.gradient(loss_fn, z)
    final_raw_grad_norm = all(isfinite, final_grad) && isfinite(final_loss) ? norm(final_grad) : Inf
    final_postclip_grad_norm = isfinite(grad_clip) ? min(final_raw_grad_norm, grad_clip) : final_raw_grad_norm

    if final_loss < best_loss
        best_loss = final_loss
        best_z .= z
        best_iter = maxiters
    end

    return (
        z=copy(z),
        z_best=copy(best_z),
        best_loss=best_loss,
        best_iter=best_iter,
        final_loss=final_loss,
        final_raw_grad_norm=final_raw_grad_norm,
        final_postclip_grad_norm=final_postclip_grad_norm,
        losses=losses,
        raw_grad_norms=raw_grad_norms,
        grad_norms=grad_norms,
        learning_rates=learning_rates,
        clipped_flags=clipped_flags,
        clip_count=count(identity, clipped_flags),
        clip_fraction=isempty(clipped_flags) ? 0.0 : mean(clipped_flags),
        loss_tail_relspan=tail_relative_span(losses),
        grad_tail_relspan=tail_relative_span(raw_grad_norms),
        terminated_early=terminated_early,
        solve_seconds=solve_seconds,
        iterations=length(losses),
    )
end

