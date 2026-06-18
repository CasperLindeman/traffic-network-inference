using ForwardDiff
using LinearAlgebra

turning_matrix_rmse(P_est::AbstractMatrix, P_true::AbstractMatrix) =
    TrafficNetworks.rmse(turning_entries(P_est), turning_entries(P_true))

function map_loss_forwarddiff(z, y_obs, setup::FourToFourInferenceSetup; prior_scale=1.0, sigma_obs=0.01)
    y_pred = simulator_forwarddiff(z, setup)
    residual = (y_pred .- y_obs) ./ sigma_obs
    prior_term = z ./ prior_scale
    return 0.5 * sum(abs2, residual) + 0.5 * sum(abs2, prior_term)
end

function run_adam_map_forwarddiff(
    y_obs::AbstractVector,
    setup::FourToFourInferenceSetup;
    prior_scale=1.0,
    sigma_obs=0.01,
    z0=zeros(12),
    learning_rate=0.08,
    maxiters=500,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    grad_clip=250.0,
)
    loss_fn = z -> map_loss_forwarddiff(z, y_obs, setup; prior_scale=prior_scale, sigma_obs=sigma_obs)
    z = Float64.(collect(z0))
    best_z = copy(z)
    best_loss = Inf

    m = zeros(length(z))
    v = zeros(length(z))
    grad = zeros(length(z))
    cfg = ForwardDiff.GradientConfig(loss_fn, z)

    solve_seconds = @elapsed begin
        for iter in 1:maxiters
            loss = loss_fn(z)
            ForwardDiff.gradient!(grad, loss_fn, z, cfg)
            grad_norm = norm(grad)

            if grad_norm > grad_clip
                grad .*= grad_clip / grad_norm
            end

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

    P_est = turning_matrix(best_z)
    y_est = simulator(best_z, setup)

    return (
        z=copy(z),
        z_best=best_z,
        P_est=P_est,
        y_est=y_est,
        best_loss=best_loss,
        solve_seconds=solve_seconds,
        iterations=maxiters,
    )
end
