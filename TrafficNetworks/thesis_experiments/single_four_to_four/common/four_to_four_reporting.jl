using Printf

final_state_rmse(estimate::AbstractMatrix, truth::AbstractMatrix) = TrafficNetworks.rmse(estimate, truth)

function print_adam_summary(adam_results, esmda_results, true_state, esmda_state, adam_state)
    println("ADAM MAP baseline")
    println("-----------------")
    @printf("Iterations: %d\n", adam_results.iterations)
    @printf("solve time: %.2f s\n", adam_results.solve_seconds)
    @printf("best MAP objective: %.4f\n", adam_results.best_loss)
    @printf(
        "turning RMSE: %.4f\n",
        turning_matrix_rmse(adam_results.P_est, esmda_results.P_true),
    )
    @printf(
        "predictive RMSE: %.4f\n",
        TrafficNetworks.rmse(adam_results.y_est, esmda_results.y_true),
    )
    @printf(
        "fit RMSE to noisy observations: %.4f\n",
        TrafficNetworks.rmse(adam_results.y_est, esmda_results.y_obs),
    )
    @printf("final-state RMSE: %.4f\n", final_state_rmse(adam_state, true_state))
    println()

    println("ESMDA posterior summary")
    println("----------------------")
    @printf("turning RMSE: %.4f\n", turning_rmse(esmda_results))
    @printf("predictive RMSE: %.4f\n", predictive_rmse(esmda_results))
    @printf("fit RMSE to noisy observations: %.4f\n", TrafficNetworks.rmse(esmda_results.y_post_mean, esmda_results.y_obs))
    @printf("final-state RMSE: %.4f\n", final_state_rmse(esmda_state, true_state))
    println()

    print_matrix_side_by_side("True turning matrix:", esmda_results.P_true, "ADAM estimate:", adam_results.P_est)
    println()
    print_matrix_side_by_side("ESMDA posterior mean:", esmda_results.P_post_mean, "ADAM estimate:", adam_results.P_est)
end

function print_variation_summary(setup::FourToFourInferenceSetup, y_true::AbstractVector)
    println("Variation summary")
    println("-----------------")
    for road_id in eachindex(setup.road_lengths)
        rho0 = road_initial_profile(setup, road_id)
        @printf(
            "road %d initial density span: [%.4f, %.4f], span = %.4f\n",
            road_id,
            minimum(rho0),
            maximum(rho0),
            maximum(rho0) - minimum(rho0),
        )
    end
    println()
    for road_id in eachindex(setup.inflows)
        inflow_vals = [setup.inflows[road_id](t) for t in setup.control_times]
        @printf(
            "incoming road %d inflow span: [%.4f, %.4f], span = %.4f\n",
            road_id,
            minimum(inflow_vals),
            maximum(inflow_vals),
            maximum(inflow_vals) - minimum(inflow_vals),
        )
    end
    println()

    Y = reshape_observations(y_true, setup)
    for road_pos in eachindex(setup.observed_road_ids)
        road_id = setup.observed_road_ids[road_pos]
        vals = vec(Y[:, :, road_pos])
        @printf(
            "observed road %d data span: [%.4f, %.4f], span = %.4f\n",
            road_id,
            minimum(vals),
            maximum(vals),
            maximum(vals) - minimum(vals),
        )
    end
end

