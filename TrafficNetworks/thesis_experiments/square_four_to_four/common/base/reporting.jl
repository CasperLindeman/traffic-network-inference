function print_matrix_side_by_side(label_left, left, label_right, right)
    println(label_left, "            ", label_right)
    for row in 1:size(left, 1)
        @printf(
            "  [%.3f  %.3f  %.3f  %.3f]    [%.3f  %.3f  %.3f  %.3f]\n",
            left[row, 1], left[row, 2], left[row, 3], left[row, 4],
            right[row, 1], right[row, 2], right[row, 3], right[row, 4],
        )
    end
end

function print_setup_summary(setup::SquareFourToFourSetup)
    println("Square 4x4-junction stress test")
    println("--------------------------------")
    println("Junctions: ", join(JUNCTION_LABELS, ", "))
    println("Observed roads: ", setup.observed_road_ids)
    println("  incoming observed roads: ", intersect(setup.observed_road_ids, EXTERNAL_INCOMING_ROADS))
    println("  outgoing observed roads: ", intersect(setup.observed_road_ids, EXTERNAL_OUTGOING_ROADS))
    println("  internal observed roads: ", intersect(setup.observed_road_ids, CONNECTOR_ROADS))
    println("Observed cells: ", setup.observed_cell_ids)
    println("Observation times (min): ", round.(60.0 .* setup.control_times; digits=2))
    println("Observation vector length: ", observation_length(setup))
    println("Generated noise sigma: ", setup.generated_noise_sigma)
    println("Likelihood sigma: ", setup.likelihood_sigma)
    println()
end

function print_variation_summary(setup::SquareFourToFourSetup, variation)
    println("Variation summary")
    println("-----------------")
    for (k, road_id) in enumerate(EXTERNAL_INCOMING_ROADS)
        @printf(
            "%s IC span: %.4f, inflow span: %.4f\n",
            road_label(road_id),
            variation.incoming_ic_spans[k],
            variation.inflow_spans[k],
        )
    end
    println()
    for (k, road_id) in enumerate(setup.observed_road_ids)
        @printf(
            "%s (%s) observed-data span: %.4f, IC span: %.4f\n",
            road_label(road_id),
            road_role_label(road_id),
            variation.observed_data_spans[k],
            variation.observed_ic_spans[k],
        )
    end
    println()
end

function print_sensitivity_summary(sensitivity)
    println("Sensitivity diagnostics")
    println("-----------------------")
    @printf("All-junction uniform replacement RMSE: %.4f\n", sensitivity.all_uniform_rmse)
    @printf("All-junction uniform replacement max abs error: %.4f\n", sensitivity.all_uniform_maxabs)
    for junction in 1:N_JUNCTIONS
        @printf("%s replaced by uniform split -> observation RMSE %.4f\n", JUNCTION_LABELS[junction], sensitivity.per_junction_uniform_rmse[junction])
    end
    println()
end

function print_method_metrics(label, P_est, y_est, P_true, y_true, y_obs, true_state, est_state, solve_seconds, observed_road_ids)
    println(label)
    println(repeat("-", length(label)))
    @printf("solve time: %.2f s\n", solve_seconds)
    @printf("overall turning RMSE: %.4f\n", overall_turning_rmse(P_est, P_true))
    rmse_per_junction = junction_turning_rmses(P_est, P_true)
    for junction in 1:N_JUNCTIONS
        @printf("  %s turning RMSE: %.4f\n", JUNCTION_LABELS[junction], rmse_per_junction[junction])
    end
    @printf("predictive RMSE: %.4f\n", predictive_rmse(y_est, y_true))
    @printf("fit RMSE: %.4f\n", predictive_rmse(y_est, y_obs))
    @printf("final-state RMSE (all roads): %.4f\n", final_state_rmse(est_state, true_state))
    @printf("final-state RMSE (observed roads): %.4f\n", observed_road_state_rmse(est_state, true_state, observed_road_ids))
    println()
end

function print_adam_restart_summary(adam_restarts, P_true, y_true)
    println("ADAM restart diagnostic")
    println("-----------------------")
    for restart in adam_restarts
        @printf(
            "%s: z0 norm = %.3f, turning RMSE = %.4f, predictive RMSE = %.4f, best loss = %.2f\n",
            restart.label,
            restart.z0_norm,
            overall_turning_rmse(restart.adam.P_est, P_true),
            predictive_rmse(restart.adam.y_est, y_true),
            restart.adam.best_loss,
        )
    end
    println()
end

function print_turning_matrix_comparison(P_true, P_esmda, P_adam)
    true_mats = turning_matrices(P_true)
    esmda_mats = turning_matrices(P_esmda)
    adam_mats = turning_matrices(P_adam)

    for junction in 1:N_JUNCTIONS
        println(JUNCTION_LABELS[junction])
        print_matrix_side_by_side("  True:", true_mats[junction], "ESMDA:", esmda_mats[junction])
        print_matrix_side_by_side("  True:", true_mats[junction], "ADAM :", adam_mats[junction])
        println()
    end
end
