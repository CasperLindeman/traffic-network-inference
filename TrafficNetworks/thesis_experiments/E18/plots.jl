#!/usr/bin/env julia

const E18_DIR = @__DIR__
const E18_FIGURES_DIR = joinpath(E18_DIR, "figures")
const E18_OUTPUTS_DIR = joinpath(E18_DIR, "outputs")
const E18_FINAL_OUTPUT_DIR = joinpath(E18_OUTPUTS_DIR, "inference_128x4_180s")
const E18_FINAL_ERROR_DIR = joinpath(E18_FINAL_OUTPUT_DIR, "final_state_error")

const E18_FIGURE_PATHS = [
    joinpath(E18_FIGURES_DIR, "e18_pruned_graph_map_overlay.png"),
    joinpath(E18_FIGURES_DIR, "e18_pruned_inference_overview.png"),
    joinpath(E18_FIGURES_DIR, "e18_target_area_west.png"),
    joinpath(E18_FIGURES_DIR, "e18_target_area_rb18_rb20.png"),
    joinpath(E18_FIGURES_DIR, "e18_target_area_rb23_rb24.png"),
    joinpath(E18_FIGURES_DIR, "e18_target_area_east.png"),
    joinpath(E18_FIGURES_DIR, "e18_turning_fraction_recovery_128x4.png"),
    joinpath(E18_FIGURES_DIR, "e18_final_state_abs_error_west.png"),
    joinpath(E18_FIGURES_DIR, "e18_final_state_abs_error_rb18_rb20.png"),
    joinpath(E18_FIGURES_DIR, "e18_final_state_abs_error_rb23_rb24.png"),
    joinpath(E18_FIGURES_DIR, "e18_final_state_abs_error_east.png"),
]

function copy_if_present(src, dst)
    if !isfile(src)
        @warn "Missing source figure" src
        return false
    end
    mkpath(dirname(dst))
    cp(src, dst; force=true)
    return true
end

function sync_e18_figures()
    copy_if_present(
        joinpath(E18_FINAL_OUTPUT_DIR, "turning_fraction_recovery.png"),
        joinpath(E18_FIGURES_DIR, "e18_turning_fraction_recovery_128x4.png"),
    )

    for name in [
        "e18_final_state_abs_error_west.png",
        "e18_final_state_abs_error_rb18_rb20.png",
        "e18_final_state_abs_error_rb23_rb24.png",
        "e18_final_state_abs_error_east.png",
    ]
        copy_if_present(joinpath(E18_FINAL_ERROR_DIR, name), joinpath(E18_FIGURES_DIR, name))
    end
end

function validate_e18_figures()
    missing = filter(!isfile, E18_FIGURE_PATHS)
    if !isempty(missing)
        error("Missing E18 figure files:\n" * join(missing, "\n"))
    end
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    sync_e18_figures()
    validate_e18_figures()
    println("E18 figures are present in $(E18_FIGURES_DIR).")
end
