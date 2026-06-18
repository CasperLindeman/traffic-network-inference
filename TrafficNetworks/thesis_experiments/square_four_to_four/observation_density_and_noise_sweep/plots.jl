import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "sources", "density_noise_grid.jl"))

const DENSITY_NOISE_OUTPUT_DIR = joinpath(@__DIR__, "outputs")
const DENSITY_NOISE_FIGURE_DIR = joinpath(@__DIR__, "figures")

function make_square_density_noise_rmse_diagnostic_figure()
    figure_path = joinpath(DENSITY_NOISE_FIGURE_DIR, "square_density_noise_rmse_diagnostics.png")
    output_paths = run_density_noise_full_lbfgs_plots(;
        output_dir=DENSITY_NOISE_OUTPUT_DIR,
        figure_dir=DENSITY_NOISE_FIGURE_DIR,
    )

    println("Replotted density/noise diagnostic figure")
    println("-----------------------------------------")
    println(figure_path)

    return (
        output_paths=output_paths,
        figure_path=figure_path,
    )
end
