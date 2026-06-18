import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "plots.jl"))

make_square_density_noise_rmse_diagnostic_figure()
