import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "plots.jl"))

make_square_likelihood_weighting_diagnostic_figures()
