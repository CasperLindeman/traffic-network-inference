import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "plots.jl"))

make_square_data_amount_and_esmda_budget_figures()
