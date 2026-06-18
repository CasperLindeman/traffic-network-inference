import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

if !isdefined(@__MODULE__, :run_one_to_two_inference)
    previous_skip = get(ENV, "TRAFFICNETWORKS_SKIP_ONE_TO_TWO_DEMO", nothing)
    ENV["TRAFFICNETWORKS_SKIP_ONE_TO_TWO_DEMO"] = "1"
    include(joinpath(@__DIR__, "inference.jl"))
    if previous_skip === nothing
        delete!(ENV, "TRAFFICNETWORKS_SKIP_ONE_TO_TWO_DEMO")
    else
        ENV["TRAFFICNETWORKS_SKIP_ONE_TO_TWO_DEMO"] = previous_skip
    end
end

using Distributions
using LaTeXStrings
using Plots
using TrafficNetworks

function plot_turning_fraction_prior_posterior(results; output_path=nothing)
    grid = collect(range(0.0, 1.0; length=600))
    prior_dist = Beta(results.prior_alpha, results.prior_beta)
    prior_density = pdf.(prior_dist, grid)
    posterior_density = TrafficNetworks.weighted_kde_unit_interval(grid, results.a_samples, results.weights)
    max_density = 1.08 * maximum(vcat(prior_density, posterior_density))

    plt = plot(
        grid,
        prior_density;
        color=:gray35,
        linewidth=2.8,
        linestyle=:dash,
        label="Prior",
        xlabel=L"\alpha",
        ylabel="Density",
        xlims=(0.0, 1.0),
        ylims=(0.0, max_density),
        size=(950, 560),
        dpi=180,
        legend=:topleft,
        left_margin=8Plots.mm,
        bottom_margin=7Plots.mm,
        grid=true,
        framestyle=:box,
        titlefontsize=18,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
    )

    plot!(
        plt,
        grid,
        posterior_density;
        color=:steelblue4,
        linewidth=3.0,
        fillrange=0.0,
        fillalpha=0.18,
        label="EnIS posterior approximation",
    )
    vline!(
        plt,
        [results.a_true];
        color=:firebrick3,
        linewidth=2.4,
        linestyle=:dash,
        label="Reference value",
    )
    vline!(
        plt,
        [results.a_post_mean];
        color=:black,
        linewidth=2.0,
        linestyle=:dot,
        label="Posterior mean",
    )

    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(plt, output_path)
    end

    return plt
end

function make_one_to_two_posterior_figure(results; output_dir=ONE_TO_TWO_OUTPUT_DIR)
    output_path = joinpath(output_dir, "one_to_two_turning_fraction_prior_posterior.png")
    fig = plot_turning_fraction_prior_posterior(results; output_path=output_path)

    return (
        results=results,
        figure=fig,
        output_path=output_path,
    )
end
