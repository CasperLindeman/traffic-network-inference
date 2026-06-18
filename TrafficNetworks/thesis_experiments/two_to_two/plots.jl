import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

if !isdefined(@__MODULE__, :run_two_to_two_inference)
    previous_skip = get(ENV, "TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO", nothing)
    ENV["TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO"] = "1"
    include(joinpath(@__DIR__, "inference.jl"))
    if previous_skip === nothing
        delete!(ENV, "TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO")
    else
        ENV["TRAFFICNETWORKS_SKIP_TWO_TO_TWO_DEMO"] = previous_skip
    end
end

using Distributions
using LaTeXStrings
using Plots
using TrafficNetworks

const TWO_TO_TWO_POSTERIOR_IMAGE_DIR = joinpath(@__DIR__, "figures")

function enis_particle_matrix(results)
    samples = Matrix{Float64}(results.param_samples)

    if size(samples, 1) == 2
        return Matrix(transpose(samples))
    elseif size(samples, 2) == 2
        return samples
    end

    error("Expected two-to-two particles with either shape (2, N_e) or (N_e, 2), got $(size(samples)).")
end

function plot_two_to_two_enis_posterior(results; output_png=nothing)
    particles = enis_particle_matrix(results)
    weights = TrafficNetworks.normalize_importance_weights(results.weights)
    post_mean = vec(sum(particles .* weights, dims=1))
    truth = results.param_true

    hist_centers, density_2d = TrafficNetworks.weighted_histogram_2d(particles, weights; bins=85, smooth_sigma_bins=1.25)
    p1_centers, p1_density = TrafficNetworks.weighted_histogram_1d(particles[:, 1], weights; bins=100, smooth_sigma_bins=1.5)
    p2_centers, p2_density = TrafficNetworks.weighted_histogram_1d(particles[:, 2], weights; bins=100, smooth_sigma_bins=1.5)

    prior_grid = collect(range(0.0, 1.0; length=500))
    prior_density = pdf.(Beta(results.prior_shape, results.prior_shape), prior_grid)
    top_ymax = 1.12 * maximum(vcat(prior_density, p1_density))
    right_xmax = 1.12 * maximum(vcat(prior_density, p2_density))

    default(tickfontsize=13, guidefontsize=15, legendfontsize=12, titlefontsize=18)

    plt = plot(
        layout=grid(2, 2; heights=[0.24, 0.76], widths=[0.64, 0.36]),
        size=(1150, 980),
        dpi=220,
        left_margin=8Plots.mm,
        bottom_margin=7Plots.mm,
        top_margin=4Plots.mm,
        right_margin=4Plots.mm,
    )

    plot!(
        plt,
        p1_centers,
        p1_density;
        subplot=1,
        color=:steelblue4,
        linewidth=2.5,
        fillrange=0.0,
        fillalpha=0.18,
        label="EnIS posterior approximation",
        xlims=(0.0, 1.0),
        ylims=(0.0, top_ymax),
        ylabel="Density",
        xticks=false,
        legend=:topleft,
        framestyle=:box,
        grid=true,
    )
    plot!(
        plt,
        prior_grid,
        prior_density;
        subplot=1,
        color=:gray35,
        linewidth=2.3,
        linestyle=:dash,
        label="Prior",
    )
    vline!(
        plt,
        [truth[1]];
        subplot=1,
        color=:firebrick3,
        linewidth=1.9,
        linestyle=:dash,
        label="",
    )
    vline!(
        plt,
        [post_mean[1]];
        subplot=1,
        color=:black,
        linewidth=1.7,
        linestyle=:dot,
        label="",
    )
    scatter!(
        plt,
        [NaN],
        [NaN];
        subplot=1,
        color=:firebrick3,
        markerstrokecolor=:white,
        markerstrokewidth=0.8,
        markershape=:star5,
        markersize=7,
        label="Reference value",
    )
    scatter!(
        plt,
        [NaN],
        [NaN];
        subplot=1,
        color=:black,
        markerstrokecolor=:white,
        markerstrokewidth=0.8,
        markershape=:diamond,
        markersize=5,
        label="Posterior mean",
    )

    plot!(
        plt;
        subplot=2,
        framestyle=:none,
        grid=false,
        ticks=false,
        legend=false,
    )

    heatmap!(
        plt,
        hist_centers,
        hist_centers,
        transpose(density_2d);
        subplot=3,
        color=cgrad([:white, :aliceblue, :lightskyblue, :steelblue, :navy]),
        colorbar=false,
        label="",
        xlims=(0.0, 1.0),
        ylims=(0.0, 1.0),
        xlabel=L"\alpha_{11}",
        ylabel=L"\alpha_{21}",
        aspect_ratio=:equal,
        framestyle=:box,
        grid=true,
    )
    contour!(
        plt,
        hist_centers,
        hist_centers,
        transpose(density_2d);
        subplot=3,
        levels=7,
        color=:steelblue4,
        linewidth=0.8,
        label="",
    )
    vline!(
        plt,
        [truth[1]];
        subplot=3,
        color=:firebrick3,
        linewidth=1.6,
        linestyle=:dash,
        label="",
    )
    hline!(
        plt,
        [truth[2]];
        subplot=3,
        color=:firebrick3,
        linewidth=1.6,
        linestyle=:dash,
        label="",
    )
    vline!(
        plt,
        [post_mean[1]];
        subplot=3,
        color=:black,
        linewidth=1.4,
        linestyle=:dot,
        label="",
    )
    hline!(
        plt,
        [post_mean[2]];
        subplot=3,
        color=:black,
        linewidth=1.4,
        linestyle=:dot,
        label="",
    )
    scatter!(
        plt,
        [truth[1]],
        [truth[2]];
        subplot=3,
        color=:firebrick3,
        markerstrokecolor=:white,
        markerstrokewidth=1.1,
        markershape=:star5,
        markersize=9,
        label="",
    )
    scatter!(
        plt,
        [post_mean[1]],
        [post_mean[2]];
        subplot=3,
        color=:black,
        markerstrokecolor=:white,
        markerstrokewidth=0.9,
        markershape=:diamond,
        markersize=6,
        label="",
    )

    plot!(
        plt,
        p2_density,
        p2_centers;
        subplot=4,
        color=:steelblue4,
        linewidth=2.5,
        fillrange=0.0,
        fillalpha=0.18,
        label="",
        xlims=(0.0, right_xmax),
        ylims=(0.0, 1.0),
        xlabel="Density",
        yticks=false,
        legend=false,
        framestyle=:box,
        grid=true,
    )
    plot!(
        plt,
        prior_density,
        prior_grid;
        subplot=4,
        color=:gray35,
        linewidth=2.3,
        linestyle=:dash,
        label="",
    )
    hline!(
        plt,
        [truth[2]];
        subplot=4,
        color=:firebrick3,
        linewidth=1.9,
        linestyle=:dash,
        label="",
    )
    hline!(
        plt,
        [post_mean[2]];
        subplot=4,
        color=:black,
        linewidth=1.7,
        linestyle=:dot,
        label="",
    )

    if output_png !== nothing
        mkpath(dirname(output_png))
        savefig(plt, output_png)
        println(output_png)
    end

    return plt
end

function make_two_to_two_posterior_figure(results; output_dir=TWO_TO_TWO_POSTERIOR_IMAGE_DIR)
    output_png = joinpath(output_dir, "two_to_two_turning_fractions_posterior.png")
    fig = plot_two_to_two_enis_posterior(results; output_png=output_png)

    return (
        results=results,
        figure=fig,
        output_png=output_png,
    )
end
