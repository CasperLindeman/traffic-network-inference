#!/usr/bin/env julia

using Random
using Test
using TrafficNetworks

const SPEC_DIR = joinpath(@__DIR__, "..", "thesis_experiments", "network_specs")
const SPEC_PATHS = sort(filter(endswith(".toml"), readdir(SPEC_DIR; join=true)))

function row_softmax_rules(spec::ExperimentNetworkSpec)
    n_rows = length(first(spec.junctions).incoming)
    n_cols = length(first(spec.junctions).outgoing)
    @test all(j -> length(j.incoming) == n_rows && length(j.outgoing) == n_cols, spec.junctions)

    parameterization = RowSoftmaxTurningParameterization(length(spec.junctions), n_rows, n_cols)
    z = zeros(parameter_count(parameterization))
    matrices = turning_matrices_from_logits(z, parameterization)

    for P in matrices
        @test size(P) == (n_rows, n_cols)
        @test all(P .>= 0.0)
        @test maximum(abs.(sum(P; dims=2) .- 1.0)) <= 1e-12
    end

    return [TurningFractionRule(P) for P in matrices]
end

function workflow_observations(hist::SimulationHistory, spec::ExperimentNetworkSpec)
    roads = observed_road_ids(spec)
    @test !isempty(roads)

    if haskey(spec.observation, "paired_cell_fractions")
        cells = observed_cell_ids(spec; mode=:paired)
        @test length(cells) == length(roads)
        return flatten_paired_cell_observations(hist, roads, cells, hist.times)
    end

    cells = observed_cell_ids(spec)
    @test !isempty(cells)
    return flatten_cell_observations(hist, roads, cells, hist.times)
end

@testset "Reusable experiment workflow" begin
    @test !isempty(SPEC_PATHS)

    for (spec_idx, spec_path) in enumerate(SPEC_PATHS)
        @testset "$(basename(spec_path))" begin
            spec = load_experiment_network_spec(spec_path)
            @test spec isa ExperimentNetworkSpec
            @test !isempty(road_ids(spec))
            @test !isempty(spec.junctions)
            @test !isempty(spec.control_times)

            rules = row_softmax_rules(spec)
            smoke_time = first(spec.control_times)
            smoke_times = [smoke_time]

            count_net = build_experiment_network(spec, rules; T=smoke_time)
            @test simulation_step_count(count_net; times=smoke_times) > 0

            net = build_experiment_network(spec, rules; T=smoke_time)
            hist = simulate!(net; times=smoke_times)
            @test hist isa SimulationHistory
            @test hist.times == smoke_times
            @test length(hist.road_histories) == length(road_ids(spec))
            @test all(H -> size(H, 2) == length(smoke_times), hist.road_histories)
            @test all(H -> all(isfinite, H), hist.road_histories)

            y_true = workflow_observations(hist, spec)
            @test !isempty(y_true)
            @test all(isfinite, y_true)
            @test all(x -> -1e-12 <= x <= 1.0 + 1e-12, y_true)

            noisy = generate_physical_observations(y_true, 0.02, MersenneTwister(10_000 + spec_idx))
            @test length(noisy.y_true) == length(y_true)
            @test length(noisy.y_obs) == length(y_true)
            @test length(noisy.sigma_true) == length(y_true)
            @test length(noisy.sigma_model) == length(y_true)
            @test all(x -> 0.0 <= x <= 1.0, noisy.y_obs)
            @test all(>=(0.0), noisy.sigma_true)
            @test all(>(0.0), noisy.sigma_model)
            @test 0.0 <= noisy.clip_fraction <= 1.0

            summary = recovery_summary(
                noisy.y_obs,
                noisy.y_true;
                lower=max.(noisy.y_obs .- noisy.sigma_model, 0.0),
                upper=min.(noisy.y_obs .+ noisy.sigma_model, 1.0),
            )
            @test summary.rmse >= 0.0
            @test summary.mean_interval_width >= 0.0
            @test 0.0 <= summary.interval_coverage_rate <= 1.0
        end
    end
end
