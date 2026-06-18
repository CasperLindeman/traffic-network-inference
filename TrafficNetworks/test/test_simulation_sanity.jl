#!/usr/bin/env julia

using Test

module LocalTrafficNetworksSimulationSanity
include("../src/fluxes.jl")
include("../src/roads.jl")
include("../src/junctions.jl")
include("../src/boundaries.jl")
include("../src/road_network.jl")
include("../src/solver.jl")
end

const Fluxes = LocalTrafficNetworksSimulationSanity.Fluxes
const Roads = LocalTrafficNetworksSimulationSanity.Roads
const Junctions = LocalTrafficNetworksSimulationSanity.Junctions
const Boundaries = LocalTrafficNetworksSimulationSanity.Boundaries
const RoadNetworks = LocalTrafficNetworksSimulationSanity.RoadNetworks
const Solvers = LocalTrafficNetworksSimulationSanity.Solvers

clone_road(road::Roads.Road) = Roads.Road(
    road.id,
    road.dx,
    copy(road.rho),
    copy(road.F),
    road.speed_limit,
    road.rho_max,
    road.gamma,
)

function reference_update_road!(road::Roads.Road, dt, left_flux, right_flux)
    rho_old = copy(road.rho)
    n = length(rho_old)
    fluxes = zeros(eltype(rho_old), n + 1)
    fluxes[1] = left_flux
    for i in 2:n
        fluxes[i] = Fluxes.godunov_flux(rho_old[i - 1], rho_old[i])
    end
    fluxes[end] = right_flux

    a = road.gamma * dt / road.dx
    for i in 1:n
        road.rho[i] = rho_old[i] - a * (fluxes[i + 1] - fluxes[i])
    end

    return road
end

all_finite_and_bounded(A; lo = -1e-12, hi = 1 + 1e-12) =
    all(x -> isfinite(x) && lo <= x <= hi, A)

println("Running traffic simulation sanity checks")
println("=" ^ 60)

@testset "Simulation sanity checks" begin
    @testset "Single-road update matches conservative reference" begin
        road_under_test = Roads.make_road(1, 1, 1.0, 4, x -> x < 0.5 ? 0.8 : 0.2, 1, 1)
        road_reference = clone_road(road_under_test)
        dt = 0.9 * Roads.cfl_dt(road_under_test, 0.5)

        Roads.update_road!(road_under_test, dt, 0.0, 0.0)
        reference_update_road!(road_reference, dt, 0.0, 0.0)

        max_diff = maximum(abs.(road_under_test.rho .- road_reference.rho))
        println("single-road step max difference vs reference: ", max_diff)
        @test max_diff <= 1e-12
    end

    @testset "Endpoint CFL helpers are conservative for queued inflow" begin
        @test isapprox(Fluxes.endpoint_wavespeed(0.0), 1.0; atol=1e-12)
        @test isapprox(Fluxes.endpoint_wavespeed(0.25), 0.0; atol=1e-12)
        @test Fluxes.endpoint_wavespeed(0.2500000001) == 0.0

        road = Roads.make_road(1, 1, 1.0, 4, x -> 0.5, 1, 1)
        interior_dt = Roads.cfl_dt(road, 0.5)
        endpoint_dt = Roads.cfl_dt(road, 0.5, 0.0, 0.25)

        @test endpoint_dt < interior_dt
        @test isapprox(endpoint_dt, 0.125; atol=1e-12)

        free_road = Roads.make_road(2, 1, 1.0, 4, x -> 0.1, 1, 1)
        queued_boundary = Boundaries.Boundary(2, t -> 0.0; queue=0.1)
        cfl_flux = Boundaries.boundary_cfl_flux(queued_boundary, free_road, 0.0)
        queued_flux = Boundaries.boundary_flux!(queued_boundary, free_road, 0.0, 0.1)

        @test cfl_flux <= queued_flux
        @test Fluxes.endpoint_wavespeed(cfl_flux) >= Fluxes.endpoint_wavespeed(queued_flux)
    end

    @testset "Single-road simulate! history is sane" begin
        road = Roads.make_road(1, 1, 1.0, 6, x -> 0.4, 1, 1)
        net = RoadNetworks.RoadNetwork(
            Roads.Road[road],
            Junctions.Junction[],
            Boundaries.Boundary[Boundaries.Boundary(1, t -> 0.10)],
            0.30,
            0.50,
        )

        hist = Solvers.simulate!(net, save_every = 1)

        @test hist isa Solvers.SimulationHistory
        @test !isempty(hist.times)
        @test size(hist.road_histories[1], 2) == length(hist.times)
        @test hist.times[end] ≈ net.T atol = 1e-12
        @test all_finite_and_bounded(hist.road_histories[1])
    end

    @testset "Inflow boundary respects cap, supply, and queue" begin
        road = Roads.make_road(1, 1, 1.0, 4, x -> 0.9, 1, 1)
        boundary = Boundaries.Boundary(1, t -> t < 0.1 ? 1.0 : 0.0)
        dt = 0.1

        first_flux = Boundaries.boundary_flux!(boundary, road, 0.0, dt)
        @test isapprox(first_flux, Fluxes.flux(0.9); atol=1e-12)
        @test first_flux <= 0.25
        @test isapprox(boundary.queue, dt * (1.0 - first_flux); atol=1e-12)

        road.rho .= 0.1
        previous_queue = boundary.queue
        second_flux = Boundaries.boundary_flux!(boundary, road, 0.1, dt)
        @test isapprox(second_flux, 0.25; atol=1e-12)
        @test isapprox(boundary.queue, previous_queue - dt * second_flux; atol=1e-12)
        @test boundary.queue >= 0.0

        free_road = Roads.make_road(2, 1, 1.0, 4, x -> 0.1, 1, 1)
        capped_boundary = Boundaries.Boundary(2, t -> 0.28)
        capped_flux = Boundaries.boundary_flux!(capped_boundary, free_road, 0.0, dt)
        @test isapprox(capped_flux, 0.25; atol=1e-12)
        @test isapprox(capped_boundary.queue, dt * (0.28 - 0.25); atol=1e-12)
    end

    @testset "Single-road terminal exit drains mass" begin
        road = Roads.make_road(1, 1, 1.0, 6, x -> 0.4, 1, 1)
        initial_mass = sum(road.rho)
        net = RoadNetworks.RoadNetwork(
            Roads.Road[road],
            Junctions.Junction[],
            Boundaries.Boundary[],
            0.20,
            0.50,
        )

        hist = Solvers.simulate!(net, save_every = 1)
        final_mass = sum(hist.road_histories[1][:, end])

        println("initial single-road mass: ", initial_mass)
        println("final single-road mass:   ", final_mass)
        @test final_mass < initial_mass - 1e-6
    end

    @testset "Requested control times are respected" begin
        road = Roads.make_road(1, 1, 1.0, 4, x -> 0.25, 1, 1)
        net = RoadNetworks.RoadNetwork(
            Roads.Road[road],
            Junctions.Junction[],
            Boundaries.Boundary[Boundaries.Boundary(1, t -> 0.10)],
            0.20,
            0.50,
        )

        requested_times = collect(0.05:0.05:0.20)
        hist = Solvers.simulate!(net, times = requested_times, save_every = 1)

        println("requested times: ", requested_times)
        println("actual times:    ", hist.times)
        @test hist.times == requested_times
    end

    @testset "Simple 1-to-2 junction flux split is consistent" begin
        roads = Roads.Road[
            Roads.make_road(1, 1, 1.0, 6, x -> 0.20, 1, 1),
            Roads.make_road(2, 1, 1.0, 6, x -> 0.10, 1, 1),
            Roads.make_road(3, 1, 1.0, 6, x -> 0.10, 1, 1),
        ]

        junction = Junctions.Junction(
            [1],
            [2, 3],
            Junctions.TurningFractionRule(reshape([0.70, 0.30], 1, 2)),
        )

        fin, fout = Junctions.compute_junction_fluxes(junction, roads)

        @test length(fin) == 1
        @test length(fout) == 2
        @test fin[1] ≈ sum(fout) atol = 1e-12
        @test fout[1] / sum(fout) ≈ 0.70 atol = 1e-12
        @test fout[2] / sum(fout) ≈ 0.30 atol = 1e-12
    end

    @testset "Simple 1-to-2 junction simulate! stays finite" begin
        roads = Roads.Road[
            Roads.make_road(1, 1, 1.0, 8, x -> 0.25, 1, 1),
            Roads.make_road(2, 1, 1.0, 8, x -> 0.10, 1, 1),
            Roads.make_road(3, 1, 1.0, 8, x -> 0.10, 1, 1),
        ]

        junction = Junctions.Junction(
            [1],
            [2, 3],
            Junctions.TurningFractionRule(reshape([0.70, 0.30], 1, 2)),
        )

        net = RoadNetworks.RoadNetwork(
            roads,
            Junctions.Junction[junction],
            Boundaries.Boundary[Boundaries.Boundary(1, t -> 0.12)],
            0.25,
            0.50,
        )

        hist = Solvers.simulate!(net, save_every = 1)

        @test length(hist.road_histories) == 3
        @test all(size(H, 2) == length(hist.times) for H in hist.road_histories)
        @test all(all_finite_and_bounded(H) for H in hist.road_histories)
    end

    @testset "Module hygiene" begin
        leaked_symbol = isdefined(LocalTrafficNetworksSimulationSanity, :compute_junction_fluxes)
        println("leaked top-level compute_junction_fluxes: ", leaked_symbol)
        @test !leaked_symbol
    end
end
