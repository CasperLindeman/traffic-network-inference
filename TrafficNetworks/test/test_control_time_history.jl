#!/usr/bin/env julia

using Test

module LocalTrafficNetworksControlTime
include("../src/fluxes.jl")
include("../src/roads.jl")
include("../src/junctions.jl")
include("../src/boundaries.jl")
include("../src/road_network.jl")
include("../src/solver.jl")
end

const Roads = LocalTrafficNetworksControlTime.Roads
const Junctions = LocalTrafficNetworksControlTime.Junctions
const Boundaries = LocalTrafficNetworksControlTime.Boundaries
const RoadNetworks = LocalTrafficNetworksControlTime.RoadNetworks
const Solvers = LocalTrafficNetworksControlTime.Solvers

@testset "Exact control-time history" begin
    road = Roads.make_road(1, 1, 1.0, 4, x -> 0.25, 1, 1)
    net = RoadNetworks.RoadNetwork(
        Roads.Road[road],
        Junctions.Junction[],
        Boundaries.Boundary[Boundaries.Boundary(1, t -> 0.10)],
        0.20,
        0.50,
    )

    requested_times = [0.03, 0.17, 0.20]
    hist = Solvers.simulate!(net; times=requested_times)

    @test hist isa Solvers.SimulationHistory
    @test hist.times == requested_times
    @test size(hist.road_histories[1]) == (length(road.rho), length(requested_times))
end
