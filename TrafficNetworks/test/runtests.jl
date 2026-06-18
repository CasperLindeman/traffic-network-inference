#!/usr/bin/env julia

using Test

@testset "TrafficNetworks tests" begin
    include("test_control_time_history.jl")
    include("test_reusable_workflow.jl")
    include("test_simulation_sanity.jl")
end
