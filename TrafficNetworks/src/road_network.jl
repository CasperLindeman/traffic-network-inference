module RoadNetworks

export RoadNetwork

using ..Roads: Road
using ..Junctions: Junction
using ..Boundaries: Boundary

struct RoadNetwork
    roads::Vector{Road}
    junctions::Vector{Junction}
    boundaries::Vector{Boundary}
    T::Float64
    CFL::Float64
end

end # module RoadNetworks
