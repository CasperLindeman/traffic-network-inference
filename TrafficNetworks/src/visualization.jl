module RoadNetworkViz

using Plots
#using Statistics: mean
using ..Solvers: SimulationHistory

export RoadGeom, NetworkGeom, plot_network, plot_history, plot_road_history, animate_network

"Geometry for one road segment."
struct RoadGeom
    road_id::Int
    p0::NTuple{2,Float64}   # start point (x,y)
    p1::NTuple{2,Float64}   # end point (x,y)
end

"Geometry container."
struct NetworkGeom
    roads::Vector{RoadGeom}
end

function plot_network(net, geom::NetworkGeom; lw=3, show_ids=true)
    # Build lookup table road_id -> geometry
    gmap = Dict{Int,RoadGeom}()
    for g in geom.roads
        gmap[g.road_id] = g
    end

    plt = plot(legend=false, aspect_ratio=:equal)

    for r in net.roads
        g = gmap[r.id]

        p0 = collect(g.p0)
        p1 = collect(g.p1)

        d = p1 .- p0
        L = sqrt(d[1]^2+d[2]^2)

        if L > 0

            p0s = p0 .+ d/L .* 0.05
            p1s = p1 .- d/L .* 0.05
        else
            p0s = p0
            p1s = p1
        end

        x = [p0s[1], p1s[1]]
        y = [p0s[2], p1s[2]]

        plot!(plt, x, y; lw=lw,color="black")

        if show_ids
            xm = (x[1] + x[2]) / 2
            ym = (y[1] + y[2]) / 2
            annotate!(plt, xm, ym, text(string(r.id), 10))
        end
    end

    return plt
end

# ---------- history plotting helpers ----------

"""
Plot the density history for a single road as a heatmap.

`net` is the road network (used to locate the road by `road_id`),
`hist` is a `SimulationHistory` returned from `simulate!`.
`times` defaults to the saved times contained in `hist`.

The x-axis is time, the y-axis is the cell index along the road.
"""
function plot_road_history(net, hist::SimulationHistory, road_id::Int; times=hist.times)
    idx = findfirst(r -> r.id == road_id, net.roads)
    if idx === nothing
        error("road_id $road_id not found in network")
    end
    data = hist.road_histories[idx]
    ncell, nt = size(data)
    plt = heatmap(times, 1:ncell, data;
                  xlabel="time", ylabel="cell",
                  title="road $road_id history", colorbar=true)
    return plt
end

"""
Plot the history for either a single road (when `road_id` is supplied) or
all roads in the network arranged in a stacked layout.

`geom` is currently unused but kept for symmetry with `plot_network`.
"""
function plot_history(net, geom::NetworkGeom, hist::SimulationHistory; road_id=nothing)
    if road_id !== nothing
        return plot_road_history(net, hist, road_id; times=hist.times)
    end
    n = length(net.roads)
    plt = plot(layout = (n, 1), legend=false)
    for (i, r) in enumerate(net.roads)
        p = plot_road_history(net, hist, r.id; times=hist.times)
        plot!(plt, p, subplot=i)
    end
    return plt
end

# ---------- animation helpers ----------

"""
    animate_network(net, geom, hist; colormap=:viridis, fps=5)

Create an animation of the road network with roads colored by density over time.

Each frame shows the network at a time step from the simulation history, with
road colors representing the average density on that road (blue=low, yellow=high).

Arguments:
  - `net::RoadNetwork`: the network
  - `geom::NetworkGeom`: the geometry for drawing
  - `hist::SimulationHistory`: saved history from simulate!(net, save_every=n)
  - `colormap::Symbol`: color scheme (:viridis, :plasma, :turbo, :heat, etc.)
  - `fps::Int`: frames per second for animation playback

Returns:
  A playable animation object that can be displayed or saved as .gif.

Usage:
  anim = animate_network(net, geom, hist, fps=5)
  display(anim)
  gif(anim, "traffic.gif")  # save as file
"""
function animate_network(net, geom::NetworkGeom, hist::SimulationHistory;
                         colormap=:viridis, fps=5, cell_marker=:rect, cell_size=6)
    # create animation object; the notebook widget supplies a play/pause control
    anim = @animate for (t_idx, t) in enumerate(hist.times)
        plt = plot(legend=false, aspect_ratio=:equal,
                  title="t = $(round(t,digits=3))", xlabel="x", ylabel="y", grid=false)
        
        # colour each cell along every road
        for (road_idx, road) in enumerate(net.roads)
            geom_entry = findfirst(g -> g.road_id == road.id, geom.roads)
            if geom_entry === nothing
                continue
            end
            g = geom.roads[geom_entry]
            p0 = collect(g.p0)
            p1 = collect(g.p1)
            d = p1 .- p0
            ncell = length(road.rho)
            for i in 1:ncell
                ξ = (i-0.5)/ncell                   # cell centre fraction
                pos = p0 .+ ξ .* d
                ρ = hist.road_histories[road_idx][i,t_idx]
                c = cgrad(colormap)[clamp(ρ,0.0,1.0)]
                scatter!(plt, [pos[1]], [pos[2]];
                         marker=cell_marker, markersize=cell_size, color=c, label="")
            end
        end
        plt
    end
    return anim   # return the Animation instead of immediately saving a gif
end

end # module