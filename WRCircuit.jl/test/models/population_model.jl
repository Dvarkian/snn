#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using PythonCall
using WRCircuit
using TimeseriesTools

FNSPopulations = models.balanced.FNSPopulations

begin # * Model
    N = 12500
    FNSnet = FNSPopulations(N)

    ne = FNSnet.E.num |> convert2(Int)
    ni = FNSnet.I.num |> convert2(Int)
end

using CairoMakie
using GraphMakie
using GraphMakie.Graphs

begin # * Retrieve connection matrices
    E2E = FNSnet.E2E.conn.require("conn_mat") |> convert2(Matrix{Bool})  # Python has columns first
    I2E = FNSnet.I2E.conn.require("conn_mat") |> convert2(Matrix{Bool})
    E2I = FNSnet.E2I.conn.require("conn_mat") |> convert2(Matrix{Bool})
    I2I = FNSnet.I2I.conn.require("conn_mat") |> convert2(Matrix{Bool})
end

function visualize_populations(FNSnet; maxn = 100)
    E2E = FNSnet.E2E.conn.require("conn_mat") |> convert2(Matrix{Bool})  # Python has columns first
    I2E = FNSnet.I2E.conn.require("conn_mat") |> convert2(Matrix{Bool})
    E2I = FNSnet.E2I.conn.require("conn_mat") |> convert2(Matrix{Bool})
    I2I = FNSnet.I2I.conn.require("conn_mat") |> convert2(Matrix{Bool})

    positions_E = FNSnet.E.positions |> convert2(Vector)  # num_E x 2
    positions_I = FNSnet.I.positions |> convert2(Vector) # num_I x 2

    ratio = length(positions_E) / length(positions_I)
    _ne = min(length(positions_E), maxn)
    _ni = min(length(positions_I), Int(maxn ÷ ratio))
    eidxs = rand(1:length(positions_E), _ne)
    iidxs = rand(1:length(positions_I), _ni)

    E2E = E2E[eidxs, eidxs]
    I2E = I2E[iidxs, eidxs]
    E2I = E2I[eidxs, iidxs]
    I2I = I2I[iidxs, iidxs]
    positions_E = positions_E[eidxs]
    positions_I = positions_I[iidxs]
    fullpositions = Point2f.(collect.(vcat(positions_E, positions_I)))

    _ne = length(positions_E)
    _ni = length(positions_I)
    node_colors = vcat(fill(("green", 0.6), _ne), fill(("red", 0.6), ni))

    intraconn = [E2E fill(false, size(E2I)); fill(false, size(I2E)) I2I]
    Gintra = SimpleDiGraph(intraconn)
    edge_colors = fill((colorant"green", 0.05), size(intraconn))
    edge_colors[(_ne + 1):end, (_ne + 1):end] .= [(colorant"crimson", 0.05)]

    interconn = [fill(false, size(E2E)) E2I; I2E fill(false, size(I2I))]
    Ginter = SimpleDiGraph(interconn)

    f = Figure()
    ax = Axis(f[1, 1]; aspect = DataAspect())
    pinter = graphplot!(ax, Ginter; layout = fullpositions, node_size = 0,
                        edge_color = (colorant"cornflowerblue", 0.05),
                        edge_plottype = :beziersegments, curve_distance = 1,
                        curve_distance_usage = true)
    pintra = graphplot!(ax, Gintra; layout = fullpositions, node_color = node_colors,
                        edge_color = edge_colors[intraconn])
    hidespines!(ax)
    hidedecorations!(ax)
    f
end

visualize_populations(FNSnet)

begin # * Check Ce, Ci and Cext
    using Statistics
    Ce = sum([E2E E2I]; dims = 1) # Num. of connections FROM excitatory neurons
    Ci = sum([I2E I2I]; dims = 1) # Num. of connections FROM inhibitory neurons
    @assert ≈(mean(Ce) ./ mean(Ci), 4.0; atol = 0.1) # Mean number of from-exc connections is 4 times that of from-inh connections

    epsilon = FNSnet.epsilon |> convert2(Float64)
    @assert ≈(mean([E2E E2I; I2E I2I]), epsilon, rtol = 0.01)

    [FNSnet.ext2E.conn.require("conn_mat") |> convert2(Matrix{Bool}) |> transpose;
     FNSnet.ext2I.conn.require("conn_mat") |> convert2(Matrix{Bool}) |> transpose]
end

begin # * Simulate
    T = 1000.0
    X = bpsolve(FNSnet, T; populations = [:E, :I], vars = [:V, :spike])

    Xe = X[Population = At(:E), Var = At(:spike)]
    Xi = X[Population = At(:I), Var = At(:spike)]
    Ve = X[Population = At(:E), Var = At(:V)]
    Vi = X[Population = At(:I), Var = At(:V)]
    X = [Xe Xi]
end

begin
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time (ms)", ylabel = "Neuron index")
    spikes = findall(X[𝑡(500 .. 600)])
    spikes = collect(Iterators.product(t, 1:N))[spikes]
    scatter!(ax, spikes, markersize = 1)
    hlines!(ax, [ne + 0.5], linestyle = :dash, color = :black)
    f |> display
end

begin # * Voltage traces
    lines(Ve[5000:5500, 1])
    display(current_figure())
end

begin # * Distribution of differences
    dV = Ve[2:end, :] - parent(Ve[1:(end - 1), :])
    hist(dV[dV .> -5], bins = 50; axis = (; yscale = log10))
end

PythonCall.GC.gc()
