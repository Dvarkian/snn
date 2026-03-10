#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using DrWatson
DrWatson.@quickactivate
using WRCircuit
WRCircuit.@preamble
set_theme!(foresight(:physics))

model = models.balanced.Balanced
modelname = "Balanced"

begin # * Simulate
    N = 500
    T = 1500.0

    m = model(N; g = 4.5, nu_hat = 0.9, epsilon = 0.1, D = 1.5, J = 0.1) # Try 4.0, nu_hat = 1.0 for gaussian
    X = bpsolve(m, T; populations = [:E, :I], vars = [:spike, :V, :input])
    V = X[Var = At(:V)]
    input = X[Var = At(:input)]
    spikes = X[Var = At(:spike)]
    pop = :E # Excitatory
end
begin # Spike raster
    f = Figure(size = (1920, 480))
    gs = subdivide(f, 1, 4)

    T = 500u"ms" .. 1000u"ms"

    # First do spike raster with firing rate
    ax = Axis(gs[1][1, 1])
    x = spiketimes(spikes[Population = At(pop)][𝑡 = T][:, 1:1000])
    spikeraster!(ax, 1:length(x), x; markersize = 5) # Switch to clustering
    f
end
begin # * Voltage traces
    ax = Axis(gs[1][2, 1])
    x = V[Population = At(pop)][𝑡 = T][:, [1]]
    map(eachslice(x; dims = 2)) do _x
        lines!(ax, ustripall(_x))
    end
    f
end
begin # * Synaptic input distribution
    τ = 1 # ms
    syn = coarsegrain(input[Population = At(pop)], τ) .|> sum
    syn = Iterators.flatten(syn) |> collect
    ax = Axis(gs[2]; yscale = log10, limits = (nothing, (1e-7, 1)))
    ziggurat!(syn; bins = 50, normalization = :pdf, color = Cycled(1))

    D = fit(Normal, syn)
    xs = range(extrema(syn)...; length = 500)
    lines!(ax, xs, pdf.(D, xs); color = Cycled(2), linestyle = :dash)

    D = fit(Stable, syn)
    D = Stable(D.α, 0, D.σ, D.μ)
    lines!(ax, xs, pdf.(D, xs); color = Cycled(4), linestyle = :dash)
    f
end
begin # Some dynamical statistics (firing rate, fano factor, etc...)
    stats = [firingrate, cv] #, fanofactor]
    for (i, s) in enumerate(stats)
        ax = Axis(gs[3][i, 1]; title = string(s))
        y = s(spikes[Population = At(pop)]) |> ustripall |> collect
        ziggurat!(ax, y, label = string(s); color = Cycled(i))
    end
    f
end
begin # * Power spectrum of voltage fluctuations
    x = V[Population = At(pop)][:, 1:100:end]
    x = set(x, 𝑡 => ustripall(times(x)) ./ 1000) # To unitless seconds
    x = set(x, Neuron => 1:size(x, 2))
    x = x .- mean(x, dims = 1)
    y = progressmap(x -> spectrum(x, 3), eachslice(x; dims = 2)) |> stack
    # y = mean(y, dims = 2)
    # y = dropdims(y, dims = 2)
    ax = Axis(gs[4])
    spectrumplot!(ax, y[2:end, :])
    f |> display
end

# begin
#     y = map(spikes) do x
#         x = set(x, SparseMatrixCSC(parent(x)))
#         x = set(x, Neuron => Dim{:neuron})
#         x = x[1:5000, 1:10:end]
#     end
#     y = set(y, Population => Dim{:population})
#     save("spikes.jld2", Dict("spikes" => y))
# end
# begin # * Raster plot
#     f = Figure()
#     ax = Axis(f[1, 1])

#     x = spiketimes(spikes[Population = At(:E)][1:5000, 1:10:end])

#     for (i, s) in enumerate(x)
#         scatter!(ax, ustripall(s), i * ones(length(s)), color = :black, markersize = 3)
#     end

#     # spikeraster!(ax, spikes) # ? Add to TimeseriesTools, along with a sorting function.
#     f
# end

# begin # * Mean-squared displacement of voltage
#     x = V[Population = At(:E)]
#     msd = progressmap(msdist, eachslice(x, dims = Neuron)) |> stack # * Takes about 5 minutes?
# end
# begin
#     μ = dropdims(mean(msd; dims = 2); dims = Neuron)
#     σ = dropdims(std(msd; dims = 2); dims = Neuron)
# end
# begin
#     f = Figure()
#     ax = Axis(f[1, 1])#; xscale = log10, yscale = log10)
#     lines!(ax, μ)
#     f
# end

if false
    x = X[Population = At(:E), Var = At(:V)]
    d = map(diff, eachslice(x, dims = Neuron))
    d = stack(d)
    d = d[abs.(d) .< 7]
end
if false
    f = Figure()
    ax = Axis(f[1, 1]; yscale = log10)
    hist!(d[1:100:end]; bins = 100, normalization = :pdf)
    # p = fit(Normal, d[1:100:end])
    # xs = -5:0.01:5
    # ys = pdf.([p], xs)
    # lines!(xs, ys)
    f
end

if false
    begin
        x = V[Population = At(:E)]
        τ = 100 # lag 0 difference
        Δ = structurefunction(x, τ)
        # Δ = Δ[abs.(Δ) .< 5]
        hist(Δ[:]; bins = 100)
    end

    begin
        x = V[Population = At(:E)]
        τs = 1:1:200
        Δ = structurefunction(x, τs)
        edges = -3:0.1:3
        H = progressmap(Δ) do x
            histcounts(x, edges)
        end
        H = ToolsArray(H, 𝑡(τs)) |> stack |> transpose
        H = H ./ maximum(H, dims = 2)
        heatmap(H)
    end
    begin
        x = V[Population = At(:E)]
        τs = 1:50
        Δ = structurefunction(x, τs)
        edges = -2.5:0.1:3
        H = progressmap(Δ) do x
            histcounts(x, edges)
        end
        H = ToolsArray(H, 𝑡(τs)) |> stack |> transpose
        # H = H ./ maximum(H, dims = 2)
        heatmap(log10.(H))
        lines(H[50, :] .+ eps(); axis = (; yscale = log10))
    end
end
