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

model = models.balanced.FNSPopulations
modelname = "Balanced"

begin # * Simulate
    N = 100000
    T = 1500.0

    m = model(N; g = 4.0, nu_hat = 1, epsilon = 0.1, D = 1.5, J = 0.1)
    X = bpsolve(m, T; populations = [:E, :I], vars = [:spike, :V, :RI])
    V = X[Var = At(:V)]
    RI = X[Var = At(:RI)]
    spikes = X[Var = At(:spike)]
    lines(V[1][1:1000, 1])
    # lines(RI[1][1:100, 1])
end
begin
    f = TwoPanel()
    τ = 1 # ms
    for i in 1:2
        syn = coarsegrain(RI[i], τ) .|> sum
        syn = Iterators.flatten(syn) |> collect
        ax = Axis(f[1, i]; yscale = log10, limits = (nothing, (1e-7, nothing)))
        ziggurat!(syn; bins = 50, normalization = :pdf)
        D = fit(Normal, syn)
        xs = range(extrema(syn)...; length = 500)
        lines!(ax, xs, pdf.(D, xs); color = crimson, linestyle = :dash)
        D = fit(Stable, syn)
        D = Stable(D.α, 0, D.σ, D.μ)
        lines!(ax, xs, pdf.(D, xs); color = cucumber, linestyle = :dash)
    end
    f
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
