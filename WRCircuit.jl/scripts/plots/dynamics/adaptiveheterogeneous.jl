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

model = models.AdaptiveHeterogeneous
modelname = "AdaptiveHeterogeneous"

begin # * Simulate
    N = 5000
    T = 1000.0

    m = model(N; epsilon = 0.1, J = 0.1) # Try 4.0, nu_hat = 1.0 for gaussian
    X = bpsolve(m, T; populations = [:E, :I], vars = [:spike, :V, :input, :g_K])

    V = X[Var = At(:V)]
    g_K = X[Var = At(:g_K)]
    input = X[Var = At(:input)]
    spikes = X[Var = At(:spike)]
    pop = :E # Excitatory
    lines(V[1][1:1000, 1])
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
    ax = Axis(gs[2]; yscale = log10, limits = (nothing, (1e-8, 1)))
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
