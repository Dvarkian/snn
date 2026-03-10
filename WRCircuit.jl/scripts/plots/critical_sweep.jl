#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using LinearAlgebra
using Optim
using MoreMaps
using Statistics
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Load sweep parameters
    files = readdir(datadir("critical_sweep"), join = true)
    ps = map(f -> parse_savename(f; connector = string(connector))[2], files)
    deltas = [p["delta"] for p in ps]
end

begin # * Load PSD data
    psd = map(Chart(Threaded(), ProgressLogger()), files) do f
        load(f, "inputs/psd")
    end
    psd = ToolsArray(psd, Dim{:δ}(deltas)) |> stack
end

begin # * Plot Psd for one delta
    p = median(psd[δ = Near(4.0)], dims = Neuron)
    p = dropdims(p, dims = Neuron)

    # * MAPPLE fit
    fmax = 1000
    _p = p[𝑓 = 10u"Hz" .. fmax * u"Hz"]
    _p = logsample(ustripall(_p))
    m = fit(MAPPLE, _p; components = 1, peaks = 1)
    fit!(m, _p)
    _m = predict(m, lookup(_p, 𝑓))

    f = Figure()
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, ustripall(p)[𝑓 = eps() .. fmax])
    lines!(ax, lookup(_p, 𝑓), _m)
    display(f)
    display(m.params.components)
end

begin # * And mad
    mad = map(Chart(Threaded(), ProgressLogger()), files) do f
        load(f, "inputs/mad")
    end
    mad = ToolsArray(mad, Dim{:δ}(deltas)) |> stack
end

begin # * Plot mad for one delta
    p = median(mad[δ = Near(4.0)], dims = Neuron)
    p = dropdims(p, dims = Neuron) |> ustripall

    # * MAPPLE fit
    m = fit(MAPPLE, p; components = 2, peaks = 0)
    fit!(m, p)
    _m = predict(m, lookup(p, 𝑡))

    f = Figure()
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, ustripall(p))
    lines!(ax, lookup(p, 𝑡), _m)
    display(f)
    display(m.params.components)
end

function susceptibility(x; dt = 100u"ms")
    # * First bin into ms bins
    X = groupby(x, 𝑡 => Base.Fix2(WRCircuit.group_dt, dt))

    # * Active neurons
    X = map(X) do x
        sum(x, dims = 𝑡) .> 0
    end

    # * Fraction of active neurons at each time step
    rho = mean.(X)

    # * Susceptibility
    chi = mean(rho .^ 2) - mean(rho)^2
end

begin # * Calculate susceptibility
    spikes = map(Chart(Threaded(), ProgressLogger()), files) do f
        load(f, "spikes")
    end
    χ = map(susceptibility, Chart(Threaded(), ProgressLogger()), spikes)
    χ = ToolsArray(χ, Dim{:δ}(deltas)) |> stack
end
begin
    lines(χ)
end

# begin # * Load distribution
#     distribution = map(Chart(Threaded(), ProgressLogger()), files) do f
#         load(f, "inputs/distribution")
#     end
#     distribution = ToolsArray(distribution, Dim{:δ}(deltas)) |> stack
# end
# begin # * Plot typical distribution
#     p = distribution[δ = Near(4.0)][:, 1600][:]

#     μ = median(p)
#     σ = (quantile(p, 0.75) - quantile(p, 0.25)) / 1.349

#     d = fit(Stable, ustripall(p))
#     d |> display
#     # d = Normal(μ, σ)

#     f = Figure()
#     ax = Axis(f[1, 1], limits = ((-1, 1), nothing),
#               title = "μ = $(round(μ; sigdigits=3)), σ = $(round(σ; sigdigits=3))")
#     hist!(ax, ustripall(p); bins = -2:0.01:2, normalization = :pdf)
#     lines!(ax, -2:0.01:2, pdf.([d], -2:0.01:2); color = :red)
#     display(f)
# end
# begin # * log plot
#     _p = abs.(p .- median(p))
#     bins = 0.01:0.01:2

#     f = Figure()
#     ax = Axis(f[1, 1]; yscale = log10, xscale = log10, limits = (nothing, (1e-2, nothing)))
#     hist!(ax, _p; bins = bins, normalization = :pdf)
#     _d = fit(Stable, vcat(_p[1:5:end], .-_p[1:5:end]))
#     lines!(ax, bins, pdf.([_d], bins) .* 2; color = :red)
#     display(f)
# end
# begin # Plot distribution of fits across neurons
#     ps = map(Chart(Threaded(), ProgressLogger()),
#              eachslice(distribution[δ = Near(4.0)], dims = Neuron)) do p
#         d = fit(Stable, ustripall(p))
#     end
#     αs = getfield.(ps, :α)
#     βs = getfield.(ps, :β)
#     μs = getfield.(ps, :μ)
#     σs = getfield.(ps, :σ)
# end
# begin # * Histogram across neurons
#     f = FourPanel()
#     gs = subdivide(f, 2, 2)

#     for (i, p) in enumerate([:α, :β, :σ, :μ])
#         x = getfield.(ps, p)
#         m = median(x)
#         ax = Axis(gs[i]; title = "$p median = $(round(m; sigdigits=3))")
#         hist!(ax, x; bins = 50, normalization = :pdf)
#         vlines!(ax, [m], color = :red)
#     end
#     display(f)
# end # !!! Maybe calculate a geometric median over these four parameters to get a better idea of the 'average neuron'
