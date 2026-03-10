#! /bin/bash
# -*- mode: julia -*-
#=
exec $HOME/build/julia-1.11.2/bin/julia -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # * Model parameters
    model = WRCircuit.models.Spatial

    dx = 0.5
    rho = 20000.0
    kernel = WRCircuit.distances.GaussianKernel
    J_ee = 0.0008
    # delta = 3.5
    nu = 7.0
    n_ext = 100

    sigma_ee = 0.04
    sigma_ei = 0.05
    sigma_ie = 0.12
    sigma_ii = 0.12

    k = 0.9
    K_ee = round(Int, 130 * k)
    K_ie = round(Int, 180 * k)
    K_ei = round(Int, 200 * k)
    K_ii = round(Int, 250 * k)

    fixed_params = (; dx, rho, kernel, J_ee, nu, n_ext,
                    sigma_ee, sigma_ei, sigma_ie, sigma_ii,
                    K_ee, K_ie, K_ei, K_ii)
end
begin
    tmax = 10u"s"
    tmin = 0u"s" # The transient. Simulations always begin at 0

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz

    monitors = ("E.spike",)
    stat_funcs = Dict(#"rate" => WRCircuit.stats.firing_rate,
                      #   "susceptibility" => WRCircuit.stats.susceptibility(bin = 10),
                      #   "spike_spectrum" => WRCircuit.stats.spike_spectrum(n_segments = 10),
                      #   "temporal_average" => WRCircuit.stats.temporal_average,
                      #   "mua" => WRCircuit.stats.mua(bin = ustrip(to_ms(mua_dt))),
                      "monitor" => WRCircuit.stats.monitor)
end
begin# * Generate dict of parameter vectors
    sweep = (; delta = range(2.0, 6.0, length = 9))
    pnames = map(string, keys(sweep))
    pvals = stack(Iterators.product(values(sweep)...), dims = 1)
    sweep_params = Dict{String, Any}(zip(pnames, eachcol(pvals))) # Now a good shape for jax
    n_iters = length(first(values(sweep_params)))
    jax_keys = WRCircuit.jax.random.split(WRCircuit.jax.random.PRNGKey(42), n_iters)
    sweep_params["key"] = WRCircuit.numpy.array.(jax_keys) # * So that each run is independent
end
begin # * Create sweep function
    run = WRCircuit.stats.create_run(model, pydict(fixed_params), monitors,
                                     ustrip(to_ms(tmax)),
                                     ustrip(to_ms(tmin)))
    stats_run = WRCircuit.stats.create_stats_run(run, pydict(stat_funcs))
end
begin # * Run simulation
    stats, sweep_parameters = WRCircuit.stats.partial_vmap(stats_run, batch_size = 5)(pydict(sweep_params))
end
begin # * Loop over stats and construct spike trains
    ts = range(0u"s", tmax,
               step = convert2(Float64)(WRCircuit.brainpy.share["dt"]) * u"ms")[2:end]
    deltas = sweep_parameters["delta"] |> convert2(Vector) |> Dim{:delta}
    spikes = stats["monitor"]["E.spike"] |> PyArray
    # spikes = permutedims(spikes, (2, 1, 3)) # * Shape is now (t, delta, n)
end
begin # * Infer spatial grid
    dx = fixed_params[:dx]
    n = sqrt(size(spikes, 3)) |> Int
    δx = dx / n
    x = range(δx / 2, dx, step = δx)
    grid = Iterators.product(x, x) |> collect
end
begin # * Reshape spike trains
    spikes = reshape(spikes, (size(spikes, 1), size(spikes, 2), n, n))
    spikes = ToolsArray(spikes, (deltas, 𝑡(ts), DimensionalData.X(x), DimensionalData.Y(x)))
    spikes = rectify(spikes, dims = :delta; tol = 10)
    spikes = permutedims(spikes, (2, 3, 4, 1)) # * Shape is now (t, delta, x, y)
end
begin # * Convert to a list of (x, y) points for each time t
    tbin = 1u"ms"
    tbins = range(first(times(spikes)), last(times(spikes)), step = tbin) |> intervals
    spiketimes = groupby(spikes, 𝑡 => Bins(tbins))
    spiketimes = map(spiketimes) do x
        dropdims(any(x, dims = 𝑡), dims = 𝑡)
    end
    spiketimes = permutedims(stack(spiketimes), (4, 1, 2, 3))
    spiketimes = set(spiketimes, 𝑡 => mean.(times(spiketimes)))
    spikeidxs = map(eachslice(spiketimes, dims = (𝑡, :delta))) do x
        is = findall(x)
        isempty(is) ? [Point2f([NaN, NaN])] : Point2f.(grid[is])
    end
end
# begin # * Set up plot
#     d = 6
#     delta = lookup(spikeidxs, :delta)[d]

#     f = Figure(size = (800, 600))
#     Idxs = spikeidxs[delta = Near(delta)]
#     idxs = Observable(first(Idxs))
#     ax = Axis(f[1, 1], title = "δ = $delta", limits = ((0, dx), (0, dx)))
#     scatter!(ax, idxs, markersize = 5)
#     f
# end
# begin # * Animation loop
#     record(f, "WRCircuit_sweep.mp4", eachindex(Idxs), framerate = 48) do i
#         idxs[] = Idxs[i]
#     end
# end
begin # * Animate all at once
    nrows = 3
    ncols = ceil(Int, length(deltas) ÷ nrows)
    f = Figure(size = (ncols * 200, nrows * 200))
    gs = subdivide(f, nrows, ncols)
    idxs = map(eachslice(spikeidxs, dims = :delta)) do x
        x = x[1] |> parent |> Observable
    end
    for (i, delta) in enumerate(deltas)
        ax = Axis(gs[i], title = "δ = $delta", limits = ((0, dx), (0, dx)), aspect = 1)
        hidedecorations!(ax)
        scatter!(ax, idxs[i], markersize = 3)
    end
    f
end
begin # * Animate
    using Term.Progress

    ts = 1:5:size(spikeidxs, 1)
    pbar = ProgressBar()
    job = addjob!(pbar; N = length(ts))
    with(pbar) do
        record(f, "WRCircuit_sweep_all_new.mp4", ts, framerate = 24) do t
            for (i, delta) in enumerate(deltas)
                idxs[i][] = spikeidxs[t, i]
            end
            Term.update!(job)
        end
    end
    f
end
# begin # * try a heatmap approach
#     f = Figure(size = (800, 600))
#     ax = Axis(f[1, 1], title = "δ = $delta", limits = ((0, dx), (0, dx)))
#     XX = spiketimes[delta = Near(delta)]
#     xx = XX[1, :, :] |> parent |> Observable
#     heatmap!(ax, x, x, xx, colormap = seethrough(:turbo), colorrange = (0, 1))
#     f
# end
# begin # * Animation loop
#     record(f, "WRCircuit_sweep_heatmap.mp4", axes(XX)[1], framerate = 48) do i
#         xx[] = XX[i, :, :] |> parent
#     end
# end
