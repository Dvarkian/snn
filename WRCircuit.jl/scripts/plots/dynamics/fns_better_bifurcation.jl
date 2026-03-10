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

begin
    model = WRCircuit.models.FNS
    begin # FNS parameters
        N_e = 4000
        J_e = 0.0008 # Microsiemens
        nu = 8
        n_ext = 100
        K_ee = 72
        K_ei = 88
        K_ie = 28
        K_ii = 39
    end
end
begin
    tmax = 10u"s"
    tmin = 1u"s" # The transient. Simulations always begin at 0
    fixed_params = (; N_e, J_e, n_ext, K_ee, K_ie, K_ei, K_ii)

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz

    monitors = ("E.spike", "E.V", "E.input")
    stat_funcs = Dict("rate" => WRCircuit.stats.firing_rate,
                      "susceptibility" => WRCircuit.stats.susceptibility(bin = 10),
                      "spike_spectrum" => WRCircuit.stats.spike_spectrum(n_segments = 10),
                      "temporal_average" => WRCircuit.stats.temporal_average,
                      "grand_distribution" => WRCircuit.stats.grand_distribution(n_bins = 1000),
                      "mua" => WRCircuit.stats.mua(bin = ustrip(to_ms(mua_dt))))
end
begin# * Generate dict of parameter vectors
    sweep = (;
             delta = range(1.0, 7.0, length = 60),
             nu = range(5.0, 10.0, length = 5))
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
    stats, sweep_parameters = WRCircuit.stats.progress_vmap(stats_run, batch_size = 5)(pydict(sweep_params))
end
begin
    WRCircuit.stats.save("fns_better_bifurcation.pickle",
                         (stats, sweep_params, fixed_params))
end
if false
    begin # * Load stats
        load_stats, sweep_parameters, fixed_parameters = WRCircuit.stats.load("fns_better_bifurcation.pickle")
    end
    begin
        begin # * Extract a ToolsArray of one statistic
            deltas = sweep_parameters["delta"] |> convert2(Array)
            nus = sweep_parameters["nu"] |> convert2(Array)
            deltas = reshape(deltas, (length(unique(deltas)), length(unique(nus))))
            nus = reshape(nus, (length(unique(deltas)), length(unique(nus))))
            rate = load_stats["rate"]["E.spike"] |> convert2(Array)
            rate = reshape(rate, size(deltas)..., size(rate, 2))
            meanrate = dropdims(mean(rate, dims = 3), dims = 3)
        end
        begin
            f = Figure()
            ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "ν")
            colorrange = extrema(meanrate)
            heatmap!(ax, sort(unique(deltas)), sort(unique(nus)), meanrate; colorrange,
                     colormap = :viridis)
            Colorbar(f[1, 2]; colormap = :viridis, colorrange,
                     label = "mean rate (Hz)")
            f
        end
    end

    begin # * Load mua
        _mua = stats["mua"]["E.spike"] |> convert2(Array)
        ts = range(tmin, stop = tmax, step = uconvert(u"s", mua_dt))[2:end] |> ustripall
        deltas = sweep_parameters["delta"] |> convert2(Array)
    end
    begin
        mua = Timeseries(_mua', ts, Dim{:delta}(deltas))
        mua = mua .- mean(mua, dims = 𝑡)
        spectra = spectrum(mua)
    end
end
