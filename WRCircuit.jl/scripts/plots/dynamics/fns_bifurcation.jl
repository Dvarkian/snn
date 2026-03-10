#! /bin/bash

# -*- mode: julia -*-
#=
exec $HOME/build/julia-1.11.2/bin/julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# $HOME/build/julia-1.11.2/bin/julia maybe
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin
    model = models.FNS
    modelname = "FNS"

    begin # FNS parameters
        N_e = 4000
        J_e = 0.0008 # Microsiemens
        delta = 4.0
        nu = 10
        n_ext = 80

        omicron = 0.06 # This omicron actually controls the frequency of ping; smaller alpha, higher frequency...
        omega = [0.3 0.47; 0.37 0.65]
    end
end
begin
    parameters = (; N_e, J_e, delta, nu, n_ext)

    deltas = range(2.0, 6.0, length = 10)
    omicrons = range(0.06, 0.06, length = 1)

    T = 10u"s"
    transient = 5u"s"
end
begin # * Generate a bifurcation diagram over deltas and o-scale
    stats = map(omicrons) do omicron
        @info "Simulating for omicron = $omicron"
        WRCircuit.clear_live_arrays()
        begin
            omega_ee, omega_ie, omega_ei, omega_ii = omega .* omicron
            m = model(; parameters..., omega_ee, omega_ie, omega_ei, omega_ii,
                      key = jax.random.PRNGKey(42)) # Build once to get connectivity

            N = m.E.size |> convert2(Vector)
            conn = m.get_connectivity()
            conn = utils.pytree_to_numpy(conn) # Freeze the connectivity
            _params = m.get_input_params() |> convert2(Dict{Symbol, Any})
            model_class = m.__class__

            res = bpsweep(model_class, conn, _params, :delta => deltas;
                          duration = T,
                          transient,
                          populations = [:E],
                          vars = [:spike, :input],
                          num_parallel = 1,
                          batch_size = 20,
                          batch_seed = 42)
            res = res[Population = At(:E), Var = At(:spike)]
        end

        stats = progressmap(eachslice(res, dims = :delta)) do spikes
            begin # * Susceptibility
                dt = 10u"ms"
                x = sum.(coarsegrain(spikes, dt)) # Bin over time
                x = set(x, 𝑡 => mean.(times(x)))
                ρ = mean(collect(x) .> 0, dims = 2) # Fraction of active neurons at each time step
                χ = mean(ρ .^ 2) - mean(ρ)^2
            end
            begin# * Firing rate
                λ = sum(spikes, dims = 𝑡) ./ duration(spikes)
                λ = uconvert.(u"Hz", mean(λ))
            end
            # begin # * Maximum of power spectrum
            #     lfp = unitarylfp(times(spikes), parent(spikes), :E)
            #     lfp = set(lfp, 𝑡 => times(lfp) .* u"ms")
            #     lfp = set(lfp, 𝑡 => uconvert.(u"s", times(lfp)))
            #     lfp = rectify(lfp; dims = 𝑡, tol = 1)
            #     # * Take the power spectrum
            #     s = spectrum(lfp .- mean(lfp))
            #     # * Find the maximum frequency
            #     f_max = findmax(s) |> last
            #     f_max = freqs(s)[f_max]
            # end
            # WRCircuit.clear_live_arrays() # Does this operate @everywhere? Seems not
            return ToolsArray([χ, λ], (Dim{:statistic}([:χ, :λ]),)) # Can only return non-python objects
        end |> stack
    end
    stats = ToolsArray(stats, (Dim{:omicron}(omicrons),)) |> stack
end
begin
    save("fns_bifurcation.jld2", (@strdict stats))
end
begin
    f = Figure(size = (800, 600))
    colorrange = extrema(lookup(stats, :omicron)) .+ [0, 0.001]
    map(enumerate(eachslice(ustripall(stats), dims = :statistic))) do (i, stat)
        statname = refdims(stat) |> only |> only
        ax = Axis(f[i, 1]; ylabel = "$statname")
        map(eachslice(stat, dims = :omicron)) do stat
            omicron = refdims(stat)
            omicron = omicron[dimname.(omicron) .== ["omicron"]] |> only |> only
            stat = upsample(stat, 10; dims = 1)
            lines!(ax, stat; color = omicron, colorrange, label = "o = $omicron")
        end
        axislegend(ax)
    end
    map(contents(f.layout)[1:(end - 1)]) do ax
        hidexdecorations!(ax)
        ax.xgridvisible = true
        ax.xticksvisible = true
    end
    last(contents(f.layout)).xlabel = "δ"
    save("fns_bifurcation.pdf", f)
    f
end
