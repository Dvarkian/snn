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
    model = models.WRCircuit
    modelname = "WRCircuit"

    begin # WRCircuit parameters
        dx = 0.75 # mm
        rho = 30000.0
        kernel = distances.GaussianKernel
        J_e = 0.0008 # Microsiemens
        delta = 2.0 # 2.9
        nu = 10.0 # Preserve; the minimum required for spontaneous spiking
        n_ext = 65 # Should be around 10% of total number of inputs??

        sigma_ee = 0.075
        sigma_ei = 0.1
        sigma_ie = 0.2
        sigma_ii = 0.2

        omega_ee = 0.0038
        omega_ie = 0.025

        omega_ei = 0.006 # The relative strength of these two components controls intensity/sparseness of pattern?
        omega_ii = 0.0275
    end
end
begin
    parameters = (; rho, dx, J_e, nu, n_ext, delta, omega_ee, omega_ei, omega_ie, omega_ii,
                  sigma_ee, sigma_ei, sigma_ie, sigma_ii, kernel)

    deltas = range(1.5, 3, length = 50)
    nus = range(10, 10, length = 1)

    T = 10u"s"
    transient = 5u"s"
end
begin
    stats = map(nus) do nu
        @info "Simulating for nu = $nu"
        WRCircuit.clear_live_arrays()
        begin
            m = model(; parameters..., nu, key = jax.random.PRNGKey(42)) # Build once to get connectivity

            N = m.E.size |> convert2(Vector)
            domain = m.E.embedding.domain |> convert2(Vector)
            Δx = domain ./ N
            xs = range.(0 .+ Δx / 2, domain .- Δx / 2, N)

            conn = m.get_connectivity()
            conn = models.WRCircuit.pytree_to_numpy(conn) # Freeze the connectivity
            _params = m.get_input_params() |> convert2(Dict{Symbol, Any})
            model_class = m.__class__

            res = bpsweep(model_class, conn, _params, :delta => deltas;
                          duration = T,
                          transient,
                          populations = [:E],
                          vars = [:spike],
                          num_parallel = 10,
                          batch_size = 10,
                          batch_seed = 42)
            res = res[Population = At(:E), Var = At(:spike)]
        end

        stats = progressmap(eachslice(res, dims = :delta)) do spikes
            begin # * Susceptibility
                dt = 10u"ms"
                x = sum.(coarsegrain(spikes, dt)) # Bin over time
                x = map(eachslice(x, dims = 1)) do x
                    ToolsArray(reshape(x, N...), (X(xs[1]), Y(xs[2])))
                end |> stack
                x = permutedims(x, (3, 1, 2))
                x = set(x, 𝑡 => mean.(times(x)))
                ρ = mean(collect(x) .> 0, dims = (2, 3)) # Fraction of active neurons at each time step
                χ = mean(ρ .^ 2) - mean(ρ)^2
            end
            begin# * Firing rate
                λ = sum(spikes, dims = 𝑡) ./ duration(spikes)
                λ = uconvert.(u"Hz", mean(λ))
            end
            # WRCircuit.clear_live_arrays() # Does this operate @everywhere? Seems not
            return ToolsArray([χ, λ], (Dim{:statistic}([:χ, :λ]),)) # Can only return non-python objects
        end |> stack
    end
    stats = ToolsArray(stats, (Dim{:nu}(nus),)) |> stack
end
begin
    save("fns_bifurcation.jld2", (@strdict stats))
end
begin
    f = Figure(size = (800, 600))
    colorrange = extrema(lookup(stats, :nu)) .+ [0, 0.001]
    map(enumerate(eachslice(ustripall(stats), dims = :statistic))) do (i, stat)
        statname = refdims(stat) |> only |> only
        ax = Axis(f[i, 1]; ylabel = "$statname")
        map(eachslice(stat, dims = :nu)) do stat
            nu = refdims(stat)
            nu = nu[dimname.(nu) .== ["nu"]] |> only |> only
            lines!(ax, stat; color = nu, colorrange, label = "ν = $nu")
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
