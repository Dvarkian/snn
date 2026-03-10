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

    begin # Shencong Parameters
        delta = 0.007 # Grid spacing
        dx = 65 * delta # Originally 64*delta # 64 x 64 integer grid, 7um spacing
        rho = 20000
        p_ee = 0.2 #0.8 # These have been tweaked in order to match the mean sum of weights without explicitly setting between-population weight strengths
        p_ei = 0.175 #0.7
        p_ie = 0.1 #0.4
        p_ii = 0.14 #0.57
        sigma_ee = 7.5 * delta
        sigma_ei = 9.5 * delta
        sigma_ie = 19 * delta
        sigma_ii = 19 * delta
        kernel = distances.ExponentialKernel
        J_e = 0.0008 # Microsiemens
        delta = 3
        nu = 10
        n_ext = 70 # 200
    end
end
begin
    parameters = (; rho, dx, J_e, nu, n_ext, delta, p_ee, p_ei, p_ie, p_ii, sigma_ee,
                  sigma_ei, sigma_ie, sigma_ii, kernel)

    deltas = range(1, 5, length = 20)
    nus = range(5, 15, length = 3)

    T = 15u"s"
    transient = 5u"s"
end
begin
    stats = map(nus) do nu
        @info "Simulating for nu = $nu"
        WRCircuit.clear_live_arrays()
        m = model(; key = jax.random.PRNGKey(42),
                  parameters..., nu)
        brainpy.reset_state(m)
        N = m.E.size |> convert2(Vector)
        domain = m.E.embedding.domain |> convert2(Vector)
        Δx = domain ./ N
        xs = range.(0 .+ Δx / 2, domain .- Δx / 2, N)

        begin
            res = bpsweep(m, :delta, deltas;
                          duration = T,
                          transient,
                          populations = [:E],
                          vars = [:spike],
                          num_parallel = length(deltas))
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
    colorrange = extrema(lookup(stats, :nu))
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
