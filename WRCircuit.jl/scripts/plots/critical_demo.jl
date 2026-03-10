#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
using Bootstrap
DrWatson.@quickactivate
using WRCircuit
using JLD2
using LinearAlgebra
using Optim
using MoreMaps
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin # ! No determinism here unless running on CPU!!
    model = WRCircuit.models.Spatial
    begin # FNS parameters
        rho = 20000
        dx = 0.5
        # sigma_ee = 0.07
        # sigma_ei = 0.095
        # sigma_ie = 0.17
        # sigma_ii = 0.17
        sigma_ee = 0.06  # from decay=7.5
        sigma_ei = 0.07  # from decay=9.5
        sigma_ie = 0.14  # from decay=19
        sigma_ii = 0.14  # from decay=19
        # K_ee = 140
        # K_ei = 160
        # K_ie = 100
        # K_ii = 140
        # K_ee = 200
        # K_ei = 300
        # K_ie = 140
        # K_ii = 190
        K_ee = 260
        K_ei = 340
        K_ie = 225
        K_ii = 290
        nu = 10.0
        n_ext = 100
        Delta_g_K = 0.002
    end
end

begin
    tmax = 55u"s" # * 35s max for l400; 65s for h100
    tmin = 5u"s" # The transient. Simulations always begin at 0
    fixed_params = (; rho,
                    dx,
                    sigma_ee,
                    sigma_ei,
                    sigma_ie,
                    sigma_ii,
                    K_ee,
                    K_ei,
                    K_ie,
                    K_ii,
                    nu,
                    n_ext,
                    Delta_g_K,
                    key = WRCircuit.PRNGKey(52))

    # monitors = ["E.spike", ("E.input", local_idxs)] |> pytuple
    # stat_funcs = Dict("rate" => WRCircuit.stats.firing_rate,
    #                   "susceptibility" => WRCircuit.stats.susceptibility(bin = 10))
end

begin # * Run simulation
    # WRCircuit.brainpy.math.random.seed(52) # ! This is broken.....
    m = model(; fixed_params...) # Key here is broken too... why...
    x = bpsolve(m, tmax; populations = [:E], vars = [:spike, :V, :input],
                transient = tmin) # :I

    @info "Saving data"
    epositions = m.E.positions
    epositions = map(epositions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end
    ipositions = m.I.positions
    ipositions = map(ipositions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end
    tagsave(datadir("critical_demo.jld2"),
            Dict("x" => x,
                 "fixed_params" => fixed_params,
                 "epositions" => epositions,
                 "ipositions" => ipositions), safe = true)
end
