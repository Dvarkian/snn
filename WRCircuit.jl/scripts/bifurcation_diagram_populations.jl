#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using DrWatson
DrWatson.@quickactivate
using PythonCall
using WRCircuit
using Unitful
using Statistics
using TimeseriesTools
using CairoMakie
using LinearAlgebra
using Distributed
using USydClusters
using Term
using SparseArrays

model = WRCircuit.models.balanced.FNSPopulations
modelname = "Balanced"

begin # * Fixed parameters
    N = 50000
    T = 1000.0
end

if false
    m = model(N; g = 4.0, nu_hat = 8.0)
    x = bpsolve(m, T; populations = [:E, :I], vars = [:spike])
end

if false # * Start procs
    if haskey(ENV, "JULIA_DISTRIBUTED") && length(procs()) == 1
        using USydClusters
        ourprocs = USydClusters.Physics.addprocs(20; mem = 64, ncpus = 16,
                                                 project = projectdir())
        @everywhere using WRCircuit
        @everywhere using PythonCall
        @everywhere using Statistics
        @everywhere using TimeseriesTools
        @everywhere model = models.balanced.FNSPopulations
    end

    X = pmap(params) do (g, ν̂)
        # jax.clear_caches()
        # PythonCall.GIL.lock(GC.gc)
        # PythonCall.GC.gc() # Check these work with xla_bridge.get_backend().live_arrays()
        m = model(N; g, nu_hat = ν̂)
        x = bpsolve(m, T; populations = [:E, :I], vars = [:spike])
        mean(order_parameter(x[Population = At(population), Var = At(:spike)]))
    end
end

begin # * Sweep parameters
    order_parameter = firingrate
    population = :E

    ν̂s = range(0, 4, length = 10) |> Dim{:ν̂}
    gs = range(0, 8, length = 25) |> Dim{:g}
    params = map(Iterators.product(ν̂s, gs)) do (ν̂, g)
        Dict("ν̂" => ν̂, "g" => g)
    end
end
if false
    folder = datadir(modelname, "bifurcation_diagram")

    exprs = map(params) do (g, ν̂)
        quote
            using DrWatson
            ENV["JULIA_CONDAPKG_BACKEND"] = "Null" # To make loading faster
            ENV["JULIA_PYTHONCALL_EXE"] = projectdir(".CondaPkg", "env", "bin", "python")
            using WRCircuit
            model = models.balanced.FNSPopulations
            m = model($N; g = $g, nu_hat = $ν̂)
            x = bpsolve(m, $T; populations = [:E], vars = [:spike])
            save(joinpath($folder, savename(Dict("g" => $g, "ν̂" => $ν̂), "jld2")),
                 Dict("x" => x))
        end
    end

    USydClusters.Physics.runscripts(exprs; ncpus = 32, mem = 32,
                                    walltime = 1, project = projectdir(),
                                    qsub_flags = "-q l40s")
end
function produce(D)
    WRCircuit.clear_live_arrays() # Scorched earth
    @unpack g, ν̂ = D
    m = model(N; g = g, nu_hat = ν̂)
    x = bpsolve(m, T; populations = [:E, :I], vars = [:spike])
    x = map(x) do x
        set(x, x |> parent |> SparseMatrixCSC)
    end
    c = mean(cv(x[Population = At(population), Var = At(:spike)]))
    f = mean(firingrate(x[Population = At(population), Var = At(:spike)]))
    Dict("cv" => c, "firing_rate" => f)
end
begin
    folder = datadir(modelname, "bifurcation_diagram")
    O = progressmap(x -> produce_or_load(produce, x, folder; loadfile = true), params;
                    parallel = false, backend = :Term)
end
begin
    o = getindex.(first.(O), "firing_rate")
end
begin
    f = Figure()
    ax = Axis(f[1, 1])
    for ν in ν̂s
        x = o[ν̂ = At(ν)]
        lines!(ax, lookup(x, 1), parent(x))
    end
    f
end
begin # * Plot bifurcation diagram
    f = Figure()
    ax = Axis(f[1, 1])
    heatmap!(ax, log10.(ustripall(o) .+ eps())')
    f
end

save(f, plotdir("bifurcation_diagram_populations.pdf"))
