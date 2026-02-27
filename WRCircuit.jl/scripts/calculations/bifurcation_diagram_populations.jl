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

model = models.balanced.FNSPopulations
modelname = "Balanced"

begin # * Fixed parameters
    N = 50000
    T = 1000.0
end

begin # * Sweep parameters
    ν̂s = range(0, 4, length = 2) |> Dim{:ν̂}
    gs = range(0, 8, length = 2) |> Dim{:g}
    params = map(Iterators.product(ν̂s, gs)) do (ν̂, g)
        Dict("ν̂" => ν̂, "g" => g)
    end
end

# begin
#     m = model(N; g = 4.0, nu_hat = 8.0)
#     x = bpsolve(m, T; populations = [:E, :I], vars = [:spike, :V])
#     spikes = x[Var = At(:spike)]
#     V = x[Var = At(:V)]
#     spikes = map(spikes) do x # Sparse arrays are smaller
#         set(x, x |> parent |> SparseMatrixCSC)
#     end
#     V = map(V) do x # Float32 is smaller
#         set(x, map(Float32, parent(x)))
#     end
# end

if haskey(ENV, "JULIA_DISTRIBUTED") && length(procs()) == 1 # ? You should start this on gpu node h100 for maximum resource utilization
    using USydClusters
    ourprocs = USydClusters.Physics.addprocs(1; mem = 32, ncpus = 32,
                                             project = projectdir(), qsub_flags = `-q l40s`)
    @everywhere using DrWatson
    @everywhere using WRCircuit
    @everywhere using PythonCall
    @everywhere using Statistics
    @everywhere using TimeseriesTools
    @everywhere using SparseArrays
    @everywhere model = models.balanced.FNSPopulations
end

begin
    folder = datadir(modelname, "bifurcation_diagram")
    pmap(params) do param
        WRCircuit.clear_live_arrays()
        @unpack g, ν̂ = param
        m = model(N; g = g, nu_hat = ν̂)
        x = bpsolve(m, T; populations = [:E, :I], vars = [:spike, :V])

        spikes = x[Var = At(:spike)]
        V = x[Var = At(:V)]
        # spikes = map(spikes) do x # Sparse arrays are smaller
        #     set(x, x |> parent |> SparseMatrixCSC)
        # end
        # V = map(V) do x # Float32 is smaller
        #     set(x, map(Float32, parent(x)))
        # end

        filename = savename((@strdict g ν̂), "jld2"; connector)
        D = @strdict g ν̂#  spikes V
        save(joinpath(folder, filename), D)
    end
end

# if false
#     folder = datadir(modelname, "bifurcation_diagram")

#     exprs = map(params) do (g, ν̂)
#         quote
#             using DrWatson
#             ENV["JULIA_CONDAPKG_BACKEND"] = "Null" # To make loading faster
#             ENV["JULIA_PYTHONCALL_EXE"] = projectdir(".CondaPkg", "env", "bin", "python")
#             using WRCircuit

#             model = models.balanced.FNSPopulations
#             m = model($N; g = $g, nu_hat = $ν̂)
#             x = bpsolve(m, $T; populations = [:E, :I], vars = [:spike, :V])

#             spikes = x[Var = At(:spike)]
#             V = x[Var = At(:V)]
#             spikes = map(spikes) do x # Sparse arrays are smaller
#                 set(x, x |> parent |> SparseMatrixCSC)
#             end
#             V = map(V) do x # Float32 is smaller
#                 set(x, map(Float32, parent(x)))
#             end

#             save(joinpath($folder, savename(Dict("g" => $g, "ν̂" => $ν̂), "jld2")),
#                  Dict("x" => x))
#         end
#     end

#     USydClusters.Physics.runscripts(exprs; ncpus = 32, mem = 32,
#                                     walltime = 1, project = projectdir(),
#                                     qsub_flags = "-q l40s")
# end
