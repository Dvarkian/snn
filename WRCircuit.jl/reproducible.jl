# Set JAX determinism flags BEFORE importing any JAX/BrainPy code
using DrWatson
DrWatson.@quickactivate
using WRCircuit

# Set up a fixed seed
model = WRCircuit.models.Spatial
seed = 42
key = WRCircuit.PRNGKey(seed)

# Simulation parameters (smaller network for speed)
params = (rho = 20000,
          dx = 0.5,
          key = key)

duration = 10000.0  # ms, short for quick testing
transient = 0.0  # No transient for this test

println("Creating model 1...")
model1 = model(; params...)

println("\nRunning model 1.")

# Run model 1
# runner1 = WRCircuit.bprun(model1, duration; monitors = ("E.spike", "I.spike"))
x1 = bpsolve(model1, duration; populations = [:E], vars = [:spike, :V, :input])

println("Creating model 2...")
model2 = model(; params...)

println("\nRunning model 2.")

# Run model 2
# runner2 = WRCircuit.bprun(model2, duration; monitors = ("E.spike", "I.spike"))
x2 = bpsolve(model2, duration; populations = [:E], vars = [:spike, :V, :input])

# Extract spike data
# e_spikes_1 = runner1.mon["E.spike"] |> convert2(Array)
# i_spikes_1 = runner1.mon["I.spike"] |> convert2(Array)
# e_spikes_2 = runner2.mon["E.spike"] |> convert2(Array)
# i_spikes_2 = runner2.mon["I.spike"] |> convert2(Array)
e_match = x1[3] ≈ x2[3] # ! PASSES!

# Compare outputs
# e_match = e_spikes_1 == e_spikes_2
# i_match = i_spikes_1 == i_spikes_2

println("\n" * "="^50)
println("REPRODUCIBILITY TEST RESULTS")
println("="^50)
println("Excitatory spikes match: $(e_match): sum = $(sum(x1[3]))")

if e_match
    println("\n✓ SUCCESS: Simulations are deterministic!")
else
    println("\n✗ FAILURE: Simulations differ!")
end

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

begin
    model = WRCircuit.models.Spatial
    begin # FNS parameters
        rho = 20000
        dx = 0.5
        sigma_ee = 0.06  # from decay=7.5
        sigma_ei = 0.07  # from decay=9.5
        sigma_ie = 0.14  # from decay=19
        sigma_ii = 0.14  # from decay=19
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
    tmax = 3u"s" # * Bump up
    tmin = 1u"s" # The transient. Simulations always begin at 0
    fixed_params = (; rho,
                    dx,
                    # sigma_ee,
                    # sigma_ei,
                    # sigma_ie,
                    # sigma_ii,
                    # K_ee,
                    # K_ei,
                    # K_ie,
                    # K_ii,
                    # nu,
                    # n_ext,
                    # Delta_g_K,
                    key = key)
end

begin # * Run simulation
    WRCircuit.brainpy.math.random.seed(52)
    params1 = (; rho, dx, key = WRCircuit.PRNGKey(52))
    m = model(; params1...)
    x = bpsolve(m, tmax; populations = [:E], vars = [:spike, :V, :input],
                transient = tmin)
end

begin # * Run simulation
    WRCircuit.brainpy.math.random.seed(52)
    params2 = (; rho, dx, key = WRCircuit.PRNGKey(52))
    m = model(; params2...)
    x2 = bpsolve(m, tmax; populations = [:E], vars = [:spike, :V, :input],
                 transient = tmin)
    x[3] ≈ x2[3] || error("Results do not match!")
end

# # Run the SAME model twice from the SAME initial key
# key = WRCircuit.PRNGKey(52)
# params = (rho = 20000, dx = 0.5[3], key = key)

# m1 = model(; params...)
# x1 = bpsolve(m1, tmax; populations = [:E], vars = [:spike, :V, :input], transient = tmin)

# m2 = model(; params...)  # Create again with same key
# x2 = bpsolve(m2, tmax; populations = [:E], vars = [:spike, :V, :input], transient = tmin)

# @info "Results match: $(x1[3] ≈ x2[3])"
