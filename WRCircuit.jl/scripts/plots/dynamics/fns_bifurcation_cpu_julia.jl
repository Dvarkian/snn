#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# ENV["WRCircuit_BACKEND"] = "cpu" # ! This does nothing if set here...
# ENV["CUDA_VISIBLE_DEVICES"] = ""
using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using Suppressor
WRCircuit.@preamble
set_theme!(foresight(:physics))

begin
    model = models.WRCircuit
    modelname = "WRCircuit"

    begin # Shencong Parameters
        delta = 0.007 # Grid spacing
        dx = 64 * delta # Originally 64*delta # 64 x 64 integer grid, 7um spacing
        rho = 20000
        p_ee = 0.26 #0.8 # These have been tweaked in order to match the mean sum of weights without explicitly setting between-population weight strengths
        p_ei = 0.23 #0.7
        p_ie = 0.13 #0.4
        p_ii = 0.19 #0.57
        sigma_ee = 7.5 * delta
        sigma_ei = 9.5 * delta
        sigma_ie = 19 * delta
        sigma_ii = 19 * delta
        kernel = distances.ExponentialKernel
        J_e = 0.0008 # Microsiemens
        delta = 2.5
        nu = 10
        n_ext = 70 # 200
    end
end
begin
    @info "Building model"
    parameters = (; rho,
                  dx,
                  J_e,
                  nu,
                  n_ext,
                  delta,
                  p_ee,
                  p_ei,
                  p_ie,
                  p_ii,
                  sigma_ee,
                  sigma_ei,
                  sigma_ie,
                  sigma_ii)
    m = model(; key = jax.random.PRNGKey(42),
              kernel = distances.ExponentialKernel,
              parameters...)
    XX = bpsolve(m, 2000u"ms"; populations = [:E], vars = [:spike])

    spikes = XX[Var = At(:spike)]
end
begin # * Animate spikes
    dt = 10u"ms"
    N = m.E.size |> convert2(Vector)
    domain = m.E.embedding.domain |> convert2(Vector)
    Δx = domain ./ N
    xs = range.(0 .+ Δx / 2, domain .- Δx / 2, N)

    f = Figure()
    ax = Axis(f[1, 1], aspect = DataAspect())
    x = spikes[1] # Excitatory spikes
    x = sum.(coarsegrain(x, dt)) # Bin over time
    x = set(x, 𝑡 => mean.(times(x))) |> ustripall # To unitless seconds
    h = Observable(parent(x[1, :]))
    H = lift(h) do h
        reshape(h, N...)
    end
    _t = Observable(first(times(x)))
    colorrange = extrema(x)
    heatmap!(ax, xs..., H; colorrange, colormap = seethrough(reverse(cgrad(:inferno))))
    rm("network.mp4"; force = true)
    @info "Recording animation"
    record(f, "network.mp4", eachindex(times(x)), framerate = 12) do i
        h[] = parent(x[i, :])
        _t[] = times(x)[i]
    end
    @info "Animation saved to `./network.mp4`"
end

# N = m.E.size |> convert2(Vector)
# domain = m.E.embedding.domain |> convert2(Vector)
# Δx = domain ./ N
# xs = range.(0 .+ dx / 2, domain .- dx / 2, N)

# begin
#     addprocs(1; env = ["WRCircuit_BACKEND" => "cpu", "CUDA_VISIBLE_DEVICES" => ""])
#     @everywhere ENV["JULIA_CONDAPKG_OFFLINE"] = true
#     # @everywhere ENV["TF_CPP_MIN_LOG_LEVEL"] = 0
#     # @everywhere ENV["JAX_PLATFORM_NAME"] = "cpu"
#     @everywhere ENV["WRCircuit_BACKEND"] = "cpu" # Does not seem to work after startup
#     @everywhere using WRCircuit
#     @everywhere using JLD2
#     @everywhere jax.default_device = jax.devices("cpu")[0]
#     @everywhere brainpy.math.set_platform("cpu")
# end
begin
    deltas = range(1.5, 2.5, length = 15)
    T = 15u"s"
    transient = 5000u"ms"
end
begin
    conn = m.get_connectivity() # * Now how to copy this over without running into non-hashable type issues?
    out = pmap(deltas) do delta
        @info "δ = $delta"
        m̂ = models.WRCircuit(; key = jax.random.PRNGKey(42),
                              kernel = distances.ExponentialKernel,
                              parameters...,
                              delta,
                              copy_conn = conn)
        res = bpsolve(m̂, T; populations = [:E], vars = [:spike], transient)
        N = m̂.E.size |> convert2(Vector)
        spikes = res[Var = At(:spike)][Population = At(:E)]
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
            λ = sum(spikes, dims = 𝑡) ./ duration(spikes)
            λ = uconvert.(u"Hz", mean(λ))
        end
        # WRCircuit.clear_live_arrays() # Does this operate @everywhere? Seems not
        return χ, λ # Can only return non-python objects
    end
end
begin
    χ = ToolsArray(first.(out), (Dim{:delta}(deltas),))
    λ = ToolsArray(last.(out), (Dim{:delta}(deltas),))
    save("fns_bifurcation.jld2", (@strdict χ λ))
end
begin
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "Susceptibility")
    lines!(ax, χ)
    ax = Axis(f[2, 1]; xlabel = "δ", ylabel = "Mean firing rate (Hz)")
    lines!(λ)
    f
end
