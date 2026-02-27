using Bootstrap
using Normalization
using StatsBase
using Random
import Accessors: @set

export firingrate, cv, plotdir, connector, bootstrapaverage, bootstrapmedian,
       structurefunction, histcounts, timebins, unitarylfp, to_ms, to_mm, pytree2dict

function _preamble()
    quote
        using DrWatson
        using PythonCall
        using Unitful
        using Statistics
        using Bootstrap
        using TimeseriesTools
        using CairoMakie
        using TimeseriesMakie
        using Foresight
        using LinearAlgebra
        using Distributed
        using Term
        using MoreMaps
        using SparseArrays
        using MeanSquaredDisplacement
        using Distributions
        using StableDistributions
        using JLD2
        using Random
    end
end
macro preamble()
    _preamble()
end
@preamble

bpdt() = pyconvert(Float32, brainpy.share["dt"])

plotdir(args...) = projectdir("plots", args...)
export plotdir

const connector = '&'

const UnivariateSpikeTrain = Base.typeintersect(SpikeTrain, UnivariateTimeseries)
const MultivariateSpikeTrain = Base.typeintersect(SpikeTrain, MultivariateTimeseries)
function firingrate(x::UnivariateSpikeTrain)
    λ = sum(x) / duration(x)
    uconvert(unit(eltype(x)) * u"Hz", λ)
end
function firingrate(X::MultivariateSpikeTrain)
    firingrate.(eachslice(X, dims = dims(X)[2:end]))
end
function cv(x::UnivariateSpikeTrain)
    ts = times(x[x])
    isis = diff(ts)
    if length(isis) < 2
        return 0.0 # No spikes, mean isi is Inf, say cv is 0
    else
        return std(isis) / mean(isis)
    end
end
function cv(X::MultivariateSpikeTrain)
    cv.(eachslice(X, dims = dims(X)[2:end]))
end

"""
Structure Function Analysis of Turbulent Flows

Generates a vector of differences at different lags, that can be used ot compute
lag-dependent statistics of the difference distribution
"""
function structurefunction(x::AbstractMatrix, τ::Int)
    Δ = @views x[(τ + 1):end, :] - x[1:(end - τ), :]
end
function structurefunction(x::AbstractVector, τ::Int)
    Δ = @views x[(τ + 1):end] - x[1:(end - τ)]
end
function structurefunction(x::RegularTimeseries, τ::Int; kwargs...)
    structurefunction(parent(x), τ; kwargs...)
end
function structurefunction(x::RegularTimeseries, τ::AbstractFloat)
    structurefunction(x, Int(τ ÷ step(x)); kwargs...)
end
function structurefunction(x::AbstractArray, τs)
    progressmap(τs) do τ
        structurefunction(x, τ)
    end
end
function histcounts(x, edges)
    H = fit(Histogram, x[:], edges)
    # H = normalize(H, mode = :density)
    centers = (edges[1:(end - 1)] + edges[2:end]) ./ 2
    N = sum(H.weights) .* step(edges)
    y = H.weights ./ N
    return ToolsArray(y, Dim{:bin}(centers))
end

# * Move to TimeseriesTools
function timebins(x::RegularTimeseries, τ::Number)
    un = unit(eltype(times(x)))
    if unit(τ) != un && unit(τ) == NoUnits
        τ = τ * un
    end
    tbins = range(first(times(x)), last(times(x)), step = τ)
    tbins = [i .. i + τ for i in tbins]
    x = DimensionalData.groupby(x, 𝑡 => Bins(tbins))
end
function TimeseriesTools.coarsegrain(x::RegularTimeseries, τ::Number)
    negdims = setdiff(1:ndims(x), dimnum(x, 𝑡))
    x = timebins(x, τ)
    x = eachslice.(x; dims = negdims |> Tuple) |> stack
    x = permutedims(x, circshift(1:ndims(x), -1))
end
TimeseriesTools.coarsegrain(x::UnivariateRegular, τ::Number) = timebins(x, τ)

# import StableDistributions.fit
# function clampedfit(::Type{<:Stable}, x::AbstractArray{<:Real})
#     α₀, _β₀, σ₀, δ₀ = fit_quantile(Stable, x)
#     u = exp.(LinRange(α₀ > 1 ? -2 : 10α₀ - 12, 0, 10))
#     ecf = [mean(cis(t * (val - δ₀) / σ₀) for val in x) for t in u] # ecf of normalized data
#     r, θ = abs.(ecf), angle.(ecf)

#     mU, mR = [ones(10) -log.(u)], -log.(-log.(r))
#     b, αₑₛₜ = (mU' * mU) \ (mU' * mR) # ols regression

#     αₑₛₜ >= 2 && return convert(Stable, fit(Normal, x))

#     σ₁ = exp(b / αₑₛₜ)
#     η(u) = tan(αₑₛₜ * π / 2) * (u - u^αₑₛₜ) # u > 0
#     mΘ = [η.(u) u]
#     c, δ₁ = (mΘ' * mΘ) \ (mΘ' * θ) # ols regression
#     βₑₛₜ = -c / exp(b)
#     βₑₛₜ = clamp(βₑₛₜ, -1, 1)
#     μ₁ = δ₁ - βₑₛₜ * σ₁ * tan(αₑₛₜ * π / 2) # back to type-1 parametrization
#     σₑₛₜ = σ₀ * σ₁ # unnormalize
#     μₑₛₜ = σ₀ * μ₁ + δ₀ # unnormalize

#     return Stable(αₑₛₜ, βₑₛₜ, σₑₛₜ, μₑₛₜ)
# end

# ? Unitary LFP (Telenczuck 2020)

to_ms(x::Real) = x * u"ms" # Assume ms already
to_ms(x::Quantity) = uconvert(u"ms", x)
to_mm(x::Real) = x * u"mm" # Assume mm already
to_mm(x::Quantity) = uconvert(u"mm", x)

function check_inputs(times, spikes, spike_type)
    if !(spike_type in [:E, :I])
        error("spike_type must be 'exc' or 'inh'.")
    end
    if ndims(spikes) != 2
        error("spikes must be 2D.")
    end
    if length(times) != size(spikes, 1)
        error("Mismatch between length of times and first dimension of spikes.")
    end
end

function generate_positions(n, xmax, ymax, seed)
    rng = isnothing(seed) ? Random.GLOBAL_RNG : MersenneTwister(seed)
    return rand(rng, Float32, n) .* xmax, rand(rng, Float32, n) .* ymax
end

function get_amplitudes(location)
    if location === :soma
        return 0.48, 3.0
    elseif location === :deep
        return -0.16, -0.2
    elseif location === :superficial
        return 0.24, -1.2
    elseif location === :surface
        return -0.08, 0.3
    else
        error("Location not implemented.")
    end
end

function unitarylfp(times, spikes, spike_type;
                    xmax = 0.2, ymax = 0.2, va = 200.0, lambda_ = 0.2,
                    sig_i = 2.1, sig_e = 2.1 * 1.5, location = :soma,
                    seed = nothing)
    check_inputs(times, spikes, spike_type)
    times = map(ustrip ∘ to_ms, times)
    times = map(Float32, times)
    nt, nn = size(spikes)
    px, py = generate_positions(nn, xmax, ymax, seed)
    dist = sqrt.((px .- xmax / 2) .^ 2 .+ (py .- ymax / 2) .^ 2)
    ae, ai = get_amplitudes(location)
    A = exp.(-dist ./ lambda_) .* (spike_type === :E ? ae : ai)
    delay = 10.4 .+ dist ./ va
    delay = delay
    τ = spike_type === :E ? 2 * sig_e^2 : 2 * sig_i^2
    τ = Float32(τ)
    idxs = findall(==(1), spikes)
    iis = [I[1] for I in idxs]
    ids = [I[2] for I in idxs]
    tts = times[iis] .+ delay[ids]
    tts = map(Float32, tts)
    amps = map(Float32, A[ids])
    time_mat = @. times' - tts
    exp_mat = @. exp(-time_mat^2 / τ)
    weighted = amps .* exp_mat
    x = vec(sum(weighted, dims = 1))
    return TimeseriesTools.Timeseries(x, times * u"ms")
end

# * Working with 'pytrees'
function pytree2dict(d::Dict)
    d = deepcopy(d)
    for (k, v) in d
        if k isa Py
            _k = pyconvert(String, k)
            delete!(d, k)
            k = _k
        end
        k = Symbol(k)
        if v isa Dict
            d[k] = pytree2dict(v)
        elseif string(pytype(v)) == "<class 'dict'>"
            d[k] = pytree2dict(Dict{Any, Any}(PyDict(v)))
        elseif string(pytype(v)) == "<class 'tuple'>"
            d[k] = pyconvert(Array, v)
        elseif string(pytype(v)) == "<class 'numpy.ndarray'>"
            d[k] = pyconvert(Array, v)
        elseif string(pytype(v)) == "<class 'list'>"
            d[k] = pyconvert(Array, v)
        elseif string(pytype(v)) == "<class 'float'>"
            d[k] = pyconvert(Float32, v)
        elseif string(pytype(v)) == "<class 'int'>"
            d[k] = pyconvert(Int32, v)
        elseif string(pytype(v)) == "<class 'str'>"
            d[k] = pyconvert(String, v)
        elseif string(pytype(v)) == "<class 'NULL'>"
            d[k] = nothing
        else
            @warn "Unknown type $k=>$(pytype(v)), converting to string"
            d[k] = "$v"
        end
    end
    return d
end
function pytree2dict(d::Py; kwargs...)
    d = Dict{Any, Any}(PyDict(d))
    return pytree2dict(d; kwargs...)
end
function defaults(model_class::Py; kwargs...)
    name = model_class.__name__
    m = model_class()
    d = Dict{Any, Any}(PyDict(m.to_dict()))
    d = d[name]
    pytree2dict(d; kwargs...)
end

function group_dt(x::T, dt) where {T}
    round(x / dt) * dt
end
function compute_rates(spikes::SpikeTrain, dt)
    rates = groupby(spikes, 𝑡 => Base.Fix2(group_dt, dt))
    rates = map(rates) do r
        dropdims(sum(r, dims = 𝑡), dims = 𝑡) ./ uconvert(u"s", dt)
    end |> stack
    rates = permutedims(rates, (𝑡, Neuron))
    rates = rectify(rates, dims = 𝑡)
end

function log10spectrum(x::AbstractSpectrum)
    fs = map(log10, lookup(x, 𝑓))
    set(map(log10, x), 𝑓 => Log10𝑓(fs))
end

function convert2(u::Unitful.Units, x::AbstractRange)
    a = uconvert(u, first(x))
    b = uconvert(u, last(x))
    return range(a, b, length = length(x))
end
