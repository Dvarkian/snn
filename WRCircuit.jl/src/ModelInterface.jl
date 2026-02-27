using Statistics
using TimeseriesTools
using DimensionalData
using IntervalSets
using Unitful
export bprun, bpsolve, bpsweep, Neuron, Population, Monitor

DimensionalData.@dim Neuron ToolsDim
DimensionalData.@dim Population ToolsDim
DimensionalData.@dim Monitor ToolsDim

function python_reshape(A, ds...)
    return PermutedDimsArray(reshape(A, ds), reverse(1:length(ds)))
end

function bprun(net::Py, time; monitors = ("E.spike", "I.spike", "E.V", "I.V"), jit = true,
               kwargs...)
    runner = brainpy.DSRunner(net; monitors, jit, kwargs...)
    runner.run(time)
    return runner
end
function bpsolve(net::Py, time::Quantity; kwargs...)
    bpsolve(net, ustrip(uconvert(u"ms", time)); kwargs...)
end

"""
Extract the monitored variables as julia arrays
"""
function bpformat(runner; populations, vars, transient)
    t = runner.mon["ts"].view() |> convert2(Vector)
    dt = brainpy.share["dt"] |> convert2(Float64)
    lastt = last(t) |> convert2(Float64)
    @assert dt ≈ mean(diff(t))
    t = range(start = first(t), step = dt, length = length(t))
    @assert last(t) ≈ lastt
    t = t .* u"ms" # Add time units
    t = t .+ step(t) |> 𝑡 # Correct for python starting at 0
    ps = Iterators.product(populations, vars)
    monitors = popvars2monitors(populations, vars)
    _X = [runner.mon[m].view() |> convert2(PyArray) for m in monitors]
    X = map(_X, ps) do x, (p, v)
        vs = Symbol.(["$p"] .* string.(1:size(x, 2)))
        Timeseries(x, t, Neuron(vs))[𝑡(OpenInterval(transient, Inf * u"s"))]
    end
    X = reshape(X, length(populations), length(vars))
    X = ToolsArray(X, (Population(populations), Var(vars)))
end

function popvars2monitors(populations, vars)
    ps = collect(Iterators.product(populations, vars))[:]
    monitors = ["$p.$v" for (p, v) in ps] |> Tuple
    return monitors
end
function monitors2popvars(monitors)
    ps = split.(monitors, ".")
    ps = [(Symbol(p[1]), Symbol(p[2])) for p in ps]
    populations = first.(ps) |> unique |> collect
    vars = last.(ps) |> unique |> collect
    return return populations, vars
end
function bpsolve(net::Py, time::Real; populations = [:E, :I], vars = [:V],
                 transient = 0u"ms",
                 inputs = nothing,
                 kwargs...)
    monitors = popvars2monitors(populations, vars)
    if !isnothing(inputs)
        inputs = models.format_input(inputs)
    end
    runner = bprun(net, time; monitors, inputs, kwargs...)
    return bpformat(runner; populations, vars, transient)
end

function bpformat(res, param::Symbol, vals; dt = brainpy.share["dt"], transient, monitors)
    # ? res has first dim == 'monitors'. Each element has shape (param, ts, neurons)
    populations, vars = monitors2popvars(monitors)
    nt = res[0][0].shape[0] |> convert2(Int)
    t = range(start = dt, step = dt, length = nt) .* u"ms"

    _X = [m |> convert2(PyArray) for m in res]
    X = map(_X, monitors) do x, m
        p = first(split(m, "."))
        vs = Symbol.(["$p"] .* string.(1:size(x)[end]))
        x = ToolsArray(x, (Dim{param}(vals), 𝑡(t), Neuron(vs)))
        return x[𝑡(OpenInterval(transient,
                                Inf * u"s"))]
    end
    X = map(Iterators.product(populations, vars)) do (p, v) # Get the monitors in the right order
        return X[monitors .== ["$p.$v"]] |> only
    end
    X = reshape(X, length(populations), length(vars))
    return ToolsArray(X, (Population(populations), Var(vars))) |> stack
end

"""
Format an arbitrary batch computation, e.g. monitor output of stats_run
"""
function batchformat(batch_res, sweep_parameters, ::Val{:monitor}; metadata,
                     delete_key = true)
    transient = metadata[:transient]
    tmax = metadata[:tmax]
    dt = metadata[:dt]
    monitors = keys(batch_res) |> collect#.keys() |> convert2(Vector{String})
    sweep_parameters = deepcopy(sweep_parameters)
    if haskey(sweep_parameters, "key") && delete_key
        sweep_parameters = delete!(sweep_parameters, "key")
    end
    vs = values(sweep_parameters)
    vs = map(vs) do v
        if ndims(v) == 2
            return eachrow(v)
        else
            return v
        end
    end
    sweep_parameters = map((vs...,) -> NamedTuple(Symbol.(keys(sweep_parameters)) .=> vs),
                           vs...)
    res = map(monitors) do m
        population, var = split(m, ".")
        res = batch_res[m] |> PyArray
        ts = range(transient, tmax, step = dt)[1:size(res, 2)]
        neurons = Symbol.(population .* string.(1:size(res, 3)))
        res = ToolsArray(res, (Obs(sweep_parameters), 𝑡(ts), Neuron(neurons)))
        res = map(eachslice(res, dims = Obs)) do r
            r = permutedims(r, (𝑡, Neuron))
            if r isa SpikeTrain
                r = set(r, sparse(r))
            end
            return r
        end
        return res
    end
    res = map(sweep_parameters) do p
        out = map(res) do r
            r[Obs = At(p)]
        end
        NamedTuple{Tuple(Symbol.(monitors))}(out)
    end
    res = Dict(sweep_parameters .=> res)
end
function batchformat(batch_res, sweep_parameters, ::Val{:mua}; metadata, delete_key = true)
    transient = metadata[:transient]
    tmax = metadata[:tmax]
    dt = metadata[:mua_dt]
    monitors = keys(batch_res) |> collect#.keys() |> convert2(Vector{String})
    sweep_parameters = deepcopy(sweep_parameters)
    if haskey(sweep_parameters, "key") && delete_key
        sweep_parameters = delete!(sweep_parameters, "key")
    end
    vs = values(sweep_parameters)
    vs = map(vs) do v
        if ndims(v) == 2
            return eachrow(v)
        else
            return v
        end
    end
    sweep_parameters = map((vs...,) -> NamedTuple(Symbol.(keys(sweep_parameters)) .=> vs),
                           vs...)
    res = map(monitors) do m
        res = batch_res[m] |> PyArray
        ts = range(transient, tmax, step = dt)[1:size(res, 2)]
        return ToolsArray(res, (Obs(sweep_parameters), 𝑡(ts)))
    end
    res = map(sweep_parameters) do p
        out = map(res) do r
            r[Obs = At(p)]
        end
        NamedTuple{Tuple(Symbol.(monitors))}(out)
    end
    res = Dict(sweep_parameters .=> res)
end
function batchformat(r, sweep_parameters, param::Symbol; kwargs...)
    return batchformat(r, sweep_parameters, Val(param); kwargs...)
end
function batchformat(stats, sweep_parameters; metadata, delete_key = true)
    stats = pyconvert(Dict, stats)
    sweep_parameters = pyconvert(Dict, sweep_parameters)
    map(collect(stats)) do (stat, res)
        d = pyconvert(Dict, res)
        d = batchformat(d, sweep_parameters, Symbol(stat); metadata, delete_key)
        return Symbol(stat) => d
    end |> NamedTuple
end

@generated sortparams(nt::NamedTuple{KS}) where {KS} = :(NamedTuple{$(Tuple(sort(collect(KS))))}(nt))

# function _bpsweep(model, param::Val{:delta}, vals;
#                   duration,
#                   transient,
#                   populations = [:E, :I],
#                   vars = [:V],
#                   num_parallel = 10)
#     monitors = popvars2monitors(populations, vars)
#     duration = uconvert(u"ms", duration) |> ustrip
#     dt = brainpy.share["dt"] |> convert2(Float64)
#     res = model.sweep_deltas(jax.numpy.array(vals);
#                              duration,
#                              monitors,
#                              num_parallel)
#     param = typeof(param).parameters |> only
#     X = bpformat(res, param, vals; dt, transient, monitors)
#     return X
# end
# function bpsweep(model, param::Symbol, vals; kwargs...)
#     _bpsweep(model, Val(param), vals; kwargs...)
# end
# function bpsweep(model, param::Pair; kwargs...)
#     _bpsweep(model, Val(first(param)), last(param); kwargs...)
# end

# function bpsweep(model_class, conn::Py, params::Dict, param::Pair; batch_size,
#                  batch_seed = 42,
#                  kwargs...)
#     vals = last(param)
#     param = first(param)
#     # * Subset the vals into batches of size batch
#     batch_vals = Iterators.partition(vals, batch_size)
#     X = map(batch_vals) do bvals
#         clear_live_arrays()
#         model = model_class(; params..., copy_conn = conn,
#                             key = jax.random.PRNGKey(batch_seed)) # Fast construction
#         _bpsweep(model, Val(param), bvals; kwargs...)
#     end
#     pcat(x, y) = cat(x, y, dims = param)
#     return reduce(pcat, X)
# end

# """ Only for jittable funcs. Each value fo params should be an interator of the same length"""
# function bpsweep(func, params::Dict; batch_size = 5, clear_buffer, kwargs...)
#     return stats.sweep_progress(func, params; batch_size, clear_buffer, kwargs...)
# end

function PRNGKey(seed::Integer)
    jax.random.PRNGKey(seed)
end

function create_run(model;
                    params = (),
                    monitors,
                    tmax,
                    transient = 0.0,
                    concrete_out = false)
    tmax = ustrip(to_ms(tmax))
    transient = ustrip(to_ms(transient))
    return stats.create_run(model, pydict(params), monitors, tmax, transient, concrete_out)
end

function create_stats_run(run, stat_funcs)
    return stats.create_stats_run(run, pydict(stat_funcs))
end

function string_keys(d::Dict{Symbol, T}) where {T}
    return Dict(string(k) => v for (k, v) in d)
end
function string_keys(d::Py)
    return d
end
function string_keys(d::Dict{String, T}) where {T}
    return d
end
function string_keys(d::NamedTuple)
    return d |> pairs |> Dict |> string_keys
end

function partial_vmap(func; batch_size = nothing, kwargs...)
    function _map(x; batch_size = batch_size)
        if isnothing(batch_size)
            batch_size = maximum(length.(values(x)))
        end
        d = pydict(string_keys(x))
        stats.partial_vmap(func; batch_size, kwargs...)(d)
    end
end

function model_parameters(m::Py; params...)
    ps = m.to_dict(m) |> convert2(Dict{String, Any})
    ps = (; ps..., params...)
    return ps
end
