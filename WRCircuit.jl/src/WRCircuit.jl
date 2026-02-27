module WRCircuit
# Set JAX determinism flags BEFORE importing any JAX/BrainPy code
# ENV["XLA_FLAGS"] = "XLA_FLAGS=--xla_gpu_deterministic_ops=true"
using DrWatson
# using CUDA
# using cuDNN
using Libdl
using PythonCall
using CondaPkg
import PythonCall: pycopy!

export convert2, jax_device

begin # * Python imports
    const sys = PythonCall.pynew()
    const os = PythonCall.pynew()
    const brainpy = PythonCall.pynew()
    const neurons = PythonCall.pynew()
    const positions = PythonCall.pynew()
    const synapses = PythonCall.pynew()
    const models = PythonCall.pynew()
    const running = PythonCall.pynew()
    const utils = PythonCall.pynew()
    const stats = PythonCall.pynew()
    const distances = PythonCall.pynew()
    const jax = PythonCall.pynew()
    const jax_lib = PythonCall.pynew()
    const xla_bridge = PythonCall.pynew()
    const numpy = PythonCall.pynew()
    const gc = PythonCall.pynew()
end

function __init__()
    if CondaPkg.backend() === :MicroMamba
        libpath = joinpath(projectdir(".CondaPkg/env/lib/"))
    elseif CondaPkg.backend() === :Pixi
        libpath = joinpath(projectdir(".CondaPkg/.pixi/envs/default/lib/"))
    end
    if !isdir(libpath)
        throw(error("Could not find the environment library path at $libpath. Make sure you have set up CondaPkg.jl with the Pixi or MicroMamba backend."))
    else
        push!(Base.DL_LOAD_PATH, libpath)
    end
    if isfile(joinpath(libpath, "libcudnn.so"))
        dlopen("libcudnn")
    else
        throw(error("Could not find libcudnn.so in $libpath. Make sure you have installed cudnn in the CondaPkg.jl environment."))
    end

    pycopy!(sys, pyimport("sys"))
    sys.path.insert(0, dirname(@__DIR__))

    pycopy!(os, pyimport("os"))
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # For clusters/multiprocessing

    pycopy!(jax, pyimport("jax"))
    pycopy!(jax_lib, pyimport("jax.lib"))
    pycopy!(xla_bridge, pyimport("jax.lib.xla_bridge"))
    pycopy!(brainpy, pyimport("brainpy"))
    pycopy!(neurons, pyimport("src.neurons"))
    pycopy!(positions, pyimport("src.positions"))
    pycopy!(synapses, pyimport("src.synapses"))
    pycopy!(models, pyimport("src.models"))
    pycopy!(running, pyimport("src.running"))
    pycopy!(utils, pyimport("src.utils"))
    pycopy!(stats, pyimport("src.stats"))
    pycopy!(distances, pyimport("src.distances"))
    pycopy!(numpy, pyimport("numpy"))
    pycopy!(gc, pyimport("gc"))

    if haskey(ENV, "WRCircuit_BACKEND")
        backend = ENV["WRCircuit_BACKEND"]
        if backend == "cpu"
            jax.default_device = jax.devices("cpu")[0]
            jax.config.update("jax_platform_name", "cpu")
            brainpy.math.set_platform("cpu")
        elseif backend == "gpu"
            jax.default_device = jax.devices("gpu")[0]
            jax.config.update("jax_platform_name", "gpu")
            brainpy.math.set_platform("gpu")
        else
            @warn "Unknown WRCircuit.jl backend $backend"
        end
    end

    # * Set default dt
    if haskey(ENV, "BRAINPY_DT")
        brainpy.math.set_dt(parse(Float64, ENV["BRAINPY_DT"]))
    else
        brainpy.math.set_dt(0.1)
    end

    begin # * CUDA checks
        # CUDA.has_cuda() || (@warn "CUDA is not available")
        _jax_backend = xla_bridge.get_backend().platform
        pyconvert(String, _jax_backend) == "gpu" || (@warn "JAX is not using the GPU")
    end
end
_cudnn_version() = jax._src.lib.cuda_versions.cudnn_get_version()
_cudnn_build_version() = jax._src.lib.cuda_versions.cudnn_build_version()
convert2(x::Type) = Base.Fix1(pyconvert, x)
jax_device() = xla_bridge.get_backend().platform |> convert2(String)
jax_live_arrays() = xla_bridge.get_backend().live_arrays()
function clear_live_arrays()
    [x.delete() for x in WRCircuit.jax_live_arrays()]
    PythonCall.GC.gc()
end

include("ModelInterface.jl")
include("Utils.jl")
include("Plots.jl")

end # module
