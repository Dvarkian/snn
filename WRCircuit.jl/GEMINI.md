# Gemini Code Assistant Context

This file provides context for the Gemini code assistant to understand the WRCircuit.jl project.

## Project Overview

WRCircuit.jl is a computational neuroscience project for simulating spiking neural networks, specifically a Wilson-Cowan circuit. It uses a hybrid approach with Julia as the main language for orchestrating simulations and Python for performance-critical computations.

The project is structured as a Julia package that wraps a Python backend. The Python backend uses the `brainpy` and `jax` libraries for defining and simulating the neural network models. This allows for high-performance simulations on both CPUs and GPUs.

The core of the project is the `WRCircuit.jl` module, which handles the setup of the Python environment and the communication between Julia and Python. The neural network models are defined in Python in the `src/models` directory. The `src/ModelInterface.jl` file provides a Julia API for interacting with the Python models.

## Building and Running

### Setup

The project uses a combination of Julia's `Pkg` and `CondaPkg` to manage dependencies. To set up the project, you need to have both Julia and Conda installed.

1.  **Instantiate the Julia environment:**
    ```julia
    using Pkg
    Pkg.instantiate()
    ```

2.  **Install Python dependencies:**
    The Python dependencies are managed by `CondaPkg.toml`. They should be installed automatically when the project is instantiated. If not, you can install them manually:
    ```julia
    using CondaPkg
    CondaPkg.resolve()
    CondaPkg.add("brainpy")
    CondaPkg.add("jax")
    # ... and so on for all dependencies in CondaPkg.toml
    ```

### Running Simulations

Simulations are run from Julia. The `scripts` directory contains several examples of how to run simulations. For example, to run a simulation of the spatial model, you could use a script like this:

```julia
using WRCircuit
using Unitful

# Define the model parameters
params = Dict(
    :rho => 20000,
    :dx => 0.5,
    :sigma_ee => 0.06,
    # ... other parameters
)

# Create the model
model = WRCircuit.Spatial(; params...)

# Run the simulation
results = WRCircuit.bpsolve(model, 1000u"ms")

# Plot the results
WRCircuit.plot(results)
```

### Testing

The project has a `test` directory with some tests. To run the tests, you can use the `runtests.jl` file:

```julia
include("test/runtests.jl")
```

## Development Conventions

*   **Julia and Python Interoperability:** The project relies heavily on the `PythonCall.jl` package for interoperability between Julia and Python. Julia is used for high-level scripting and data analysis, while Python is used for the core numerical computations.
*   **DimensionalData.jl:** The simulation results are represented using `DimensionalData.jl` arrays, which provides a convenient way to work with multi-dimensional data with named dimensions.
*   **BrainPy:** The neural network models are built using the `brainpy` library. This library provides a flexible and high-performance framework for building and simulating spiking neural networks.
*   **JAX:** `brainpy` uses `JAX` as its backend for numerical computations. This allows the simulations to be run on both CPUs and GPUs.
*   **Git:** The project is version-controlled with Git and hosted on GitHub.
*   **CI/CD:** The project uses GitHub Actions for continuous integration and testing.
