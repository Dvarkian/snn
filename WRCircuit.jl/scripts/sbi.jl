#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#

# =============================================================================
# Educational Script: Simulation-Based Inference with SimulationBasedInference.jl
# =============================================================================
#
# This script demonstrates how to use Simulation-Based Inference (SBI) to infer
# parameters of a simulator model. SBI is particularly useful when:
#   1. The likelihood function is intractable or expensive to compute
#   2. You have a "black-box" simulator that can generate synthetic data
#   3. Traditional MCMC methods are too slow or don't converge well
#
# We'll use a simple example here: inferring the decay rate of an exponential
# process. This can later be extended to infer parameters of the neural network
# models in this project (e.g., synaptic strengths K_ee, K_ei, firing rates, etc.)
#
# =============================================================================

using DrWatson
DrWatson.@quickactivate

using SimulationBasedInference
using Distributions
using Random
using Statistics
using CairoMakie
using Foresight
import CairoMakie: Axis

set_theme!(foresight(:physics))

# Set random seed for reproducibility
const rng = Random.MersenneTwister(42)

# =============================================================================
# PART 1: Define the Forward Model (Simulator)
# =============================================================================
#
# In SBI, the "forward model" or "simulator" is a function that takes parameters
# and produces synthetic data. Here we use a simple exponential decay:
#
#   y(t) = y₀ * exp(-α * t)
#
# where α is the decay rate we want to infer.
#
# In the context of neural networks (like the Nonspatial model in this project),
# this would be replaced with a function that:
#   - Takes network parameters (K_ee, K_ei, nu, etc.)
#   - Runs the BrainPy simulation
#   - Returns summary statistics (firing rates, CV, spectral properties)

println("="^70)
println("PART 1: Defining the Forward Model")
println("="^70)

# True parameter value (unknown to the inference algorithm)
α_true = 0.3
y0 = 1.0

# Time points for observations
t_obs = 0.0:0.5:10.0
n_obs = length(t_obs)

# The simulator function
function exponential_simulator(params)
    α = params[1]
    return y0 .* exp.(-α .* t_obs)
end

# Generate "observed" data with noise
σ_true = 0.05  # True noise level
y_true = exponential_simulator([α_true])
y_observed = y_true .+ σ_true .* randn(rng, n_obs)

println("True decay rate: α = $α_true")
println("Number of observations: $n_obs")
println("Noise level: σ = $σ_true")

# =============================================================================
# PART 2: Visualize the Forward Model
# =============================================================================

println("\n" * "="^70)
println("PART 2: Visualizing the Forward Model and Observations")
println("="^70)

fig1 = Figure(size = (800, 400))
ax1 = Axis(fig1[1, 1],
           xlabel = "Time",
           ylabel = "y(t)",
           title = "Exponential Decay: True Model vs Noisy Observations")

# Plot true model
lines!(ax1, collect(t_obs), y_true, label = "True model (α=$α_true)", linewidth = 2)

# Plot observations
scatter!(ax1, collect(t_obs), y_observed, label = "Noisy observations", markersize = 8)

# Plot a few samples from the prior (to show parameter uncertainty)
α_samples = rand(rng, Uniform(0.1, 0.8), 5)
for (i, α) in enumerate(α_samples)
    y_sample = exponential_simulator([α])
    lines!(ax1, collect(t_obs), y_sample, color = (:gray, 0.3),
           label = (i == 1 ? "Prior samples" : nothing))
end

axislegend(ax1, position = :rt)
display(fig1)

# Save figure
save(plotsdir("sbi_forward_model.png"), fig1)
println("Saved: $(plotsdir("sbi_forward_model.png"))")

# =============================================================================
# PART 3: Define Priors
# =============================================================================
#
# The prior encodes our beliefs about the parameters BEFORE seeing the data.
# For SBI to work well, the prior should:
#   1. Cover the plausible range of parameter values
#   2. Not be too broad (makes inference harder)
#   3. Be proper (integrate to 1)
#
# SimulationBasedInference.jl uses the `prior()` function to create named priors.

println("\n" * "="^70)
println("PART 3: Defining Priors")
println("="^70)

# Prior for the decay rate α
# We use a Uniform distribution because we have weak prior information
model_prior = prior(α = Uniform(0.05, 1.0))

# Prior for the noise standard deviation σ
# Exponential prior encodes belief that noise is small but positive
noise_prior = prior(σ = Exponential(0.1))

println("Model prior: α ~ Uniform(0.05, 1.0)")
println("Noise prior: σ ~ Exponential(0.1)")

# Visualize the priors
fig2 = Figure(size = (800, 300))

ax2a = Axis(fig2[1, 1], xlabel = "α", ylabel = "Density", title = "Prior on α")
α_range = 0.0:0.01:1.2
lines!(ax2a, α_range, pdf.(Uniform(0.05, 1.0), α_range), linewidth = 2)
vlines!(ax2a, [α_true], color = :red, linestyle = :dash, label = "True value")
axislegend(ax2a)

ax2b = Axis(fig2[1, 2], xlabel = "σ", ylabel = "Density", title = "Prior on σ")
σ_range = 0.0:0.005:0.5
lines!(ax2b, σ_range, pdf.(Exponential(0.1), σ_range), linewidth = 2)
vlines!(ax2b, [σ_true], color = :red, linestyle = :dash, label = "True value")
axislegend(ax2b)

display(fig2)
save(plotsdir("sbi_priors.png"), fig2)
println("Saved: $(plotsdir("sbi_priors.png"))")

# =============================================================================
# PART 4: Manual SBI via Rejection Sampling (ABC)
# =============================================================================
#
# Before using the SimulationBasedInference.jl package, let's understand SBI
# conceptually by implementing simple Approximate Bayesian Computation (ABC).
#
# ABC Algorithm:
#   1. Sample parameters θ from the prior
#   2. Simulate data x using the forward model with θ
#   3. Compare simulated data to observed data using a distance metric
#   4. Accept θ if distance < threshold ε
#   5. The accepted samples approximate the posterior
#
# This is the simplest form of SBI and helps build intuition.

println("\n" * "="^70)
println("PART 4: Manual ABC (Approximate Bayesian Computation)")
println("="^70)

function simple_abc(observed_data, simulator, prior_dist;
                    n_samples = 10000, epsilon = 0.5, rng = Random.GLOBAL_RNG)
    accepted_samples = Float64[]
    accepted_distances = Float64[]
    all_distances = Float64[]

    for i in 1:n_samples
        # 1. Sample from prior
        α_sample = rand(rng, prior_dist)

        # 2. Simulate
        simulated = simulator([α_sample])

        # 3. Compute distance (sum of squared differences)
        distance = sqrt(mean((simulated .- observed_data) .^ 2))
        push!(all_distances, distance)

        # 4. Accept/reject
        if distance < epsilon
            push!(accepted_samples, α_sample)
            push!(accepted_distances, distance)
        end
    end

    acceptance_rate = length(accepted_samples) / n_samples
    return accepted_samples, accepted_distances, all_distances, acceptance_rate
end

# Run ABC
println("Running ABC with 10,000 samples...")
abc_samples, abc_distances, all_distances, acceptance_rate = simple_abc(y_observed,
                                                                        exponential_simulator,
                                                                        Uniform(0.05, 1.0),
                                                                        n_samples = 10000,
                                                                        epsilon = 0.15,
                                                                        rng = rng)

println("Acceptance rate: $(round(acceptance_rate * 100, digits=2))%")
println("Number of accepted samples: $(length(abc_samples))")
println("ABC posterior mean: $(round(mean(abc_samples), digits=3))")
println("ABC posterior std: $(round(std(abc_samples), digits=3))")
println("True value: $α_true")

# Visualize ABC results
fig3 = Figure(size = (1000, 400))

ax3a = Axis(fig3[1, 1], xlabel = "Distance", ylabel = "Count",
            title = "Distance Distribution (ABC)")
hist!(ax3a, all_distances, bins = 50)
vlines!(ax3a, [0.15], color = :red, linestyle = :dash, label = "Threshold ε=0.15")
axislegend(ax3a)

ax3b = Axis(fig3[1, 2], xlabel = "α", ylabel = "Density",
            title = "ABC Posterior vs Prior")
hist!(ax3b, abc_samples, bins = 30, normalization = :pdf, label = "ABC posterior")
lines!(ax3b, α_range, pdf.(Uniform(0.05, 1.0), α_range),
       color = :gray, linestyle = :dash, label = "Prior")
vlines!(ax3b, [α_true], color = :red, linewidth = 2, label = "True value")
vlines!(ax3b, [mean(abc_samples)], color = :blue, linewidth = 2, label = "Posterior mean")
axislegend(ax3b, position = :rt)

display(fig3)
save(plotsdir("sbi_abc_results.png"), fig3)
println("Saved: $(plotsdir("sbi_abc_results.png"))")

# =============================================================================
# PART 5: Using SimulationBasedInference.jl
# =============================================================================
#
# Now let's use the SimulationBasedInference.jl package for more sophisticated
# inference methods. The package provides:
#
#   - SimulatorForwardProblem: Wraps your simulator
#   - SimulatorInferenceProblem: Combines forward model, prior, and likelihood
#   - Various solvers: EnIS (Importance Sampling), EKS (Ensemble Kalman), etc.
#
# The workflow is:
#   1. Define a SimulatorForwardProblem with your simulator and observables
#   2. Define priors for parameters
#   3. Define a likelihood function
#   4. Create a SimulatorInferenceProblem
#   5. Solve using your preferred algorithm

println("\n" * "="^70)
println("PART 5: Using SimulationBasedInference.jl")
println("="^70)

# Define the simulator as a function that SimulationBasedInference expects
# The function takes a parameter vector and returns a matrix of predictions
# The output shape must match what we declare in the SimulatorObservable
function sbi_simulator(params)
    α = params[1]  # Extract first (and only) parameter from vector
    # Return as a column vector (n_obs x 1 matrix) to match observable shape
    return reshape(y0 .* exp.(-α .* collect(t_obs)), :, 1)
end

# Define the initial parameter vector (used for shape inference)
p0 = [α_true]

# Create a SimulatorObservable
# - :y is the name of the observable
# - state -> state.u extracts the output from the forward solve state
# - (n_obs, 1) specifies the shape of each observation
observable = SimulatorObservable(:y, state -> state.u, (n_obs, 1))

# Create a SimulatorForwardProblem from the function, initial parameters, and observable
forward_prob = SimulatorForwardProblem(sbi_simulator, p0, observable)

println("Forward problem created successfully")

# Define the likelihood
# IsotropicGaussianLikelihood takes:
#   - The observable (links to forward problem output)
#   - The observed data (must match observable shape)
#   - A prior on the noise scale
# Note: noise_scale_prior should be a Distribution, not a prior() object
lik = IsotropicGaussianLikelihood(observable, reshape(y_observed, :, 1), Exponential(0.1))

# Create the inference problem
# Note: SimulatorInferenceProblem takes forward_prob, prior, and likelihood
# No solver is passed here - that's specified in solve()
inference_prob = SimulatorInferenceProblem(forward_prob, model_prior, lik)

println("Inference problem created successfully")

# =============================================================================
# PART 6: Solve with Ensemble Importance Sampling (EnIS)
# =============================================================================
#
# EnIS is one of the simplest SBI methods:
#   1. Sample an ensemble of parameters from the prior
#   2. Run the simulator for each parameter set
#   3. Compute importance weights based on how well each simulation matches data
#   4. The weighted ensemble approximates the posterior

println("\n" * "="^70)
println("PART 6: Ensemble Importance Sampling (EnIS)")
println("="^70)

println("Running EnIS with 1000 ensemble members...")
enis_sol = solve(inference_prob, EnIS(), ensemble_size = 1000, rng = rng)

# Extract results
# get_weights returns importance weights for each ensemble member
enis_weights = get_weights(enis_sol)
# get_transformed_ensemble returns the parameter ensemble (n_params x n_ensemble)
enis_ensemble = get_transformed_ensemble(enis_sol)
enis_α = enis_ensemble[1, :]  # First (and only) parameter

# Compute weighted statistics manually (most reliable across versions)
enis_mean = sum(enis_α .* enis_weights) / sum(enis_weights)
enis_var = sum(enis_weights .* (enis_α .- enis_mean) .^ 2) / sum(enis_weights)
enis_std = sqrt(enis_var)

println("EnIS posterior mean: $(round(enis_mean, digits=3))")
println("EnIS posterior std: $(round(enis_std, digits=3))")
println("True value: $α_true")

# =============================================================================
# PART 7: Compare Results
# =============================================================================

println("\n" * "="^70)
println("PART 7: Comparing Inference Methods")
println("="^70)

fig4 = Figure(size = (1200, 400))

# ABC posterior
ax4a = Axis(fig4[1, 1], xlabel = "α", ylabel = "Density", title = "ABC Posterior")
hist!(ax4a, abc_samples, bins = 30, normalization = :pdf, color = (:blue, 0.6))
vlines!(ax4a, [α_true], color = :red, linewidth = 2, label = "True value")
vlines!(ax4a, [mean(abc_samples)], color = :blue, linewidth = 2, linestyle = :dash,
        label = "Mean")
axislegend(ax4a, position = :rt)

# EnIS posterior (weighted histogram)
ax4b = Axis(fig4[1, 2], xlabel = "α", ylabel = "Density", title = "EnIS Posterior")
hist!(ax4b, enis_α, weights = enis_weights, bins = 30, normalization = :pdf,
      color = (:green, 0.6))
vlines!(ax4b, [α_true], color = :red, linewidth = 2, label = "True value")
vlines!(ax4b, [enis_mean], color = :green, linewidth = 2, linestyle = :dash, label = "Mean")
axislegend(ax4b, position = :rt)

# Posterior predictive check
ax4c = Axis(fig4[1, 3], xlabel = "Time", ylabel = "y(t)",
            title = "Posterior Predictive Check")

# Sample from posterior and simulate
n_posterior_samples = 50
posterior_indices = sample(rng, 1:length(enis_α), Weights(enis_weights),
                           n_posterior_samples, replace = true)
for (i, idx) in enumerate(posterior_indices)
    α_post = enis_α[idx]
    y_post = exponential_simulator([α_post])
    lines!(ax4c, collect(t_obs), y_post, color = (:green, 0.1),
           label = (i == 1 ? "Posterior samples" : nothing))
end

# Plot true model and observations
lines!(ax4c, collect(t_obs), y_true, color = :red, linewidth = 2, label = "True model")
scatter!(ax4c, collect(t_obs), y_observed, color = :black, markersize = 8,
         label = "Observations")
axislegend(ax4c, position = :rt)

display(fig4)
save(plotsdir("sbi_comparison.png"), fig4)
println("Saved: $(plotsdir("sbi_comparison.png"))")

# =============================================================================
# PART 8: Summary Statistics
# =============================================================================

println("\n" * "="^70)
println("PART 8: Summary")
println("="^70)

println("""
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE RESULTS SUMMARY                        │
├─────────────────────────────────────────────────────────────────────┤
│  True value:           α = $(round(α_true, digits=4))                              │
├─────────────────────────────────────────────────────────────────────┤
│  ABC:                                                               │
│    - Posterior mean:   $(round(mean(abc_samples), digits=4))                              │
│    - Posterior std:    $(round(std(abc_samples), digits=4))                              │
│    - Acceptance rate:  $(round(acceptance_rate * 100, digits=2))%                              │
├─────────────────────────────────────────────────────────────────────┤
│  EnIS:                                                              │
│    - Posterior mean:   $(round(enis_mean, digits=4))                              │
│    - Posterior std:    $(round(enis_std, digits=4))                              │
│    - Ensemble size:    1000                                         │
└─────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# PART 9: Extension to Neural Network Models (Commentary)
# =============================================================================
#
# To apply SBI to the neural network models in this project:
#
# 1. DEFINE SUMMARY STATISTICS
#    Since raw spike data is high-dimensional, we need summary statistics:
#    - Population firing rates (E and I)
#    - Coefficient of variation (CV) of inter-spike intervals
#    - Spectral peak frequency
#    - Synchrony measures
#
# 2. CREATE THE SIMULATOR FUNCTION
#    ```julia
#    function neural_simulator(params)
#        K_ee, K_ei, nu = params.K_ee, params.K_ei, params.nu
#        m = WRCircuit.models.Nonspatial(; N_e=2000, K_ee, K_ei, nu, ...)
#        x = bpsolve(m, 5u"s"; populations=[:E, :I], vars=[:spike], transient=2u"s")
#
#        # Compute summary statistics
#        spikes_E = x[Population=At(:E), Var=At(:spike)]
#        rate_E = firingrate(spikes_E)
#        cv_E = cv(spikes_E)
#        # ... more statistics
#
#        return [rate_E, cv_E, ...]
#    end
#    ```
#
# 3. DEFINE PRIORS
#    ```julia
#    model_prior = prior(
#        K_ee = Uniform(50, 200),
#        K_ei = Uniform(100, 300),
#        nu = Uniform(5.0, 20.0)
#    )
#    ```
#
# 4. RUN INFERENCE
#    The same workflow applies - just with your neural network simulator
#    and appropriate summary statistics.
#
# =============================================================================

println("\n" * "="^70)
println("Script completed successfully!")
println("="^70)
println("""
Figures saved to: $(plotsdir())
  - sbi_forward_model.png
  - sbi_priors.png
  - sbi_abc_results.png
  - sbi_comparison.png
""")
