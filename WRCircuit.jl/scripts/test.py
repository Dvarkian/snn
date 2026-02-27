from jax import numpy as jnp, random as jr
from sbijax import NLE
from sbijax.nn import make_maf
from tensorflow_probability.substrates.jax import distributions as tfd


def prior_fn():
    prior = tfd.JointDistributionNamed(
        dict(theta=tfd.Normal(jnp.zeros(2), jnp.ones(2))), batch_ndims=0
    )
    return prior


def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
    y = theta["theta"] + p.sample(seed=seed)
    return y


fns = prior_fn, simulator_fn
model = NLE(fns, make_maf(2))

y_observed = jnp.array([-1.0, 1.0])
data, _ = model.simulate_data(jr.PRNGKey(1))
params, _ = model.fit(jr.PRNGKey(2), data=data)
posterior, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)
