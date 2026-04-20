from ..neurons import FNSNeuron, LIFNeuron
from ..synapses import Synapse, maybe_initializer, DeltaSynapse
from ..positions import ClusteredPositions, Positions
import numpy as np
import jax.numpy as jnp
import types
from scipy.stats import levy_stable
from brainpy._src.initialize.base import _InterLayerInitializer
import scipy

import brainpy as bp
import brainpy.math as bm
from abc import ABC, abstractmethod


def fixedprob_to_dict(self):
    return {"prob": self.prob}


bp.connect.FixedProb.to_dict = fixedprob_to_dict


# class LevyAlphaStable(_InterLayerInitializer):
#     """Initialize weights with Levy alpha-stable distribution using SciPy.

#     Parameters
#     ----------
#     alpha : float
#         Stability parameter (0 < alpha <= 2).
#     beta : float, optional
#         Skewness parameter in [-1, 1]. Default is 0 (symmetric).
#     loc : float, optional
#         Location parameter. Default is 0.
#     scale : float, optional
#         Positive scale parameter. Default is 1.
#     seed : int, optional
#         Random seed. Default is None.
#     """

#     def __init__(self, alpha, beta=0.0, loc=0.0, scale=1.0, seed=None):
#         self.alpha = alpha
#         self.beta = beta
#         self.loc = loc
#         self.scale = scale
#         self.seed = seed  # Just store the seed; SciPy will handle RNG

#     def __call__(self, shape, dtype=None):
#         samples = levy_stable.rvs(
#             self.alpha,
#             self.beta,
#             loc=self.loc,
#             scale=self.scale,
#             size=shape,
#             random_state=self.seed,  # SciPy uses this to seed its RNG
#         )
#         return bm.asarray(samples, dtype=dtype)

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}("
#             f"alpha={self.alpha}, beta={self.beta}, "
#             f"loc={self.loc}, scale={self.scale})"
#         )


class ParetoSynaptic(_InterLayerInitializer):
    """Initialize weights according to the translated power-law (Pareto)
    distribution described in the paper snippet.

    Parameters
    ----------
    alpha : float
        The Pareto (power-law) exponent. Must be > 1 for the mean to exist.
    mean_j : float
        The desired mean, <J>.
    A_alpha : float
        Constant factor that appears in Eq. (A15) or (D1).
        (In many papers, this might be 2*A_alpha = some known constant.)
    D_L : float
        Another constant factor from the paper.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    """

    def __init__(self, alpha, mean_j, D_L, seed=None):
        if alpha <= 1:
            raise ValueError("alpha must be > 1 if you want a finite mean.")
        self.alpha = alpha
        self.sign = 1 if mean_j > 0 else -1
        self.mean_j = np.abs(mean_j)
        A_alpha = scipy.special.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / np.pi
        self.D_L = D_L
        self.rng = bm.random.default_rng(seed, clone=False)

        # From Eq. (D3) in the snippet:
        #    x1 = (2 * alpha * D_L * <J>^alpha / alpha)^(1/alpha)
        # or with the constant A_alpha included as in the text:
        #    x1 = ((2 * A_alpha * D_L * <J>^alpha) / alpha)^(1/alpha)
        numerator = 2.0 * A_alpha * self.D_L * (self.mean_j**self.alpha)
        self.x1 = (numerator / self.alpha) ** (1.0 / self.alpha)

        # From Eq. (D4):
        #    x0 = <J> - x1 * alpha / (alpha - 1)
        self.x0 = self.mean_j - self.x1 * self.alpha / (self.alpha - 1)

    def __call__(self, shape, dtype=None):
        # np.random.pareto(alpha) returns samples X >= 1, with PDF = alpha * X^(-alpha-1).
        # So each sample is in [1, ∞).  Multiplying by x1 -> samples in [x1, ∞).
        # Finally, adding x0 -> samples in [x0 + x1, ∞).
        raw_samples = self.rng.pareto(self.alpha, size=shape)
        # Translated power-law variable:
        samples = self.x0 + self.x1 * raw_samples
        samples = samples * self.sign

        return bm.asarray(samples, dtype=dtype)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, mean_j={self.mean_j}, "
            f"x0={self.x0}, x1={self.x1})"
        )


class Heterogeneous(bp.Network):
    def __init__(self, N, epsilon=0.1, D=1.5, nu_hat=2, g=5, J=0.1, alpha=2.0):
        super().__init__()

        # * Note that nu_thr = theta / (epsilon * Ne * J * tau)

        self.epsilon = epsilon
        self.D = D
        self.nu_hat = nu_hat  # Normalized input rate
        self.g = g
        self.J = J
        self.alpha = alpha

        num_inh = N // 5
        num_exc = N - num_inh

        theta = 20  # mV
        tau = 20  # ms
        tau_ref = 2.0  # ms
        self.nu_thr = (
            1000 * theta / (epsilon * num_exc * J * tau)
        )  # Tau is in milliseconds. *1000 to convert to Hz
        self.nu = self.nu_hat * self.nu_thr

        # geometry
        exc_positions = ClusteredPositions((-1.5, 0), 1)
        inh_positions = ClusteredPositions((1.5, 0), 1)

        # neurons
        self.E = LIFNeuron(
            size=num_exc,
            embedding=exc_positions,
            V_rest=0.0,  # For simple IF neuron in paper
            V_th=theta,
            V_reset=10.0,
            R=1,
            tau=tau,
            tau_ref=tau_ref,
            V_initializer=bp.init.Normal(0, 1.0),
        )

        # Create a population of inhibitory neurons
        self.I = LIFNeuron(
            size=num_inh,
            embedding=inh_positions,
            V_rest=0.0,
            V_th=theta,
            V_reset=10.0,
            R=1,
            tau=tau,
            tau_ref=tau_ref,
            V_initializer=bp.init.Normal(0, 1.0),
        )

        # Synapses
        delay_step = D // bp.share["dt"]
        delay_step = int(delay_step)

        JE = ParetoSynaptic(self.alpha, self.J, 1 / 2)  # D_L = 1/2 for alpha = 2.0
        JI = ParetoSynaptic(self.alpha, -g * self.J, 1 / 2)

        output = bp.synouts.CUBA(
            target_var="_RI"
        )  # Variable for tracking synaptic inputs

        self.E2E = DeltaSynapse(  # * This is the slow part. Can we jax it?
            self.E,
            self.E,
            bp.connect.FixedProb(
                prob=epsilon, allow_multi_conn=True
            ),  # allow_multi_conn=True speeds up construction SO MUCH!!! Because it allows for jax
            delay_step=delay_step,
            g_max=JE,
            output=output,
        )
        self.E2I = DeltaSynapse(
            self.E,
            self.I,
            bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
            delay_step=delay_step,
            g_max=JE,
            output=output,
        )
        self.I2E = DeltaSynapse(
            self.I,
            self.E,
            bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
            delay_step=delay_step,
            g_max=JI,
            output=output,
        )
        self.I2I = DeltaSynapse(
            self.I,
            self.I,
            bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
            delay_step=delay_step,
            g_max=JI,
            output=output,
        )

        # External population
        self.ext = bp.dyn.PoissonGroup(
            self.E.num,  # So that the average number of connections to each population matches Ce
            self.nu,
            keep_size=False,
            sharding=None,
            spk_type=None,
            name=None,
            mode=None,
            seed=None,
        )
        self.ext2E = DeltaSynapse(
            self.ext,
            self.E,
            bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
            delay_step=delay_step,
            g_max=JE,
            output=output,
        )
        self.ext2I = DeltaSynapse(
            self.ext,
            self.I,
            bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
            delay_step=delay_step,
            g_max=JE,
            output=output,
        )

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)

    def to_dict(self):
        return {
            self.__class__.__name__: {
                "epsilon": self.epsilon,
                "D": self.D,
                "nu": maybe_initializer(self.nu),
                "g": self.g,
                "J": self.J,
                "populations": {
                    "E": self.E.to_dict(),
                    "I": self.I.to_dict(),
                },
                "synapses": {
                    "E2E": self.E2E.to_dict(),
                    "E2I": self.E2I.to_dict(),
                    "I2E": self.I2E.to_dict(),
                    "I2I": self.I2I.to_dict(),
                },
            }
        }
