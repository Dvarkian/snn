from ..neurons import FNSNeuron, LIFNeuron
from ..synapses import Synapse, maybe_initializer, DeltaSynapse
from ..positions import ClusteredPositions, Positions
import numpy as np
import jax.numpy as jnp
import types

import brainpy as bp
from abc import ABC, abstractmethod


def fixedprob_to_dict(self):
    return {"prob": self.prob}


bp.connect.FixedProb.to_dict = fixedprob_to_dict


class AdaptiveHeterogeneous(bp.Network):
    def __init__(self, N, epsilon=0.1, nu=100, g=4, J=0.1):
        super().__init__()

        # * ! Note that nu_thr = theta / (epsilon * Ne * J * tau)

        # ! Reconcile these
        self.epsilon = epsilon  # ? Probability of connection
        self.nu = nu  # ? Input rate
        self.g = g  # ? E:I ratio
        self.J = J  # ? Synaptic strength

        # * Fixed parameters
        self.D = 2.0  # Transmission delay, ms
        self.C = 0.25  # Membrane capacitance, nF
        self.g_L = 0.0167  # Leak conductance, mS
        self.V_L = -70.0  # Leaky current resting potential, mV
        self.V_K = -85.0  # Potassium current resting potential, mV
        self.V_th = -50.0  # Spike threshold, mV
        self.V_rt = -60.0  # Reset potential, mV
        self.tau_ref = 4.0  # Refractory period, ms
        self.tau_K = 80.0  # Adaptation time constant, ms
        self.Delta_g_K = 0.0  # 10.0  # Conductance increase per spike, nS
        self.V_initializer = bp.init.Uniform(-70.0, -50.0)

        num_inh = N // 5
        num_exc = N - num_inh

        # theta = 20  # mV
        # tau = 20  # ms
        # tau_ref = 2.0  # ms
        # self.nu_thr = (
        #     1000 * theta / (epsilon * num_exc * J * tau)
        # )  # Tau is in milliseconds. *1000 to convert to Hz
        # self.nu = self.nu_hat * self.nu_thr

        # geometry
        exc_positions = ClusteredPositions((-1.5, 0), 1)
        inh_positions = ClusteredPositions((1.5, 0), 1)

        # neurons
        self.E = FNSNeuron(
            size=num_exc,
            C=self.C,  # Membrane capacitance, nF
            g_L=self.g_L,  # Leak conductance, nS
            V_L=self.V_L,  # Leaky current resting potential, mV
            V_K=self.V_K,  # Potassium current resting potential, mV
            V_th=self.V_th,  # Spike threshold, mV
            V_rt=self.V_rt,  # Reset potential, mV
            tau_ref=self.tau_ref,  # Refractory period, ms
            tau_K=self.tau_K,  # Adaptation time constant, ms
            Delta_g_K=self.Delta_g_K,  # Conductance increase per spike, nS
            V_initializer=self.V_initializer,
            embedding=exc_positions,
        )

        # Create a population of inhibitory neurons
        self.I = FNSNeuron(
            size=num_inh,
            C=self.C,  # Membrane capacitance, nF
            g_L=self.g_L,  # Leak conductance, nS
            V_L=self.V_L,  # Leaky current resting potential, mV
            V_K=self.V_K,  # Potassium current resting potential, mV
            V_th=self.V_th,  # Spike threshold, mV
            V_rt=self.V_rt,  # Reset potential, mV
            tau_ref=self.tau_ref,  # Refractory period, ms
            tau_K=self.tau_K,  # Adaptation time constant, ms
            Delta_g_K=self.Delta_g_K,  # Conductance increase per spike, nS
            V_initializer=self.V_initializer,
            embedding=inh_positions,
        )

        Je = self.J
        Ji = -g * self.J

        self.E2E = DeltaSynapse(
            pre=self.E,
            post=self.E,
            delay=self.D,
            g_max=Je,
            conn=bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
        )
        self.E2I = DeltaSynapse(
            self.E,
            self.I,
            delay=self.D,
            g_max=Je,
            conn=bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
        )
        self.I2E = DeltaSynapse(
            self.I,
            self.E,
            delay=self.D,
            g_max=Ji,
            conn=bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
        )
        self.I2I = DeltaSynapse(
            self.I,
            self.I,
            delay=self.D,
            g_max=Ji,
            conn=bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
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
            delay=self.D,
            g_max=Je,
            conn=bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
        )
        self.ext2I = DeltaSynapse(
            self.ext,
            self.I,
            delay=self.D,
            g_max=Je,
            conn=bp.connect.FixedProb(prob=epsilon, allow_multi_conn=True),
        )

        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)
