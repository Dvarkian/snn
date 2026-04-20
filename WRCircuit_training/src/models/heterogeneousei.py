import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import networkx as nx
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import jit, vmap
import uuid
from brainpy.dyn.neurons import GradNeuDyn
from brainpy.dyn import NeuDyn
from brainpy.initialize import (
    ZeroInit,
    OneInit,
    Uniform,
    variable_,
    noise as init_noise,
)
from brainpy import share
from brainpy.types import Shape, ArrayType
from brainpy.check import is_initializer
from brainpy import odeint, sdeint, JointEq
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial

from ..neurons import *


class EmbeddedLif(bp.dyn.LifRef):
    def __init__(
        self,
        *args,
        embedding: Union[None, AbstractPositions] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        maybe_default_embedding(self, embedding)

    def to_dict(
        self,
        keys=[
            "V_rest",
            "V_reset",
            "V_th",
            "R",
            "tau",
            "V_initializer",
            "tau_ref",
            "ref_var",
        ],
    ):
        out = {
            key: maybe_initializer(value)
            for key, value in self.__dict__.items()
            if key in keys
        }
        out["embedding"] = {self.embedding.__class__.__name__: self.embedding.to_dict()}
        return out


class HeterogenousCircuit(bp.Network):
    def __init__(self, num_exc, num_inh, method="exp_auto"):
        super().__init__()

        # geometry
        domain = (2 * np.pi,)
        exc_positions = GridPositions(domain)
        inh_positions = RandomPositions(domain)

        epsilon = 0.1  # Connection probability
        g = 4.0  # ratio exc weight/inh weight
        alpha = 2.0  # Order
        J = 0.1  # Amplitude of exc psp
        J_inh = -g * J  # Amplitude of inh psp
        CE = int(epsilon * num_exc)  # number of excitatory synapses per neuron
        CI = int(epsilon * num_inh)  # number of inhibitory synapses per neuron

        theta = 20.0  # membrane threshold potential in mV
        tauMem = 20.0  # # time constant of membrane potential in ms
        nu_th = theta / (J * CE * tauMem)
        eta = 1.0 * 0.3  # external rate relative to threshold rate
        nu_ex = eta * nu_th
        p_rate = 1000.0 * nu_ex * CE

        # excitatory neurons
        self.E = EmbeddedLif(
            num_exc,
            V_rest=0.0,
            V_reset=10.0,
            V_th=theta,
            R=1.0,
            tau=tauMem,
            V_initializer=bp.init.Constant(10.0),
            tau_ref=2.0,
            ref_var=True,
            method=method,
            embedding=exc_positions,
        )

        # inhibitory neurons: same as excitatory
        self.I = EmbeddedLif(
            num_inh,
            V_rest=0.0,
            V_reset=10.0,
            V_th=theta,
            R=1.0,
            tau=tauMem,
            V_initializer=bp.init.Constant(10.0),
            tau_ref=2.0,
            ref_var=True,
            method=method,
            embedding=inh_positions,
        )

        delay_step = int(2.0 // bp.share["dt"])

        self.ext = bp.neurons.PoissonGroup(num_exc, freqs=p_rate / CE)
        conn_ext2E = bp.connect.FixedPostNum(CE)
        self.ext2E = bp.synapses.Delta(
            self.ext, self.E, conn_ext2E, delay_step=delay_step, g_max=J
        )

        conn_E2E = bp.connect.FixedPreNum(CE)
        conn_E2I = bp.connect.FixedPreNum(CE)
        conn_I2E = bp.connect.FixedPreNum(CI)
        conn_I2I = bp.connect.FixedPreNum(CI)

        # Synapses
        # ! We have no coupling weights yet; make an initializer for the pareto distribution
        self.E2E = bp.synapses.Delta(
            self.E, self.E, conn_E2E, delay_step=delay_step, g_max=J
        )
        self.E2I = bp.synapses.Delta(
            self.E, self.I, conn_E2I, delay_step=delay_step, g_max=J
        )
        self.I2E = bp.synapses.Delta(
            self.I, self.E, conn_I2E, delay_step=delay_step, g_max=J_inh
        )
        self.I2I = bp.synapses.Delta(
            self.I, self.I, conn_I2I, delay_step=delay_step, g_max=J_inh
        )

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)
