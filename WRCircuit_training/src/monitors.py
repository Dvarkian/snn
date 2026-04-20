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
from jax import lax
import copy
import os
from tqdm import trange, tqdm

import json
import pickle

from brainpy import share
from typing import Union, Callable, Optional, Sequence, Any, Dict, Tuple, List
from brainpy.types import ArrayType
from functools import partial


# def com(net):

#     def _com():


