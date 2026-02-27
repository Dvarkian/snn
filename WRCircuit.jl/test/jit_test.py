import jax
import brainpy as bp
import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_dir)
import src


def simple_jit_test():
    @jax.jit
    def add(a, b):
        return a + b

    add(1, 2)
    return True


def jit_positions():
    f = src.positions.ClusteredPositions((-1.5, 0), 1)
    f = bp.math.jit(f)
    f((10, 10))
    return True


def jit_LIFNeurons():
    positions = src.positions.ClusteredPositions((-1.5, 0), 1)
    positions = bp.math.jit(positions)
    neuron = bp.math.jit(src.neurons.LIFNeuron.__init__)
    E = neuron(
        size=100,
        embedding=positions,
        V_rest=0.0,  # For simple IF neuron in paper
        V_th=20,
        V_reset=10.0,
        R=1,
        tau=20,
        tau_ref=2,
        V_initializer=bp.init.Normal(0, 1.0),
    )

    def run(E):
        runner = bp.DSRunner(E, monitors=("E.spike",))
        runner.run(1000.0)
        return runner

    run = bp.math.jit(run)
    runner = run(E)
    # print(runner.mon.ts)
    return True


def main():
    simple_jit_test()
    jit_positions()
    # jit_LIFNeurons()
    return True


if __name__ == "__main__":
    main()
