import jax
import brainpy as bp
import brainpy.math as bm
from brainpy.channels import INa_HH1952, IL
import matplotlib.pyplot as plt

print(bp.__version__)
bm.set_platform("cpu")


class IK(bp.dyn.IonChannel):  # bp.Channel
    def __init__(self, size, E=-77.0, g_max=36.0, phi=1.0, method="rk4"):  # 'exp_auto'
        super(IK, self).__init__(size)
        self.g_max = g_max
        self.E = E
        self.phi = phi

        self.n = bm.Variable(
            bm.zeros(size)
        )  # variables should be packed with bm.Variable

        self.integral = bp.odeint(self.dn, method=method)

    def dn(self, n, t, V):
        alpha_n = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
        beta_n = 0.125 * bm.exp(-(V + 65) / 80)
        return self.phi * (alpha_n * (1.0 - n) - beta_n * n)

    def update(self, tdi, V):
        self.n.value = self.integral(self.n, tdi.t, V, dt=tdi.dt)

    def current(self, V):
        return self.g_max * self.n**4 * (self.E - V)


class HH(bp.dyn.CondNeuGroup):  # bp.CondNeuGroup
    def __init__(self, size):
        super().__init__(size, V_initializer=bp.init.Uniform(-70.0, -70.0))
        self.IK = IK(size, E=-77.0, g_max=36.0)
        self.INa = INa_HH1952(size, E=50.0, g_max=120.0)
        self.IL = IL(size, E=-54.39, g_max=0.03)


def I_inject(shared):
    return jax.numpy.logical_and(0 <= shared["t"], shared["t"] <= 300) * 6.0


neu = HH(size=1)

runner = bp.DSRunner(neu, monitors=["V"], inputs=I_inject)

runner.run(200)  # the running time is 200 ms

plt.figure()
plt.plot(runner.mon["ts"], runner.mon["V"])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()
