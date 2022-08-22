import jax.numpy as jnp
import jax
from dm_control import suite
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
from dm_control.suite import acrobot
from dm_control.suite import common
from dm_control.rl import control
from dm_control.utils import io as resources

from physics import *

class DoublePendulumDataset:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
    
    def plot_trajectory(self, trajectory):
        frames = []
        L = self.l1 + self.l2

        history_x, history_y = [], []
        for state in trajectory:
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
            ax.set_aspect('equal')
            ax.grid()

            theta1, theta2, _, _ = state
            x1, y1 = self.l1 * np.sin(theta1), -self.l1 * np.cos(theta1)
            x2, y2 = self.l2 * np.sin(theta2) + x1, -self.l2 * np.cos(theta2) + y1

            history_x.append(x2)
            history_y.append(y2)

            ax.plot([0, x1, x2], [0, y1, y2], 'o-', lw=2, color='blue')
            ax.plot(history_x, history_y, '.-', lw=1, color='orange')

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(buf)
            plt.close(fig)
        return frames


class AnalyticalDoublePendulumDataset(DoublePendulumDataset):
    def __init__(self, m1=1, m2=1, l1=1, l2=1, g=9.8):
        super().__init__(l1, l2)
        self.m1, self.m2, self.g = m1, m2, g
    
    def get_data(self, x0, times): 
        x = jax.device_get(solve_analytical(x0, times)) # dynamics for first N time steps
        xt = jax.device_get(jax.vmap(f_analytical)(x)) # time derivatives of each state
        x = jax.device_put(jax.vmap(normalize_dp)(x))
        return x, xt

    def kinetic(self, q, q_dot):
        t1, t2 = q     # theta 1 and theta 2
        w1, w2 = q_dot # omega 1 and omega 2

        # kinetic energy (T)
        T1 = 0.5 * self.m1 * (self.l1 * w1)**2
        T2 = 0.5 * self.m2 * ((self.l1 * w1)**2 + (self.l2 * w2)**2 +
                        2 * self.l1 * self.l2 * w1 * w2 * jnp.cos(t1 - t2))
        return T1 + T2

    def potential(self, q, q_dot):
        t1, t2 = q     # theta 1 and theta 2
        w1, w2 = q_dot # omega 1 and omega 2
        # potential energy (V)
        y1 = -self.l1 * jnp.cos(t1)
        y2 = y1 - self.l2 * jnp.cos(t2)
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        return V

    def lagrangian(self, q, q_dot):
        T = self.kinetic(q, q_dot)
        V = self.potential(q, q_dot)
        return T - V
    
    def total_energy(self, q, q_dot):
        T = self.kinetic(q, q_dot)
        V = self.potential(q, q_dot)
        return T + V

    def f_analytical(self, state, t=0):
        t1, t2, w1, w2 = state
        a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * jnp.cos(t1 - t2)
        a2 = (self.l1 / self.l2) * jnp.cos(t1 - t2)
        f1 = -(self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (w2**2) * jnp.sin(t1 - t2) - \
            (self.g / self.l1) * jnp.sin(t1)
        f2 = (self.l1 / self.l2) * (w1**2) * jnp.sin(t1 - t2) - (self.g / self.l2) * jnp.sin(t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)
        return jnp.stack([w1, w2, g1, g2])


class MujocoDoublePendulumDataset(DoublePendulumDataset):
    def __init__(self):
        super().__init__(0.5, 0.5)
    
    def get_env(self):
        physics = acrobot.Physics.from_xml_string(resources.GetResource('xmls/acrobot.xml'), common.ASSETS)
        task = acrobot.Balance(sparse=False, random=None)
        return control.Environment(physics, task, time_limit=10)

    def get_data(self, x0, times):
        interval = times[1] - times[0]
        assert abs(interval * 100 - round(interval * 100)) < 1e-5
        steps = round((times[-1] - times[0]) * 100)
        presteps = round(times[0] * 100)

        # env = suite.load('acrobot', 'swingup')
        env = self.get_env()
        state = env.physics.get_state()
        state[:2] = self._x2q(x0)
        env.physics.set_state(state)
        env.physics.forward()

        x = []
        xt = []
        i = 0
        for _ in range(presteps):
            env.physics.step()
        for step in range(steps):
            if step == round((times[i] - times[0]) * 100):
                x.append(np.hstack([self._q2x(env.physics.data.qpos), env.physics.data.qvel]))
                xt.append(np.hstack([env.physics.data.qvel, env.physics.data.qacc]))
                i += 1
            env.physics.step()
        return np.vstack(x), np.vstack(xt)

    def _x2q(self, x):
        q1 = np.pi - x[0]
        q2 = np.pi - q1 - x[1]
        return np.array([q1, q2])
    
    def _q2x(self, q):
        x1 = np.pi - q[0]
        x2 = np.pi - q[0] - q[1]
        return np.array([x1, x2])

if __name__ == '__main__':
    from utils import *
    ds = MujocoDoublePendulumDataset()
    t = np.linspace(0, 100, 1001)
    x0 = np.array([3 / 7 * np.pi, 3 / 4 * np.pi, 0, 0], dtype=np.float32)
    x, xt = ds.get_data(x0, t)
    v = np.mean(x[:, 2:], axis=1)
    plt.plot(t[:v.shape[0]], v)
    plt.show()
    # frames = ds.plot_trajectory(x)
    # generate_gif(frames, dt=10)
    