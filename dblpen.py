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
from dataset import MujocoDataset
import xml.etree.ElementTree as ET

from physics import *
from utils import *

class DoublePendulumDataset:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
    
    def plot_trajectory(self, trajectory):
        frames = []
        L = self.l1 + self.l2

        history_x, history_y = [], []
        for state in trajectory:
            fig = plt.figure(figsize=(320 / 100, 240 / 100))
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
        return np.array(frames)


class AnalyticalDoublePendulumDataset(DoublePendulumDataset):
    def __init__(self, m1=1, m2=1, l1=1, l2=1, g=9.8):
        super().__init__(l1, l2)
        self.m1, self.m2, self.g = m1, m2, g
    
    def get_data(self, x0, times): 
        x = jax.device_get(solve_analytical(x0, times)) # dynamics for first N time steps
        xt = jax.device_get(jax.vmap(f_analytical)(x)) # time derivatives of each state
        x = jax.device_put(jax.vmap(normalize_dp)(x))
        return x, xt, np.zeros_like(x)

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


class MujocoDoublePendulumDataset(MujocoDataset):
    def get_env(self):
        tree = ET.parse('xmls/acrobot.xml')
        root = tree.getroot()
        if self.use_friction:
            for option in root.iter('option'):
                for flag in option.iter('flag'):
                    flag.set('frictionloss', 'enable')
            for joint in root.iter('joint'):
                joint.attrib.pop('frictionloss', None)
                joint.attrib.pop('damping', None)
        physics = acrobot.Physics.from_xml_string(ET.tostring(root, encoding='utf8').decode(), common.ASSETS)
        task = acrobot.Balance(sparse=False, random=None)
        return control.Environment(physics, task, time_limit=10)

    def init_env(self, env, x0):
        state = env.physics.get_state()
        state[:2] = x2q(x0)
        env.physics.set_state(state)
        env.physics.forward()

    def get_x(self, env):
        return np.hstack([q2x(env.physics.data.qpos), env.physics.data.qvel])

    def get_xt(self, env):
        return np.hstack([env.physics.data.qvel, env.physics.data.qacc])

    def get_u(self, action):
        return np.array([0, action[-1]])
    
    def plot_trajectory(self, trajectory):
        env = self.get_env()
        frames = []
        for x in trajectory:
            q = x2q(x)
            state = env.physics.get_state()
            state = np.zeros_like(state)
            state[:] = q[:]
            env.physics.set_state(state)
            env.physics.step()
            frames.append(env.physics.render(camera_id=0))
        return np.array(frames)


class DoublePendulumBodyDataset(MujocoDoublePendulumDataset):
    def get_x(self, env):
        return np.hstack([
            env.physics.data.geom_xpos[2:].reshape(-1), # position (2*3=6)
            mat2euler(env.physics.data.geom_xmat[2:]).reshape(-1), # orientation (2*3=6)
            env.physics.data.sensordata[:12], # linear, angular velocities (2*3*2=12)
        ])

    def get_xt(self, env):
        return env.physics.data.sensordata

    def get_u(self, action):
        return np.array([0, action[-1]])

class MultiplePendulumDataset(MujocoDoublePendulumDataset):
    def __init__(self, trials, use_friction=True, use_action=False):
        self.trials = trials
        self.rng = np.random.RandomState(42)
        super().__init__(use_friction, use_action)
    
    def get_data(self, x0, times):
        xs = []
        xts = []
        us = []
        for _ in range(self.trials):
            x0 = self.rng.uniform(low=-0.5, high=0.5, size=(2,)) * np.pi # [-pi/2, pi/2]
            x, xt, u = super().get_data(x0, times)
            xs.append(x)
            xts.append(xt)
            us.append(u)
        return np.concatenate(xs), np.concatenate(xts), np.concatenate(us)

if __name__ == '__main__':
    from utils import *
    ds = DoublePendulumBodyDataset(use_action=False, use_friction=False)
    t = np.linspace(0, 100, 1001)
    x0 = np.array([3 / 7 * np.pi, 3 / 4 * np.pi, 0, 0], dtype=np.float32)
    x, xt, u = ds.get_data(x0, t)
    print(np.all(x[:, 6:] == xt[:, :6]))
    # plt.plot(t[:x.shape[0]], np.mean(x[:, :6], axis=1), label='x')
    plt.plot(t[:x.shape[0]], np.mean(x[:, 6:], axis=1), label='v')
    plt.plot(t[:x.shape[0]], np.mean(xt[:, 6:], axis=1), label='xt')
    # plt.plot(t[:x.shape[0]], u[:, 1], label='u')
    plt.legend()
    plt.show()
    # frames = ds.plot_trajectory(x)
    # generate_gif(frames, dt=10)
    