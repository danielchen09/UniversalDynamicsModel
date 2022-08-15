import jax.numpy as jnp
import jax
from dm_control import suite

from physics import *

class AnalyticalDoublePendulumDataset:
    def __init__(self, m1=1, m2=1, l1=1, l2=1, g=9.8):
        self.m1, self.m2, self.l2, self.l2, self.g1 = m1, m2, l1, l2, g
    
    def get_data(self, x0, times): 
        x = jax.device_get(solve_analytical(x0, times)) # dynamics for first N time steps
        xt = jax.device_get(jax.vmap(f_analytical)(x)) # time derivatives of each state
        x = jax.device_put(jax.vmap(normalize_dp)(x))
        return x, xt

    def lagrangian(self, q, q_dot):
        t1, t2 = q     # theta 1 and theta 2
        w1, w2 = q_dot # omega 1 and omega 2

        # kinetic energy (T)
        T1 = 0.5 * self.m1 * (self.l1 * w1)**2
        T2 = 0.5 * self.m2 * ((self.l1 * w1)**2 + (self.l2 * w2)**2 +
                        2 * self.l1 * self.l2 * w1 * w2 * jnp.cos(t1 - t2))
        T = T1 + T2

        # potential energy (V)
        y1 = -self.l1 * jnp.cos(t1)
        y2 = y1 - self.l2 * jnp.cos(t2)
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2

        return T - V

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
