import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from functools import partial
from jax.experimental.host_callback import call
import diffrax
from diffrax import Euler, diffeqsolve, ODETerm, Dopri5, Heun, PIDController, Tsit5
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import *

def lagrangian(q, q_dot, m1, m2, l1, l2, g):
    t1, t2 = q     # theta 1 and theta 2
    w1, w2 = q_dot # omega 1 and omega 2

    # kinetic energy (T)
    T1 = 0.5 * m1 * (l1 * w1)**2
    T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 +
                    2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2))
    T = T1 + T2

    # potential energy (V)
    y1 = -l1 * jnp.cos(t1)
    y2 = y1 - l2 * jnp.cos(t2)
    V = m1 * g * y1 + m2 * g * y2

    return T - V

def f_analytical(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
    a2 = (l1 / l2) * jnp.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(t1 - t2) - \
        (g / l1) * jnp.sin(t1)
    f2 = (l1 / l2) * (w1**2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return jnp.stack([w1, w2, g1, g2])

def equation_of_motion(lagrangian, tau, state, u, t=None):
    q, q_t = jnp.split(state, 2)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t, u))
            @ (tau(q, q_t, u) + jax.grad(lagrangian, 0)(q, q_t, u)
                - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t, u) @ q_t))
    return jnp.concatenate([q_t, q_tt])

def solve_lagrangian_discrete(lagrangian, tau, initial_state, ts, us, **kwargs):
    q = [initial_state]
    dt = ts[1] - ts[0]

    # for t, u in tqdm(zip(ts, us)):
    #     x0 = q2x(q[-1][None, :])[0]
    #     state = equation_of_motion(lagrangian, tau, x0, u)
    #     x1 = wrap_angle(x0 + state * dt)
    #     x1 = x2q(x1[None, :])[0]
    #     q.append(x1)
    for t, u in tqdm(zip(ts, us)):
        state = equation_of_motion(lagrangian, tau, q[-1], u)
        x1 = wrap_angle(q[-1] + state * dt)
        q.append(x1)

    return jnp.vstack(q)

def solve_lagrangian_diffrax(lagrangian, tau, initial_state, ts, u, **kwargs):
    def eom_fn(t, x, args):
            # call(lambda z: print(z), args)
            return equation_of_motion(lagrangian, tau, x, args[jnp.floor(t // (ts[1] - ts[0])).astype(int)], t=t)
    # @partial(jax.jit, backend='cpu')
    def f(initial_state):
        solution = diffeqsolve(
            diffrax.ODETerm(eom_fn),
            Tsit5(), 
            ts[0], 
            ts[-1], 
            None, 
            initial_state,
            u,
            stepsize_controller=PIDController(**kwargs),
            saveat=diffrax.SaveAt(ts=ts))
        return solution.ys
    return f(initial_state)

def solve_lagrangian(lagrangian, tau, initial_state, t, u, **kwargs):
    # We currently run odeint on CPUs only, because its cost is dominated by
    # control flow, which is slow on GPUs.
    def eom_fn(x, t, u):
        # call(lambda z: print(z), t)
        return equation_of_motion(lagrangian, tau, x, u, t=t)
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(eom_fn, initial_state, t, u, **kwargs)
    return f(initial_state)

@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, m1=1, m2=1, l1=1, l2=1, g=9.8):
    L = partial(lagrangian, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    return solve_lagrangian(L, initial_state, t=times, rtol=1e-10, atol=1e-10)

# Double pendulum dynamics via analytical forces taken from Diego's blog
@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
    return odeint(f_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)

def normalize_dp(state):
    # wrap generalized coordinates to [-pi, pi]
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])

def rk4_step(f, x, t, h):
    # one step of runge-kutta integration
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)