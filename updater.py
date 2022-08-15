import optax
import jax
from functools import partial
import numpy as np


class GradientUpdater:
    def __init__(self, init_fn, loss_fn, optimizer):
        self.init_fn = init_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    @partial(jax.jit, static_argnums=0)
    def init(self, rng, data):
        out_rng, init_rng = jax.random.split(rng)
        params = self.init_fn(init_rng, data)
        opt_state = self.optimizer.init(params)
        return {
            'step': np.array(0),
            'rng': out_rng,
            'opt_state': opt_state,
            'params': params
        }
    
    @partial(jax.jit, static_argnums=0)
    def update(self, state, data):
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self.loss_fn)(params, rng, data)
        updates, opt_state = self.optimizer.update(g, state['opt_state'], params)
        params = optax.apply_updates(params, updates)
        return {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params
        }, {
            'step': state['step'],
            'loss': loss
        }
