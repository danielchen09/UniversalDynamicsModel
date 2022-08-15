import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from functools import partial
import haiku as hk
import optax
import matplotlib.pyplot as plt
import os
from jax.example_libraries import optimizers
from jax.example_libraries import stax
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.7/lib:'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
print(os.environ['LD_LIBRARY_PATH'])

from physics import *
from updater import GradientUpdater

time_step = 0.01
N = 1500
analytical_step = jax.jit(jax.vmap(partial(rk4_step, f_analytical, t=0.0, h=time_step)))

# x0 = np.array([-0.3*np.pi, 0.2*np.pi, 0.35*np.pi, 0.5*np.pi], dtype=np.float32)
x0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
t = np.arange(N, dtype=np.float32) # time steps 0 to N
x_train = jax.device_get(solve_analytical(x0, t)) # dynamics for first N time steps
xt_train = jax.device_get(jax.vmap(f_analytical)(x_train)) # time derivatives of each state
y_train = jax.device_get(analytical_step(x_train)) # analytical next step

t_test = np.arange(N, 2*N, dtype=np.float32) # time steps N to 2N
x_test = jax.device_get(solve_analytical(x0, t_test)) # dynamics for next N time steps
xt_test = jax.device_get(jax.vmap(f_analytical)(x_test)) # time derivatives of each state
y_test = jax.device_get(analytical_step(x_test)) # analytical next step

x_train = jax.device_put(jax.vmap(normalize_dp)(x_train))
y_train = jax.device_put(y_train)

x_test = jax.device_put(jax.vmap(normalize_dp)(x_test))
y_test = jax.device_put(y_test)


lr = 1e-3
grad_clip = 1
epochs = 1000 * 500
seed = 0
lr_decay_step = 25
lr_decay = 0.7
l2reg = 2e-3

def learned_lagrangian(forward_fn, rng, params):
    def lagrangian(q, q_t):
        assert q.shape == (2,)
        state = normalize_dp(jnp.concatenate([q, q_t]))
        return jnp.squeeze(forward_fn(params, rng, x=state), axis=-1)
    return lagrangian

@partial(jax.jit, static_argnums=0)
def loss_fn(forward_fn, params, rng, batch):
    state, targets = batch
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(forward_fn, rng, params)))(state)
    return jnp.mean((preds - targets) ** 2)

def forward_fn(x):
    mlp = hk.nets.MLP(
        [128, 128, 1], 
        activation=jax.nn.softplus, 
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
        b_init=hk.initializers.RandomNormal())
    return mlp(x)
forward_fn = hk.transform(forward_fn)
loss_fn = partial(loss_fn, forward_fn.apply)
scheduler = optax.piecewise_constant_schedule(lr, boundaries_and_scales={i * epochs // lr_decay_step: lr_decay for i in range(lr_decay_step)})
optimizer = optax.adamw(scheduler, weight_decay=l2reg)
updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)
rng = jax.random.PRNGKey(seed)
state = updater.init(rng, x_train)

train_losses = []
test_losses = []
for epoch in range(epochs):
    state, metric = updater.update(state, (x_train, xt_train))
    if metric['step'] % 1000 == 0:
        train_losses.append(metric['loss'])
        test_loss = loss_fn(state['params'], state['rng'], (x_test, xt_test))
        test_losses.append(test_loss)
        print(f'epoch {epoch}: loss={metric["loss"]}, test loss={test_loss}')

plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')
plt.plot(np.arange(len(test_losses)), test_losses, label='test loss')
plt.ylim(0, 300)
plt.legend()
plt.show()

xt_pred = jax.vmap(partial(equation_of_motion, learned_lagrangian(forward_fn.apply, state['rng'], state['params'])))(x_test)

fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
axes[0].scatter(xt_test[:, 2], xt_pred[:, 2], s=6, alpha=0.2)
axes[0].set_title('Predicting $\dot q$')
axes[0].set_xlabel('$\dot q$ actual')
axes[0].set_ylabel('$\dot q$ predicted')
axes[1].scatter(xt_test[:, 3], xt_pred[:, 3], s=6, alpha=0.2)
axes[1].set_title('Predicting $\ddot q$')
axes[1].set_xlabel('$\ddot q$ actual')
axes[1].set_ylabel('$\ddot q$ predicted')
plt.tight_layout()
plt.show()