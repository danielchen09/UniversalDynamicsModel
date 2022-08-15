from turtle import forward
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
from dblpen import AnalyticalDoublePendulumDataset

class DoublePendulumExperiment:
    def __init__(self):
        self.lr = 1e-3
        self.grad_clip = 1
        self.epochs = 1000 * 500
        self.seed = 0
        self.lr_decay_step = 25
        self.lr_decay = 0.7
        self.l2reg = 2e-3

        N = 1500
        dataset_obj = AnalyticalDoublePendulumDataset()
        x0 = np.array([3 / 7 * np.pi, 3 / 4 * np.pi, 0, 0], dtype=np.float32)
        self.t_train = np.arange(N, dtype=np.float32)
        self.x_train, self.xt_train = dataset_obj.get_data(x0, self.t_train)
        
        self.t_test = np.arange(N, 2 * N, dtype=np.float32)
        self.x_test, self.xt_test = dataset_obj.get_data(x0, self.t_test)

        def forward_fn(x):
            mlp = hk.nets.MLP(
                [128, 128, 1], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            return mlp(x)
        self.forward_fn = hk.transform(forward_fn)

        self.scheduler = optax.piecewise_constant_schedule(self.lr, boundaries_and_scales={i * self.epochs // self.lr_decay_step: self.lr_decay for i in range(self.lr_decay_step)})
        self.optimizer = optax.adamw(self.scheduler, weight_decay=self.l2reg)
        self.updater = GradientUpdater(self.forward_fn.init, self.loss_fn, self.optimizer)
        self.rng = jax.random.PRNGKey(self.seed)
        self.state = self.updater.init(self.rng, self.x_train)
        self.params = self.state['params']
        self.rng = self.state['rng']
    
    def train_step(self):
        self.state, metric = self.updater.update(self.state, (self.x_train, self.xt_train))
        self.params = self.state['params']
        self.rng = self.state['rng']
        return metric

    def train(self):
        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            metric = self.train_step()
            if metric['step'] % 1000 == 0:
                train_losses.append(metric['loss'])
                test_loss = self.test()
                test_losses.append(test_loss)
                print(f'epoch {epoch}: loss={metric["loss"]}, test loss={test_loss}')

        plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')
        plt.plot(np.arange(len(test_losses)), test_losses, label='test loss')
        plt.ylim(0, 300)
        plt.legend()
        plt.show()
        return train_losses, test_losses

    def test(self):
        return self.loss_fn(self.params, self.rng, (self.x_test, self.xt_test))

    def finalize(self):
        xt_pred = jax.vmap(partial(equation_of_motion, self._learned_lagrangian(self.forward_fn.apply, self.state['rng'], self.state['params'])))(self.x_test)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
        axes[0].scatter(self.xt_test[:, 2], xt_pred[:, 2], s=6, alpha=0.2)
        axes[0].set_title('Predicting $\dot q$')
        axes[0].set_xlabel('$\dot q$ actual')
        axes[0].set_ylabel('$\dot q$ predicted')
        axes[1].scatter(self.xt_test[:, 3], xt_pred[:, 3], s=6, alpha=0.2)
        axes[1].set_title('Predicting $\ddot q$')
        axes[1].set_xlabel('$\ddot q$ actual')
        axes[1].set_ylabel('$\ddot q$ predicted')
        plt.tight_layout()
        plt.show()

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, rng, batch):
        state, targets = batch
        preds = jax.vmap(partial(equation_of_motion, self._learned_lagrangian(params, rng)))(state)
        return jnp.mean((preds - targets) ** 2)
    
    def _learned_lagrangian(self, params, rng):
        def lagrangian(q, q_t):
            assert q.shape == (2,)
            state = normalize_dp(jnp.concatenate([q, q_t]))
            return jnp.squeeze(self.forward_fn.apply(params, rng, x=state), axis=-1)
        return lagrangian

if __name__ == '__main__':
    experiment = DoublePendulumExperiment()
    experiment.train()
    experiment.finalize()