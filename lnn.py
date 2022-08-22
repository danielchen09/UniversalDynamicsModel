import pickle
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from functools import partial
import haiku as hk
import optax
import matplotlib.pyplot as plt
from jax.example_libraries import optimizers
from jax.example_libraries import stax
import os

##############################################################################################
# if shit doesnt run remember to run `export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib` first #
##############################################################################################

from physics import *
from updater import GradientUpdater
from dblpen import AnalyticalDoublePendulumDataset, MujocoDoublePendulumDataset
from utils import generate_gif

class DoublePendulumExperiment:
    def __init__(self, dataset_obj, t_train, t_test,
                lr = 1e-3,
                grad_clip = 1,
                epochs = 1000 * 150,
                seed = 0,
                lr_decay_step = 1000 * 20,
                lr_decay = 0.6,
                l2reg = 2e-3):

        self.lr = lr
        self.grad_clip = grad_clip
        self.epochs = epochs
        self.seed = seed
        self.lr_decay_step = lr_decay_step
        self.lr_decay = lr_decay
        self.l2reg = l2reg

        self.dataset_obj = dataset_obj
        self.t_train = t_train
        self.t_test = t_test

        self.x0 = np.array([3 / 7 * np.pi, 3 / 4 * np.pi, 0, 0], dtype=np.float32)
        self.x_train, self.xt_train = dataset_obj.get_data(self.x0, self.t_train)
        
        self.x_test, self.xt_test = dataset_obj.get_data(self.x0, self.t_test)

        # q, q' -> L
        def lagrangian_forward_fn(x):
            mlp = hk.nets.MLP(
                [128, 128, 1], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            return mlp(x)
        self.lagrangian_forward_fn = hk.transform(lagrangian_forward_fn)

        # q, q', u -> tau
        def tau_forward_fn(x, u):
            x = jnp.concatenate([x, u], axis=1)
            mlp = hk.nets.MLP(
                [128, 128, 1], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            return mlp(x)

        self.scheduler = optax.piecewise_constant_schedule(self.lr, boundaries_and_scales={i: self.lr_decay for i in range(self.lr_decay_step, self.epochs, self.lr_decay_step)})
        self.optimizer = optax.adamw(self.scheduler, weight_decay=self.l2reg)
        self.updater = GradientUpdater(self.lagrangian_forward_fn.init, self.loss_fn, self.optimizer)
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
        plt.legend()
        plt.show()
        return train_losses, test_losses

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            self.state = pickle.load(f)

    def test(self):
        return self.loss_fn(self.params, self.rng, (self.x_test, self.xt_test))

    def finalize(self, save_path=None):
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.state, f)
        xt_pred = jax.vmap(partial(equation_of_motion, self._learned_lagrangian(self.state['params'], self.state['rng'])))(self.x_test)
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

        t_show = np.linspace(0, 20, 301)
        x_show, xt_show = self.dataset_obj.get_data(self.x0, t_show)
        x_pred = jax.device_get(solve_lagrangian(self._learned_lagrangian(self.state['params'], self.state['rng']), self.x0, t=t_show, rtol=1e-10, atol=1e-10))
        l_fn = lambda x: self.dataset_obj.lagrangian(*np.split(x, 2))
        l_real = jax.vmap(l_fn)(x_show)
        l_pred = jax.vmap(l_fn)(x_pred)
        plt.plot(t_show, l_real, label='real')
        plt.plot(t_show, l_pred, label='predicted')
        plt.title('real v.s. predicted lagrangian')
        plt.legend()
        plt.show()

        e_fn = lambda x: self.dataset_obj.total_energy(*np.split(x, 2))
        e_real = jax.vmap(e_fn)(x_show)
        e_pred = jax.vmap(e_fn)(x_pred)
        plt.plot(t_show, e_real, label='real')
        plt.plot(t_show, e_pred, label='predicted')
        plt.title('real v.s. predicted total energy')
        plt.ylim(-50, 50)
        plt.legend()
        plt.show()

        err = x_pred - x_show
        plt.plot(t_show, err)
        plt.title('error')
        plt.show()

        frames_real = self.dataset_obj.plot_trajectory(x_show)
        frames_pred = self.dataset_obj.plot_trajectory(x_pred)
        frames_real = np.array(frames_real)
        frames_pred = np.array(frames_pred)
        frames = np.hstack([frames_real, frames_pred])
        generate_gif(frames, 'result.gif')
        # generate_gif(frames_real, 'real.gif')
        # generate_gif(frames_pred, 'pred.gif')

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, rng, batch):
        state, targets = batch
        preds = jax.vmap(partial(equation_of_motion, self._learned_lagrangian(params, rng)))(state)
        return jnp.mean((preds - targets) ** 2)
    
    def _learned_lagrangian(self, params, rng):
        @jax.jit
        def lagrangian(q, q_t):
            assert q.shape == (2,)
            state = normalize_dp(jnp.concatenate([q, q_t]))
            return jnp.squeeze(self.lagrangian_forward_fn.apply(params, rng, x=state), axis=-1)
        return lagrangian

class AnalyticalDoublePendulumExperiment(DoublePendulumExperiment):
    def __init__(self):
        N = 1500
        dataset_obj = AnalyticalDoublePendulumDataset()
        t_train = np.arange(N, dtype=np.float32)
        t_test = np.arange(N, 2 * N, dtype=np.float32)
        super().__init__(dataset_obj, t_train, t_test)

class MujocoDoublePendulumExperiment(DoublePendulumExperiment):
    def __init__(self):
        dataset_obj = MujocoDoublePendulumDataset()
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        N = 1500
        t_train = np.arange(N, dtype=np.float32)
        t_test = np.arange(N, 2 * N, dtype=np.float32)
        super().__init__(dataset_obj, t_train, t_test,
                        epochs=1000 * 150,
                        lr_decay=0.35,
                        lr_decay_step=1000 * 40,
                        l2reg=3e-3)

if __name__ == '__main__':
    # experiment = AnalyticalDoublePendulumExperiment()
    experiment = MujocoDoublePendulumExperiment()
    experiment.train()
    # save_path = 'saves/lnn_analytical.pkl'
    save_path = 'saves/lnn_mujoco.pkl'
    experiment.finalize(save_path=save_path)