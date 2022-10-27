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
from jax.experimental.host_callback import call
import os

##############################################################################################
# if shit doesnt run remember to run `export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib` first #
##############################################################################################

from physics import *
from updater import GradientUpdater
from dblpen import AnalyticalDoublePendulumDataset, MujocoDoublePendulumDataset, MultiplePendulumDataset
from utils import generate_gif


class Experiment:
    def __init__(self, x0, dataset_obj, t_train, t_test,
                lr = 1e-3,
                grad_clip = 1,
                epochs = 1000 * 150,
                seed = 0,
                lr_decay_step = 1000 * 20,
                lr_decay = 0.6,
                l2reg = 2e-3,
                enable_tau = True,
                save_path=None,
                noise=0,
                patience=20,
                log_step=1000):

        self.lr = lr
        self.grad_clip = grad_clip
        self.epochs = epochs
        self.seed = seed
        self.lr_decay_step = lr_decay_step
        self.lr_decay = lr_decay
        self.l2reg = l2reg
        self.save_path = save_path
        self.noise = noise
        self.patience = patience
        self.log_step = log_step

        self.dataset_obj = dataset_obj
        self.t_train = t_train
        self.t_test = t_test

        self.x0 = x0
        self.x_train, self.xt_train, self.u_train = dataset_obj.get_data(self.x0, self.t_train)
        self.x_test, self.xt_test, self.u_test = dataset_obj.get_data(self.x0, self.t_test)

        # q, q' -> L
        def forward_fn(x, u):
            mlp_l = hk.nets.MLP(
                [128, 128, 1], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            mlp_t = hk.nets.MLP(
                [128, x.shape[-1] // 2], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            return mlp_l(x), mlp_t(jnp.concatenate([x, u], axis=-1))
            
        def forward_fn_no_tau(x, u):
            mlp_l = hk.nets.MLP(
                [128, 128, 1], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            mlp_t = hk.nets.MLP(
                [128, 128, x.shape[-1] // 2], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            return mlp_l(x), mlp_t(jnp.concatenate([x, u], axis=-1)) * 0

        self.forward_fn = hk.transform(forward_fn if enable_tau else forward_fn_no_tau)

        self.scheduler = optax.piecewise_constant_schedule(self.lr, boundaries_and_scales={i: self.lr_decay for i in range(self.lr_decay_step, self.epochs, self.lr_decay_step)})
        self.optimizer = optax.adamw(self.scheduler, weight_decay=self.l2reg)
        self.updater = GradientUpdater(self.forward_fn.init, self.loss_fn, self.optimizer)
        self.rng = jax.random.PRNGKey(self.seed)
        self.state = self.updater.init(self.rng, self.x_train, self.u_train)
        self.params = self.state['params']
        self.rng = self.state['rng']

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            self.state = pickle.load(f)

    def train_step(self):
        def add_noise(z):
            col_means = z.mean(axis=0)
            rand = np.random.random(z.shape) * 2 - 1 # (-1, 1)
            rand = rand * col_means * self.noise
            return z + rand

        x_train = add_noise(self.x_train)
        xt_train = add_noise(self.xt_train)
        u_train = add_noise(self.u_train)
        self.state, metric = self.updater.update(self.state, (x_train, xt_train, u_train, self.t_train))
        self.params = self.state['params']
        self.rng = self.state['rng']
        return metric
    
    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, rng, batch):
        state, targets, u, t = batch
        preds = jax.vmap(partial(equation_of_motion, self._learned_lagrangian(params, rng), self._learned_tau(params, rng)))(state, u)
        return jnp.mean((preds - targets) ** 2)

    def _learned_lagrangian(self, params, rng):
        @jax.jit
        def lagrangian(q, q_t, u):
            state = normalize_dp(jnp.concatenate([q, q_t]))
            return jnp.squeeze(self.forward_fn.apply(params, rng, x=state, u=u)[0], axis=-1)
        return lagrangian

    def _learned_tau(self, params, rng):
        @jax.jit
        def tau(q, q_t, u):
            state = normalize_dp(jnp.concatenate([q, q_t]))
            return self.forward_fn.apply(params, rng, x=state, u=u)[1]
        return tau

    def train(self):
        waiting = 0
        best_loss = float('inf')
        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            metric = self.train_step()
            if metric['step'] % self.log_step == 0:
                train_losses.append(metric['loss'])
                test_loss = self.test()
                test_losses.append(test_loss)
                print(f'epoch {epoch}: loss={metric["loss"]}, test loss={test_loss}, waiting={waiting}')
                if test_loss < best_loss:
                    best_loss = test_loss
                    waiting = 0
                else:
                    waiting += 1
                if waiting >= self.patience:
                    break
        
        train_losses = train_losses[10:]
        test_losses = test_losses[10:]
        plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')
        plt.plot(np.arange(len(test_losses)), test_losses, label='test loss')
        plt.legend()
        plt.show()
        self.save_state()
        return train_losses, test_losses

    def test(self):
        return self.loss_fn(self.params, self.rng, (self.x_test, self.xt_test, self.u_test, self.t_test))

    def save_state(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'state': self.state,
                'train_data': (self.x_train, self.xt_train, self.u_train, self.t_train),
                'test_data': (self.x_test, self.xt_test, self.u_test, self.t_test),
            }, f)
    
    def load_state(self):
        with open(self.save_path, 'rb') as f:
            save = pickle.load(f)
            self.state = save['state']
            self.x_train, self.xt_train, self.u_train, self.t_train = save['train_data']
            self.x_test, self.xt_test, self.u_test, self.t_test = save['test_data']

    def finalize(self, save_path=None):
        if not save_path:
            save_path = self.save_path
        if save_path:
            self.load_state()

        xt_pred = jax.vmap(
            partial(
                equation_of_motion, 
                self._learned_lagrangian(self.state['params'], self.state['rng']),
                self._learned_tau(self.state['params'], self.state['rng'])
            )
        )(self.x_test, self.u_test)
        shape = self.xt_test.shape[1] // 2
        
        fig, axes = plt.subplots(1, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(shape):
            axes[i].scatter(self.xt_test[:, shape + i], xt_pred[:, shape + i], s=1, alpha=0.2)
            axes[i].set_title(f'Predicting $\ddot q_{i}$')
            axes[i].set_xlabel(f'$\ddot q_{i}$ actual')
            axes[i].set_ylabel(f'$\ddot q_{i}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        self._finalize()


        # l_fn = lambda x: self.dataset_obj.lagrangian(*np.split(x, 2))
        # l_real = jax.vmap(l_fn)(x_show)
        # l_pred = jax.vmap(l_fn)(x_pred)
        # plt.plot(t_show, l_real, label='real')
        # plt.plot(t_show, l_pred, label='predicted')
        # plt.title('real v.s. predicted lagrangian')
        # plt.legend()
        # plt.show()

        # e_fn = lambda x: self.dataset_obj.total_energy(*np.split(x, 2))
        # e_real = jax.vmap(e_fn)(x_show)
        # e_pred = jax.vmap(e_fn)(x_pred)
        # plt.plot(t_show, e_real, label='real')
        # plt.plot(t_show, e_pred, label='predicted')
        # plt.title('real v.s. predicted total energy')
        # plt.ylim(-50, 50)
        # plt.legend()
        # plt.show()

        # err = x_pred - x_show
        # plt.plot(t_show, err)
        # plt.title('error')
        # plt.show()

        # frames_real = self.dataset_obj.plot_trajectory(x_show)
        # frames_pred = self.dataset_obj.plot_trajectory(x_pred)
        # frames_real = np.array(frames_real)
        # frames_pred = np.array(frames_pred)
        # frames = np.hstack([frames_real, frames_pred])
        # generate_gif(frames, 'result.gif')
        # generate_gif(frames_real, 'real.gif')
        # generate_gif(frames_pred, 'pred.gif')

    def _finalize(self):
        pass


class DoublePendulumExperiment(Experiment):
    def __init__(self, dataset_obj, t_train, t_test,
                lr = 1e-3,
                grad_clip = 1,
                epochs = 1000 * 150,
                seed = 0,
                lr_decay_step = 1000 * 20,
                lr_decay = 0.6,
                l2reg = 2e-3,
                enable_tau = True,
                save_path = None,
                noise=0,
                patience=20,
                log_step=1000):
        super().__init__(np.array([3 / 7 * np.pi, 3 / 4 * np.pi, 0, 0], dtype=np.float32), dataset_obj, t_train, t_test,
                lr = lr,
                grad_clip = grad_clip,
                epochs = epochs,
                seed = seed,
                lr_decay_step = lr_decay_step,
                lr_decay = lr_decay,
                l2reg = l2reg,
                enable_tau = enable_tau,
                save_path = save_path,
                noise=noise,
                patience=patience,
                log_step=log_step)