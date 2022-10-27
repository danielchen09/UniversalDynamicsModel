from cmath import isnan
from dataset import MultipleSwimmerDataset, SwimmerJointDataset
from dblpen import DoublePendulumBodyDataset
from lnn import *
from tqdm import tqdm


class AnalyticalDoublePendulumExperiment(DoublePendulumExperiment):
    def __init__(self):
        N = 1500
        dataset_obj = AnalyticalDoublePendulumDataset()
        t_train = np.arange(N, dtype=np.float32)
        t_test = np.arange(N, 2 * N, dtype=np.float32)
        super().__init__(dataset_obj, t_train, t_test, save_path='saves/analytical_dblpen.pkl', patience=100)
    
    def _finalize(self):
        t_show = np.arange(500, dtype=np.float32) / 100
        x_show, xt_show, u_show = self.dataset_obj.get_data(self.x0, t_show)
        x_show = x_show[:t_show.shape[0] - 1]
        u_show = u_show[:t_show.shape[0] - 1]
        x_pred = solve_lagrangian(
            self._learned_lagrangian(self.state['params'], self.state['rng']), 
            self._learned_tau(self.state['params'], self.state['rng']), 
            self.x0, t_show, u_show[0])
        x_show = x_show[:t_show.shape[0] - 1]
        x_pred = x_pred[:t_show.shape[0] - 1]
        plt.plot(np.arange(x_show.shape[0]), (x_show - x_pred).mean(axis=1))
        plt.show()

        shape = x_show.shape[1] // 2
        fig, axes = plt.subplots(2, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(2):
            for j in range(shape):
                axes[i][j].scatter(x_show[:, i * shape + j], x_pred[:, i * shape + j], s=1, alpha=0.2)
                axes[i][j].set_title(f'Predicting $\ddot q_{i}$')
                axes[i][j].set_xlabel(f'$q{"t"*i}_{j}$ actual')
                axes[i][j].set_ylabel(f'$q{"t"*i}_{j}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        x_show = x_show[::10]
        x_pred = x_pred[::10]

        frames = []
        frames.append(self.dataset_obj.plot_trajectory(x_show))
        frames.append(self.dataset_obj.plot_trajectory(x_pred))
        frames = np.hstack(frames)
        generate_gif(frames, dt=10)

class MujocoDoublePendulumExperiment(DoublePendulumExperiment):
    def __init__(self):
        dataset_obj = MujocoDoublePendulumDataset(use_action=False, use_friction=False)
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        N = 1500
        t_train = np.arange(N, dtype=np.float32) 
        t_test = np.arange(N, 2 * N, dtype=np.float32) 
        super().__init__(dataset_obj, t_train, t_test, 
                        epochs=1000 * 400,
                        lr_decay=0.1,
                        lr=1e-3,
                        lr_decay_step=1000 * 100,
                        l2reg=1e-3, 
                        patience=100,
                        save_path='saves/mujoco_dblpen.pkl',
                        enable_tau=False)
    
    def _finalize(self):
        t_show = np.arange(200, dtype=np.float32) / 100
        x_show, xt_show, u_show = self.dataset_obj.get_data(self.x0, t_show)
        x_show = x_show[:t_show.shape[0] - 1]
        u_show = u_show[:t_show.shape[0] - 1]
        x_pred = solve_lagrangian(
            self._learned_lagrangian(self.state['params'], self.state['rng']), 
            self._learned_tau(self.state['params'], self.state['rng']), 
            self.x0, t_show, u_show[0])
        x_show = x_show[:t_show.shape[0] - 1]
        x_pred = x_pred[:t_show.shape[0] - 1]
        plt.plot(np.arange(x_show.shape[0]), (x_show - x_pred).mean(axis=1))
        plt.show()

        shape = x_show.shape[1] // 2
        fig, axes = plt.subplots(2, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(2):
            for j in range(shape):
                axes[i][j].scatter(x_show[:, i * shape + j], x_pred[:, i * shape + j], s=1, alpha=0.2)
                axes[i][j].set_title(f'Predicting $\ddot q_{i}$')
                axes[i][j].set_xlabel(f'$q{"t"*i}_{j}$ actual')
                axes[i][j].set_ylabel(f'$q{"t"*i}_{j}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        x_show = x_show[::10]
        x_pred = x_pred[::10]

        frames = []
        frames.append(self.dataset_obj.plot_trajectory(x_show))
        frames.append(self.dataset_obj.plot_trajectory(x_pred))
        frames = np.hstack(frames)
        generate_gif(frames, dt=1)

class MujocoDoublePendulumExperimentFriction(DoublePendulumExperiment):
    def __init__(self, enable_tau=True):
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        total = 2000
        N = 1000
        dataset_obj = MultiplePendulumDataset(total // N)
        t_train = np.arange(N, dtype=np.float32)
        t_test = np.arange(N, 2 * N, dtype=np.float32)
        super().__init__(dataset_obj, t_train, t_test,
                        epochs=1000 * 75,
                        lr_decay=0.35,
                        lr=1e-3,
                        lr_decay_step=1000 * 40,
                        l2reg=3e-3,
                        enable_tau=enable_tau)

class MujocoDoublePendulumExperimentAction(DoublePendulumExperiment):
    def __init__(self, enable_tau=True):
        dataset_obj = MujocoDoublePendulumDataset(use_action=False, use_friction=True)
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        N = 1500
        t_train = np.arange(N, dtype=np.float32)
        t_test = np.arange(N, 2 * N, dtype=np.float32)
        super().__init__(dataset_obj, t_train, t_test,
                        epochs=1000 * 100,
                        lr_decay=0.35,
                        lr=1e-3,
                        lr_decay_step=1000 * 40,
                        l2reg=3e-3, 
                        enable_tau=enable_tau)

class MujocoDoublePendulumExperimentBody(DoublePendulumExperiment):
    def __init__(self, enable_tau=True):
        dataset_obj = DoublePendulumBodyDataset(use_action=False, use_friction=False)
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        N = 1500
        t_train = np.arange(N, dtype=np.float32)
        t_test = np.arange(N, 2 * N, dtype=np.float32)
        super().__init__(dataset_obj, t_train, t_test,
                        epochs=1000 * 150,
                        lr_decay=0.3,
                        lr=1e-3,
                        lr_decay_step=1000 * 40,
                        l2reg=1e-2, 
                        enable_tau=enable_tau,
                        save_path='saves/lnn_mujoco_body.pkl',
                        noise=0.3)

class MujocoSwimmerExperiment(Experiment):
    def __init__(self):
        dataset_obj = SwimmerJointDataset(use_action=True, use_friction=False)
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        N = 1500
        t_train = np.arange(N, dtype=np.float32) 
        t_test = np.arange(N, 2 * N, dtype=np.float32) 
        super().__init__(np.zeros(10), dataset_obj, t_train, t_test, 
                        epochs=1000 * 500,
                        lr_decay=0.1,
                        lr=1e-3,
                        lr_decay_step=1000 * 100,
                        l2reg=1e-3, 
                        patience=100,
                        save_path='saves/mujoco_swimmer.pkl',
                        enable_tau=True)
    
    def _finalize(self):
        t_show = np.arange(200, dtype=np.float32) / 100
        x_show, xt_show, u_show = self.dataset_obj.get_data(self.x0, t_show)
        x_show = x_show[:t_show.shape[0] - 1]
        u_show = u_show[:t_show.shape[0] - 1]
        # x_pred = solve_lagrangian(
        #     self._learned_lagrangian(self.state['params'], self.state['rng']), 
        #     self._learned_tau(self.state['params'], self.state['rng']), 
        #     self.x0, t_show, u_show[0])
        x_pred = solve_lagrangian_discrete(
            self._learned_lagrangian(self.state['params'], self.state['rng']),
            self._learned_tau(self.state['params'], self.state['rng']),
            np.zeros(16),
            t_show,
            u_show
        )
        x_show = x_show[:t_show.shape[0] - 1]
        x_pred = x_pred[:t_show.shape[0] - 1]
        plt.plot(np.arange(x_show.shape[0]), (x_show - x_pred).mean(axis=1))
        plt.show()

        shape = x_show.shape[1] // 2
        fig, axes = plt.subplots(2, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(2):
            for j in range(shape):
                axes[i][j].scatter(x_show[:, i * shape + j], x_pred[:, i * shape + j], s=1, alpha=0.2)
                axes[i][j].set_title(f'Predicting $\ddot q_{i}$')
                axes[i][j].set_xlabel(f'$q{"t"*i}_{j}$ actual')
                axes[i][j].set_ylabel(f'$q{"t"*i}_{j}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        x_show = x_show[::10]
        x_pred = x_pred[::10]

        frames = []
        frames.append(self.dataset_obj.plot_trajectory(x_show))
        frames.append(self.dataset_obj.plot_trajectory(x_pred))
        frames = np.hstack(frames)
        generate_gif(frames, dt=1)

class MujocoSwimmerJointExperiment(Experiment):
    def __init__(self):
        self.n_traj = 20
        # dataset_obj = SwimmerJointDataset(use_action=True, use_friction=False)
        dataset_obj = MultipleSwimmerDataset(self.n_traj, use_action=True, use_friction=False)
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        N = 1000
        t_train = np.arange(N, dtype=np.float32) / 100
        t_test = np.arange(N, 2 * N, dtype=np.float32) / 100
        print(t_train)
        super().__init__(np.zeros(10), dataset_obj, t_train, t_test,
                        epochs=1000 * 400,
                        lr_decay=0.3,
                        lr=1e-3,
                        lr_decay_step=1000 * 30,
                        l2reg=3e-3, 
                        enable_tau=True,
                        save_path='saves/lnn_swimmer_joint.pkl',
                        noise=0,
                        log_step=1000,
                        patience=20)

        self.dt = t_train[1] - t_train[0]
        
        self.x_train = q2x(wrap_angle(self.x_train))
        self.x_test = q2x(wrap_angle(self.x_test))

        # self.x0_train = self.x_train.reshape(self.n_traj, -1, self.x_train.shape[-1])
        # self.x1_train = self.x0_train[:, 1:, :].reshape(-1, self.x_train.shape[-1])
        # self.x0_train = self.x0_train[:, :-1, :].reshape(-1, self.x_train.shape[-1])
        # self.xt0_train = self.xt_train.reshape(self.n_traj, -1, self.xt_train.shape[1])
        # self.xt0_train = self.xt0_train[:, :-1, :].reshape(-1, self.xt_train.shape[-1])
        # self.u0_train = self.u_train.reshape(self.n_traj, -1, self.u_train.shape[-1])
        # self.u0_train = self.u0_train[:, :-1, :].reshape(-1, self.u_train.shape[-1])

        self.x0_train, self.x1_train, self.xt0_train, self.u0_train = convert_ds(self.x_train, self.xt_train, self.u_train, self.n_traj)
        self.x0_test, self.x1_test, self.xt0_test, self.u0_test = convert_ds(self.x_test, self.xt_test, self.u_test, self.n_traj)

        # self.x0_test = self.x_test.reshape(self.n_traj, -1, self.x_test.shape[-1])
        # self.x1_test = self.x0_test[:, 1:, :].reshape(-1, self.x_test.shape[-1])
        # self.x0_test = self.x0_test[:, :-1, :].reshape(-1, self.x_test.shape[-1])
        # self.xt0_test = self.xt_test.reshape(self.n_traj, -1, self.xt_test.shape[1])
        # self.xt0_test = self.xt0_test[:, :-1, :].reshape(-1, self.xt_test.shape[-1])
        # self.u0_test = self.u_test.reshape(self.n_traj, -1, self.u_test.shape[-1])
        # self.u0_test = self.u0_test[:, :-1, :].reshape(-1, self.u_test.shape[-1])
        

        # self.xt_train = self.xt_train.reshape(self.n_traj, -1, self.xt_train.shape[1])
        # self.u_train = self.u_train.reshape(self.n_traj, -1, self.u_train.shape[1])
        # self.t_train = self.t_train.reshape(self.n_traj, -1)
        # self.x_test = self.x_test.reshape(self.n_traj, -1, self.x_test.shape[1])
        # self.xt_test = self.xt_test.reshape(self.n_traj, -1, self.xt_test.shape[1])
        # self.u_test = self.u_test.reshape(self.n_traj, -1, self.u_test.shape[1])
        # self.t_test = self.t_test.reshape(self.n_traj, -1)

    def save_state(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump((
                self.state,
                self.x_train,
                self.xt_train,
                self.u_train,
                self.x0_train,
                self.x1_train,
                self.xt0_train,
                self.u0_train,
                self.x0_test,
                self.x1_test,
                self.xt0_test,
                self.u0_test), f)
    
    def load_state(self):
        with open(self.save_path, 'rb') as f:
            save = pickle.load(f)
            self.state, self.x_train, self.xt_train, self.u_train, self.x0_train, self.x1_train, self.xt0_train, self.u0_train, self.x0_test, self.x1_test, self.xt0_test, self.u0_test = save
        

    def _learned_dynamics(self, params, rng):
        @jax.jit
        def dynamics_fn(state, u):
            xdtt = jax.vmap(partial(equation_of_motion, self._learned_lagrangian(params, rng), self._learned_tau(params, rng)))(state, u)
            return state + xdtt * self.dt, xdtt
        return dynamics_fn


    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, rng, batch):
        x, y_true, xt_true, u = batch
        x_pred, xt_pred = self._learned_dynamics(params, rng)(x, u)
        return jnp.mean((x_pred - y_true) ** 2 + (xt_pred - xt_true) ** 2)

    def train_step(self):
        self.state, metric = self.updater.update(self.state, (self.x0_train, self.x1_train, self.xt0_test, self.u0_train))
        self.params = self.state['params']
        self.rng = self.state['rng']
        return metric

    def test(self):
        return self.loss_fn(self.params, self.rng, (self.x0_test, self.x1_test, self.xt0_test, self.u0_test))


    def _finalize(self):
        x1_pred, xt_pred = self._learned_dynamics(self.state['params'], self.state['rng'])(self.x0_test, self.u0_test)
        shape = x1_pred.shape[1] // 2
        fig, axes = plt.subplots(2, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(2):
            for j in range(shape):
                axes[i][j].scatter(self.x1_test[:, i * shape + j], x1_pred[:, i * shape + j], s=1, alpha=0.2)
                axes[i][j].set_title(f'Predicting $ q_{i}$')
                axes[i][j].set_xlabel(f'$q_{j}$ actual')
                axes[i][j].set_ylabel(f'$q_{j}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        t_show = np.arange(1000) / 100
        x_show, xt_show, u_show = self.dataset_obj.get_data(self.x0, t_show)
        breakpoint()
        x_show = x_show[:t_show.shape[0] - 1]
        x_pred = solve_lagrangian_discrete(
                self._learned_lagrangian(self.state['params'], self.state['rng']), 
                self._learned_tau(self.state['params'], self.state['rng']),
                x_show[0],
                t_show,
                u_show)[:t_show.shape[0] - 1]
        plt.plot(np.arange(x_show.shape[0]), (x_show - x_pred).mean(axis=1))
        plt.show()

        shape = x_show.shape[1] // 2
        fig, axes = plt.subplots(2, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(2):
            for j in range(shape):
                axes[i][j].scatter(x_show[:, i * shape + j], x_pred[:, i * shape + j], s=1, alpha=0.2)
                axes[i][j].set_title(f'Predicting $\ddot q_{i}$')
                axes[i][j].set_xlabel(f'$q{"t"*i}_{j}$ actual')
                axes[i][j].set_ylabel(f'$q{"t"*i}_{j}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        x_show = x_show[::10]
        x_pred = x_pred[::10]

        env = self.dataset_obj.get_env()
        frames = []
        for x_s, x_p in zip(x_show, x_pred):
            frame = []
            state = env.physics.get_state()
            state = np.zeros_like(state)
            state[3:8] = x_s[:5]
            env.physics.set_state(state)
            env.physics.step()
            frame.append(env.physics.render())
            state = env.physics.get_state()
            state = np.zeros_like(state)
            state[3:8] = x_p[:5]
            env.physics.set_state(state)
            env.physics.step()
            frame.append(env.physics.render())
            frames.append(np.hstack(frame))
        generate_gif(frames, dt=1)

class BaselineExperiment(MujocoSwimmerJointExperiment):
    def __init__(self):
        super().__init__()
        def forward_fn(x, u):
            mlp = hk.nets.MLP(
                [128, 128, 128, x.shape[-1]], 
                activation=jax.nn.softplus, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal'),
                b_init=hk.initializers.RandomNormal())
            return mlp(jnp.concatenate([x, u], axis=-1))

        self.forward_fn = hk.transform(forward_fn)
        self.scheduler = optax.piecewise_constant_schedule(self.lr, boundaries_and_scales={i: self.lr_decay for i in range(self.lr_decay_step, self.epochs, self.lr_decay_step)})
        self.optimizer = optax.adamw(self.scheduler, weight_decay=self.l2reg)
        self.updater = GradientUpdater(self.forward_fn.init, self.loss_fn, self.optimizer)
        self.rng = jax.random.PRNGKey(self.seed)
        self.state = self.updater.init(self.rng, self.x_train, self.u_train)
        self.params = self.state['params']
        self.rng = self.state['rng']

    # @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, rng, batch):
        x, y_true, xt_true, u = batch
        preds = self.forward_fn.apply(params, rng, x=x, u=u)
        return jnp.mean((preds - y_true) ** 2)

    def finalize(self, save_path=None):
        if not save_path:
            save_path = self.save_path
        if save_path:
            self.load_state()
        self._finalize()

    def _finalize(self):
        x1_pred = self.forward_fn.apply(self.state['params'], self.state['rng'], x=self.x0_test, u=self.u0_test)
        shape = x1_pred.shape[1] // 2
        fig, axes = plt.subplots(2, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(2):
            for j in range(shape):
                axes[i][j].scatter(self.x1_test[:, i * shape + j], x1_pred[:, i * shape + j], s=1, alpha=0.2)
                axes[i][j].set_title(f'Predicting $ q_{i}$')
                axes[i][j].set_xlabel(f'$q_{j}$ actual')
                axes[i][j].set_ylabel(f'$q_{j}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        t_show = np.arange(1000) / 100
        x_show, xt_show, u_show = self.dataset_obj.get_data(self.x0, t_show)
        breakpoint()
        x_show = x_show[:t_show.shape[0] - 1]
        u_show = u_show[:t_show.shape[0] - 1]
        x_pred = self.forward_fn.apply(self.state['params'], self.state['rng'], x=x_show, u=u_show)
        plt.plot(np.arange(x_show.shape[0]), (x_show - x_pred).mean(axis=1))
        plt.show()

        shape = x_show.shape[1] // 2
        fig, axes = plt.subplots(2, shape, figsize=(3 * shape, 3), dpi=120)
        for i in range(2):
            for j in range(shape):
                axes[i][j].scatter(x_show[:, i * shape + j], x_pred[:, i * shape + j], s=1, alpha=0.2)
                axes[i][j].set_title(f'Predicting $\ddot q_{i}$')
                axes[i][j].set_xlabel(f'$q{"t"*i}_{j}$ actual')
                axes[i][j].set_ylabel(f'$q{"t"*i}_{j}$ predicted')
            # axes[i].set_ylim(np.quantile(xt_pred[:, shape + i], 0.25), np.quantile(xt_pred[:, shape + i], 0.75))
        plt.tight_layout()
        plt.show()

        x_show = x_show[::10]
        x_pred = x_pred[::10]

        env = self.dataset_obj.get_env()
        frames = []
        for x_s, x_p in zip(x_show, x_pred):
            frame = []
            state = env.physics.get_state()
            state = np.zeros_like(state)
            state[3:8] = x_s[:5]
            env.physics.set_state(state)
            env.physics.step()
            frame.append(env.physics.render())
            state = env.physics.get_state()
            state = np.zeros_like(state)
            state[3:8] = x_p[:5]
            env.physics.set_state(state)
            env.physics.step()
            frame.append(env.physics.render())
            frames.append(np.hstack(frame))
        generate_gif(frames, dt=1)

if __name__ == '__main__':
    # experiment = AnalyticalDoublePendulumExperiment()
    experiment = MujocoSwimmerExperiment()
    experiment.train()
    experiment.finalize()