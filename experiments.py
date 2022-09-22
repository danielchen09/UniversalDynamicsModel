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
        super().__init__(dataset_obj, t_train, t_test)


class MujocoDoublePendulumExperiment(DoublePendulumExperiment):
    def __init__(self):
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
                        l2reg=3e-3)

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

class MujocoSwimmerJointExperiment(Experiment):
    def __init__(self):
        # dataset_obj = SwimmerJointDataset(use_action=True, use_friction=False)
        dataset_obj = MultipleSwimmerDataset(10, use_action=True, use_friction=False)
        # t_train = np.linspace(0, 10, 1001)
        # t_test = np.linspace(10, 20, 101)
        N = 150
        t_train = np.arange(N, dtype=np.float32) / 100
        t_test = np.arange(N, 2 * N, dtype=np.float32) / 100
        print(t_train)
        super().__init__(np.zeros(10), dataset_obj, t_train, t_test,
                        epochs=1000 * 150,
                        lr_decay=0.3,
                        lr=1e-3,
                        lr_decay_step=1000 * 40,
                        l2reg=3e-3, 
                        enable_tau=True,
                        save_path='saves/lnn_swimmer_joint.pkl',
                        noise=0)

    def _finalize(self):
        t_show = np.arange(1000) / 100
        x_show, xt_show, u_show = self.dataset_obj.get_data(self.x0, t_show)
        x_show = x_show[:t_show.shape[0] - 1]
        x_pred = solve_lagrangian_discrete(
                self._learned_lagrangian(self.state['params'], self.state['rng']), 
                self._learned_tau(self.state['params'], self.state['rng']),
                x_show[0],
                t_show,
                u_show)[:t_show.shape[0] - 1]
        print(((x_show - x_pred) ** 2).mean())

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
    experiment = MujocoSwimmerJointExperiment()
    experiment.train()
    # save_path = 'saves/lnn_analytical.pkl'
    save_path = 'saves/lnn_swimmer_joint_xyz.pkl'
    experiment.finalize()