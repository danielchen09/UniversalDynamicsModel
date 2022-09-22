from dm_control import suite
from dm_control.suite import acrobot
from dm_control.suite import common
from dm_control.rl import control
from dm_control.utils import io as resources
import numpy as np
import xml.etree.ElementTree as ET

from utils import *

class MujocoDataset:
    def __init__(self, use_friction=False, use_action=False):
        self.use_friction = use_friction
        self.use_action = use_action

    def get_env(self):
        pass

    def init_env(self, env, x0):
        pass

    def get_x(self, env):
        pass

    def get_xt(self, env):
        pass

    def get_u(self, action):
        pass

    def get_data(self, x0, times):
        x0, x0_t = np.split(x0, 2)
        rng = np.random.RandomState(42)
        interval = times[1] - times[0]
        assert abs(interval * 100 - round(interval * 100)) < 1e-4
        steps = round((times[-1] - times[0]) * 100)
        presteps = round(times[0] * 100)

        # env = suite.load('acrobot', 'swingup')
        env = self.get_env()
        action_spec = env.action_spec()
        self.init_env(env, x0)

        x = []
        xt = []
        u = []
        i = 0
        for _ in range(presteps):
            env.physics.step()
        for step in range(steps):
            if step == round((times[i] - times[0]) * 100):
                x.append(self.get_x(env))
                action = np.zeros(action_spec.shape)
                if self.use_action:
                    action = rng.uniform(low=action_spec.minimum, high=action_spec.maximum, size=action_spec.shape)
                    env.physics.set_control(action)
                    env.physics.forward()
                u.append(self.get_u(action))
                xt.append(self.get_xt(env))
                i += 1
            env.physics.step()
        return np.vstack(x), np.vstack(xt), np.vstack(u)


class MultipleTrialsWrapper:
    def __init__(self, dataset_obj, trials, x0_range, x0_size):
        self.dataset_obj = dataset_obj
        self.trials = trials
        self.rng = np.random.RandomState(42)
        self.x0_range = x0_range
        self.x0_size = x0_size
    
    def get_env(self):
        return self.dataset_obj.get_env()

    def get_data(self, x0, times):
        xs = []
        xts = []
        us = []
        for _ in range(self.trials):
            x0 = self.rng.uniform(low=self.x0_range[0], high=self.x0_range[1], size=self.x0_size) * np.pi # [-pi/2, pi/2]
            x, xt, u = self.dataset_obj.get_data(x0, times)
            xs.append(x)
            xts.append(xt)
            us.append(u)
        return np.concatenate(xs), np.concatenate(xts), np.concatenate(us)


class SwimmerJointDataset(MujocoDataset):
    def get_env(self):
        tree = ET.parse('xmls/swimmer6.xml')
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
    
    def get_x(self, env):
        return np.hstack([env.physics.data.qpos[3:], env.physics.data.qvel[3:]])

    def get_xt(self, env):
        return np.hstack([env.physics.data.qvel[3:], env.physics.data.qacc[3:]])
    
    def get_u(self, action):
        return action

class MultipleSwimmerDataset(MultipleTrialsWrapper):
    def __init__(self, trials, **kwargs):
        dataset_obj = SwimmerJointDataset(**kwargs)
        super().__init__(dataset_obj, trials, (-1, 1), 16)