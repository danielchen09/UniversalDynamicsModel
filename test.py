from os import times
from dataset import SwimmerJointDataset
from dblpen import AnalyticalDoublePendulumDataset, MujocoDoublePendulumDataset
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_gif, gif2mp4

swimmer_ds = SwimmerJointDataset(use_friction=False, use_action=True)

n = 1
ts = np.arange(1500) / 100
x, xt, u = swimmer_ds.get_data(np.zeros(10), ts)

plt.hist(x)
plt.show()

frames = swimmer_ds.plot_trajectory(x)
print(frames.shape)

generate_gif(frames, dt=10)
gif2mp4()