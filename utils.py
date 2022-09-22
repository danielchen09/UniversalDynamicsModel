import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import jax.numpy as jnp
from dm_control import mujoco
from scipy.spatial.transform.rotation import Rotation as R

def generate_gif(frames, save_path='test.gif', dt=100):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=dt, blit=True, repeat=False)
    anim.save(save_path)

def x2q(x):
    q = np.zeros(x.shape[0])
    for i in range(q.shape[0]):
        q[i] = x[i] - np.sum(q[0:i])
    return x

def q2x(q):
    x = jnp.zeros(q.shape[0])
    for i in range(q.shape[0]):
        x[i] = jnp.sum(x[0:i])
    return x

def wrap_angle(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def mat2euler(m):
    n = m.shape[0]
    euler = jnp.zeros((n, 3))
    for i in range(n):
        euler[i] = R.from_matrix(m[i].reshape(3, 3)).as_euler('zyx')
    return euler