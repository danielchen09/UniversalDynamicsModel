import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import jax.numpy as jnp
from dm_control import mujoco
from scipy.spatial.transform.rotation import Rotation as R
import moviepy.editor as mp

def gif2mp4(filename='test.gif'):
    name = filename.split('.')[0]
    clip = mp.VideoFileClip(f'{name}.gif')
    clip.write_videofile(f'{name}.mp4')

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
    q = np.zeros_like(x)
    if len(x.shape) == 1:
        for i in range(q.shape[0]):
            q[i] = x[i] - np.sum(x[:i])
        return jnp.array(q)
    x = np.array(x)
    for i in range(q.shape[1]):
        q[:, i] = x[:, i] - np.sum(q[:, 0:i])
    return jnp.array(q)

def q2x(q):
    x = np.zeros_like(q)
    if len(x.shape) == 1:
        for i in range(x.shape[0]):
            x[i] = np.sum(q[:i])
        return jnp.array(x)
    q = np.array(q)
    for i in range(q.shape[1]):
        x[:, i] = np.sum(q[:, 0:i], axis=1)
    return jnp.array(x)

def wrap_angle(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def mat2euler(m):
    n = m.shape[0]
    euler = jnp.zeros((n, 3))
    for i in range(n):
        euler[i] = R.from_matrix(m[i].reshape(3, 3)).as_euler('zyx')
    return euler

def convert_ds(x, xt, u, n_traj):
    x0 = x.reshape(n_traj, -1, x.shape[-1])
    x1 = x0[:, 1:, :].reshape(-1, x.shape[-1])
    x0 = x0[:, :-1, :].reshape(-1, x.shape[-1])
    xt0 = xt.reshape(n_traj, -1, xt.shape[1])
    xt0 = xt0[:, :-1, :].reshape(-1, xt.shape[-1])
    u0 = u.reshape(n_traj, -1, u.shape[-1])
    u0 = u0[:, :-1, :].reshape(-1, u.shape[-1])
    return x0, x1, xt0, u0
        