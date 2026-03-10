from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from rotorpy.utils.shapes import Quadrotor

import os

class ClosingFuncAnimation(FuncAnimation):
    def __init__(self, fig, func, *args, **kwargs):
        self._close_on_finish = kwargs.pop('close_on_finish')
        FuncAnimation.__init__(self, fig, func, *args, **kwargs)
        
    def _step(self, *args):
        still_going = FuncAnimation._step(self, *args)
        if self._close_on_finish and not still_going:
            plt.close(self._fig)

def _decimate_index(time, sample_time):
    """
    Given sorted lists of source times and sample times, return indices of
    source time closest to each sample time.
    """
    index = np.arange(time.size)
    sample_index = np.round(np.interp(sample_time, time, index)).astype(int)
    return sample_index

def animate(time, position, rotation, wind, animate_wind, world, filename=None, blit=False, show_axes=True, close_on_finish=False, show_trace=True, zoom_margin=0.5, show_wirebox=False):
    """
    Animate a completed simulation result based on the time, position, and
    rotation history. The animation may be viewed live or saved to a .mp4 video
    (slower, requires additional libraries).

    For a live view, it is absolutely critical to retain a reference to the
    returned object in order to prevent garbage collection before the animation
    has completed displaying.

    Below, M corresponds to the number of drones you're animating. If M is None, i.e. the arrays are (N,3) and (N,3,3), then it is assumed that there is only one drone.
    Otherwise, we iterate over the M drones and animate them on the same axes.

    N is the number of time steps in the simulation.

    Parameters
        time, (N,) with uniform intervals
        position, (N,M,3)
        rotation, (N,M,3,3)
        wind, (N,M,3) world wind velocity
        animate_wind, if True animate wind vector
        world, a World object
        filename, for saved video, or live view if None
        blit, if True use blit for faster animation, default is False
        show_axes, if True plot axes, default is True
        close_on_finish, if True close figure at end of live animation or save, default is False
        show_trace, if True draw trajectory trace behind the drone, default is True
        zoom_margin, padding (meters) around trajectory bounds for axis limits; smaller = more zoomed in. Default 0.5.
        show_wirebox, if True draw the world boundary wireframe, default False
    """

    # Check if there is only one drone.
    if len(position.shape) == 2:
        position = np.expand_dims(position, axis=1)
        rotation = np.expand_dims(rotation, axis=1)
        wind = np.expand_dims(wind, axis=1)
    M = position.shape[1]

    # Temporal style.
    rtf = 1.0 # real time factor > 1.0 is faster than real time playback
    render_fps = 30

    # Normalize the wind by the max of the wind magnitude on each axis, so that the maximum length of the arrow is decided by the scale factor
    wind_mag = np.max(np.linalg.norm(wind, axis=-1), axis=1)             # Get the wind magnitude time series
    max_wind = np.max(wind_mag)                         # Find the maximum wind magnitude in the time series

    if max_wind != 0:
        wind_arrow_scale_factor = 1                         # Scale factor for the wind arrow
        wind = wind_arrow_scale_factor*wind / max_wind

    # Decimate data to render interval; always include t=0.
    if time[-1] != 0:
        sample_time = np.arange(0, time[-1], 1/render_fps * rtf)
    else:
        sample_time = np.zeros((1,))
    index = _decimate_index(time, sample_time)
    time = time[index]
    position = position[index,:]
    rotation = rotation[index,:]
    wind = wind[index,:]

    # Set up axes.
    if filename is not None:
        if isinstance(filename, Path):
            fig = plt.figure(filename.name)
        else:
            fig = plt.figure(filename)
    else:
        fig = plt.figure('Animation')
    fig.clear()
    ax = fig.add_subplot(projection='3d')
    if not show_axes:
        ax.set_axis_off()

    quads = [Quadrotor(ax, wind=animate_wind, wind_scale_factor=1) for _ in range(M)]

    world_artists = world.draw(ax, draw_bounds=show_wirebox)

    # Zoom to trajectory bounds with margin (override world's default limits)
    xmin, xmax = position[:, :, 0].min() - zoom_margin, position[:, :, 0].max() + zoom_margin
    ymin, ymax = position[:, :, 1].min() - zoom_margin, position[:, :, 1].max() + zoom_margin
    zmin, zmax = position[:, :, 2].min() - zoom_margin, position[:, :, 2].max() + zoom_margin
    width = max(xmax - xmin, ymax - ymin, zmax - zmin)
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    half = width / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    # Trajectory trace lines (one per drone)
    trace_lines = []
    if show_trace:
        for i in range(M):
            line, = ax.plot(position[0:1, i, 0], position[0:1, i, 1], position[0:1, i, 2],
                            color=f'C{i % 10}', alpha=0.7, linewidth=1.5)
            trace_lines.append(line)

    title_artist = ax.set_title('t = {}'.format(time[0]))

    def init():
        ax.draw(fig.canvas.get_renderer())
        artists = world_artists + [title_artist] + [q.artists for q in quads]
        if trace_lines:
            artists = artists + trace_lines
        return artists

    def update(frame):
        title_artist.set_text('t = {:.2f}'.format(time[frame]))
        for i, quad in enumerate(quads):
            quad.transform(position=position[frame,i,:], rotation=rotation[frame,i,:,:], wind=wind[frame,i,:])
        for i, line in enumerate(trace_lines):
            line.set_data(position[:frame+1, i, 0], position[:frame+1, i, 1])
            line.set_3d_properties(position[:frame+1, i, 2])
        artists = world_artists + [title_artist] + [q.artists for q in quads]
        if trace_lines:
            artists = artists + trace_lines
        return artists

    ani = ClosingFuncAnimation(fig=fig,
                        func=update,
                        frames=time.size,
                        init_func=init,
                        interval=1000.0/render_fps,
                        repeat=False,
                        blit=blit,
                        close_on_finish=close_on_finish)

    if filename is not None:
        print('Saving Animation')
        if not ".mp4" in filename:
            filename = filename + ".mp4"
        path = os.path.join(os.path.dirname(__file__),'..','data_out',filename)
        ani.save(path,
                 writer='ffmpeg',
                 fps=render_fps,
                 dpi=400)
        if close_on_finish:
            plt.close(fig)
            ani = None

    return ani