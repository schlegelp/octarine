import warnings
import numpy as np

from pathlib import Path
from tqdm.auto import trange
from scipy.spatial.transform import Rotation


def make_rotation_video(viewer, video_path, n_frames=100, fps=30, axis="z"):
    """Create a video of the viewer rotating around the specified axis.

    We're using imagio to create the video. Make sure to install if you haven't already:

      `pip install imageio`

    Parameters
    ----------
    viewer :    Viewer
                The viewer to capture frames from.
    video_path : str, optional
                The path to save the video to. Should end with .mp4 or .gif.
                If None, the frames will be returned as a list of numpy arrays
                instead of being saved to a video.
    n_frames :  int
                The number of frames in the video.
    fps :       int
                The frames per second of the video.
    axis :      str
                The axis to rotate around, one of 'x', 'y', or 'z'.

    Returns
    -------
    list of numpy arrays
        If video_path is None, returns a list of frames as numpy arrays. Otherwise, returns `None`.

    """
    if video_path is not None:
        try:
            import imageio
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "imageio is required to create videos. Please install it with `pip install imageio`."
            )

    if isinstance(video_path, str):
        video_path = Path(video_path)

    if isinstance(video_path, Path):
        if video_path.suffix not in (".mp4", ".gif"):
            warnings.warn("video_path should really end with .mp4 or .gif")

    assert axis in ("x", "y", "z"), "Axis must be one of 'x', 'y', or 'z'"
    axis_ix = {"x": 0, "y": 1, "z": 2}[axis]
    frames = []
    orig_orientation = Rotation.from_quat(viewer.get_view().get("rotation", [1, 0, 0, 0]))
    for i in trange(n_frames):  # 180 frames for full rotation
        rot_vec = [0, 0, 0]
        rot_vec[axis_ix] = 2 * np.pi / n_frames * i
        rotation = Rotation.from_rotvec(rot_vec)
        new_orientation = orig_orientation * rotation

        # Set the new rotation
        view = viewer.get_view()
        view["rotation"] = new_orientation.as_quat().tolist()
        viewer.set_view(view)

        # capture screenshot
        frames.append(viewer.screenshot(filename=None, alpha=False))

    if video_path:
        imageio.mimsave(video_path, frames, fps=fps)
    else:
        return frames
