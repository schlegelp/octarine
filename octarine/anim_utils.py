"""Timeline-based animations for the Octarine viewer.

The centre piece here is the [`Animation`][octarine.Animation] class: a small
timeline that you fill with segments - an orbit, a move to another view, a
pause - and then either play in the viewer or render to a video:

>>> import octarine as oc
>>> v = oc.Viewer()
>>> v.add_mesh(mesh)                                        # doctest: +SKIP
>>> anim = oc.Animation(v)
>>> anim.orbit(duration=6)                                  # doctest: +SKIP
>>> anim.render("orbit.mp4")                                # doctest: +SKIP

Every segment is resolved to explicit start/end camera states when it is
added, which means the timeline can be evaluated at any point in time without
having played the frames before it (`anim.set_time(2.5)`). That is what makes
scrubbing, looping and deterministic rendering work.

"""

import math
import time

from pathlib import Path

import numpy as np
import pylinalg as la

# The camera, the controllers and we all have to agree on where the camera's
# pivot is, so take that calculation from pygfx rather than re-deriving it
from pygfx.cameras._perspective import fov_distance_factor, fov_limit

from . import utils

__all__ = ["Animation", "EASINGS", "OUTPUT_FORMATS"]

# Video formats we can write (via imageio)
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif")

#: What `Animation.render` can write, as ``label -> file suffix``. An empty
#: suffix means a directory with one numbered PNG per frame. The controls
#: panel builds its format dropdown from this.
OUTPUT_FORMATS = {
    "MP4 video": ".mp4",
    "Animated GIF": ".gif",
    "PNG sequence": "",
}


def _smoothstep(t):
    return t * t * (3 - 2 * t)


#: Easing functions accepted by the `easing` parameters in this module. Each
#: maps a progress in [0, 1] to an eased progress in [0, 1]. Anything callable
#: with that signature works just as well.
EASINGS = {
    "linear": lambda t: t,
    "in": lambda t: t * t,
    "out": lambda t: 1 - (1 - t) ** 2,
    "in_out": _smoothstep,
    # Slower still at both ends than `in_out` - reads as a deliberate camera move
    "smooth": lambda t: _smoothstep(_smoothstep(t)),
}


def frame_count(duration, fps):
    """Number of frames a `duration` (in seconds) comes to at `fps`.

    The last frame stops one frame short of the end, so that a full turn loops
    seamlessly - see `Animation.times`.

    """
    return max(int(round(duration * fps)), 0)


def _get_easing(easing):
    """Turn an easing name into a function."""
    if callable(easing):
        return easing
    try:
        return EASINGS[easing]
    except KeyError:
        raise ValueError(
            f"Unknown easing '{easing}'. Use one of {tuple(EASINGS)} or a callable."
        )


def _view_dir(state):
    """Unit vector the camera is looking along."""
    return la.vec_transform_quat((0, 0, -1), np.asarray(state["rotation"], dtype=float))


def _pivot_distance(state):
    """How far the camera stands off the point it is looking at.

    Note that this is zero for an orthographic camera (``fov=0``), which sits
    *at* its pivot - see `fov_distance_factor` in pygfx.

    """
    extent = 0.5 * (state["width"] + state["height"])
    return fov_distance_factor(fov_limit(state.get("fov") or 0)) * extent


def _pivot(state):
    """Point the camera is looking at, i.e. what it rotates around."""
    position = np.asarray(state["position"], dtype=float)
    return position + _view_dir(state) * _pivot_distance(state)


def _position_from_pivot(state, pivot):
    """Camera position that looks at `pivot` with this state's rotation/extent."""
    return np.asarray(pivot, dtype=float) - _view_dir(state) * _pivot_distance(state)


def _copy_state(state):
    """Copy a camera state, arrays and all."""
    return {
        k: (np.array(v, dtype=float) if isinstance(v, np.ndarray) else v)
        for k, v in state.items()
    }


def _quat_slerp(a, b, t):
    """Shortest-arc interpolation between two quaternions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(np.dot(a, b))
    # Same rotation, opposite sign - take the short way round
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:  # too close to slerp stably; a plain lerp is exact enough
        out = a + t * (b - a)
    else:
        theta = math.acos(min(max(dot, -1.0), 1.0))
        sin_theta = math.sin(theta)
        out = (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / sin_theta
    norm = np.linalg.norm(out)
    return out / norm if norm else np.asarray(b, dtype=float)


def _lerp(a, b, t):
    return a + (b - a) * t


def _geom_lerp(a, b, t):
    """Interpolate geometrically - the right thing for extents and zoom.

    Halving the field of view should look like a constant-speed zoom whether it
    happens over the first or the last second, and that is a constant *ratio*
    per unit time, not a constant difference.

    """
    if a > 0 and b > 0:
        return a * (b / a) ** t
    return _lerp(a, b, t)


def _rotate_state(state, axis, angle, pivot):
    """Rotate a camera state around `pivot` by `angle` radians about `axis`."""
    quat = la.quat_from_axis_angle(axis, angle)
    new = _copy_state(state)
    offset = np.asarray(state["position"], dtype=float) - np.asarray(pivot, dtype=float)
    new["position"] = np.asarray(pivot, dtype=float) + la.vec_transform_quat(
        offset, quat
    )
    # Pre-multiplying applies the rotation in world space, which is what we want:
    # the camera swings around the pivot instead of turning on the spot
    new["rotation"] = la.quat_mul(quat, np.asarray(state["rotation"], dtype=float))
    return new


def _normalize_axis(axis, state):
    """Turn an axis specifier into a unit vector.

    Accepts ``"x"``, ``"y"``, ``"z"`` (optionally negated: ``"-y"``), ``"up"``
    for the scene's up direction (the camera's ``reference_up``), ``"view"``
    for the direction the camera is looking along (i.e. a roll) or any
    3-vector.

    """
    if isinstance(axis, str):
        key = axis.strip().lower()
        sign = 1.0
        if key.startswith(("-", "+")):
            sign = -1.0 if key[0] == "-" else 1.0
            key = key[1:]
        if key in ("x", "y", "z"):
            vec = np.zeros(3)
            vec["xyz".index(key)] = 1.0
        elif key in ("up", "camera_up", "reference_up"):
            vec = np.asarray(state.get("reference_up", (0, 1, 0)), dtype=float)
        elif key in ("view", "forward", "roll"):
            vec = _view_dir(state)
        else:
            raise ValueError(
                f"Unknown axis '{axis}'. Use 'x', 'y', 'z', 'up', 'view' "
                "(optionally negated) or a vector."
            )
        vec = vec * sign
    else:
        vec = np.asarray(axis, dtype=float)
        if vec.shape != (3,):
            raise ValueError(f"Expected a 3-vector as axis, got {np.shape(axis)}")

    norm = np.linalg.norm(vec)
    if not norm:
        raise ValueError("Rotation axis must not be zero-length.")
    return vec / norm


def _bounding_sphere(viewer, objects=None):
    """Centre and radius of the sphere enclosing `objects` (or the whole scene).

    Parameters
    ----------
    objects :   str | int | list | visual, optional
                Anything [`Viewer.get_bounds`][octarine.Viewer.get_bounds]
                takes. If ``None``, uses everything on the viewer.

    """
    bounds = viewer.get_bounds(objects)
    if bounds is None:
        raise ValueError("Nothing to animate around - no object takes up any space.")

    # `bounds` is (3, 2), i.e. one row per axis - the other way round from the
    # ((xmin, ymin, zmin), (xmax, ymax, zmax)) that pylinalg wants
    sphere = la.aabb_to_sphere(bounds.T)
    return sphere[:3], float(sphere[3]) or 1.0


def _framed_state(viewer, center, radius, view_dir=None, up=None, scale=1.0):
    """Camera state that frames a sphere, without moving the actual camera.

    We let pygfx do the framing (`camera.show_object`) and read the state back
    off the camera rather than reimplementing it here - the two would drift
    apart sooner or later.

    """
    camera = viewer.camera
    before = camera.get_state()
    try:
        camera.show_object(
            (*np.asarray(center, dtype=float), float(radius)),
            view_dir=view_dir,
            up=up,
            scale=scale,
        )
        return camera.get_state()
    finally:
        camera.set_state(before)


class Segment:
    """A piece of a track, covering ``[start, start + duration)``.

    Subclass this to animate something new: implement `apply` and add the
    segments to a `Track` on an `Animation`. Segments must be able to render
    any point in their time range without having seen the ones before it.

    """

    def __init__(self, duration, easing="linear"):
        if duration < 0:
            raise ValueError(f"Duration must be >= 0, got {duration}")
        self.duration = float(duration)
        self.easing = _get_easing(easing)
        self.easing_name = easing if isinstance(easing, str) else "custom"
        #: Set by the track this segment is added to
        self.start = 0.0

    @property
    def end(self):
        return self.start + self.duration

    def progress(self, t):
        """Eased progress [0-1] at local time `t`."""
        if self.duration <= 0:
            return 1.0
        return float(self.easing(min(max(t / self.duration, 0.0), 1.0)))

    def apply(self, viewer, t):
        """Apply this segment at local time `t` (seconds from its start)."""
        raise NotImplementedError

    def __repr__(self):
        return (
            f"<{type(self).__name__} {self.start:.2f}-{self.end:.2f}s "
            f"easing={self.easing_name}>"
        )


class CameraSegment(Segment):
    """Base class for segments that move the camera."""

    def __init__(self, start_state, end_state, duration, easing="linear"):
        super().__init__(duration, easing=easing)
        self.start_state = _copy_state(start_state)
        self.end_state = _copy_state(end_state)

    def state_at(self, t):
        """Camera state at local time `t`."""
        raise NotImplementedError

    def apply(self, viewer, t):
        # Going through `set_view` (rather than the camera directly) keeps
        # linked viewers in sync
        viewer.set_view(self.state_at(t))


class Hold(CameraSegment):
    """Sit still on a given view."""

    def __init__(self, state, duration):
        super().__init__(state, state, duration)

    def state_at(self, t):
        return self.start_state


class Orbit(CameraSegment):
    """Circle the camera around a pivot."""

    def __init__(self, state, pivot, axis, turns=1, duration=6, easing="linear"):
        self.pivot = np.asarray(pivot, dtype=float)
        self.axis = _normalize_axis(axis, state)
        self.turns = float(turns)
        self.angle = 2 * math.pi * self.turns
        super().__init__(
            state,
            _rotate_state(state, self.axis, self.angle, self.pivot),
            duration,
            easing=easing,
        )

    def state_at(self, t):
        return _rotate_state(
            self.start_state, self.axis, self.angle * self.progress(t), self.pivot
        )


class Transition(CameraSegment):
    """Move from one view to another.

    With ``path="arc"`` (the default) the camera swings around the point it is
    looking at instead of travelling in a straight line - the difference
    between orbiting to the far side of an object and flying through it.

    """

    def __init__(self, start_state, end_state, duration, easing="in_out", path="arc"):
        if path not in ("arc", "linear"):
            raise ValueError(f"`path` must be 'arc' or 'linear', got '{path}'")
        self.path = path
        super().__init__(start_state, end_state, duration, easing=easing)

    def state_at(self, t):
        f = self.progress(t)
        a, b = self.start_state, self.end_state
        state = _copy_state(b)

        state["rotation"] = _quat_slerp(a["rotation"], b["rotation"], f)
        for key in ("width", "height", "depth", "zoom"):
            if key in a and key in b:
                state[key] = _geom_lerp(a[key], b[key], f)
        if a.get("fov") is not None and b.get("fov") is not None:
            state["fov"] = _lerp(a["fov"], b["fov"], f)
        if "reference_up" in a and "reference_up" in b:
            up = _lerp(
                np.asarray(a["reference_up"], dtype=float),
                np.asarray(b["reference_up"], dtype=float),
                f,
            )
            norm = np.linalg.norm(up)
            if norm:
                state["reference_up"] = up / norm

        if self.path == "linear":
            state["position"] = _lerp(
                np.asarray(a["position"], dtype=float),
                np.asarray(b["position"], dtype=float),
                f,
            )
        else:
            # Interpolate what we are looking *at*, then put the camera back at
            # the distance its (interpolated) extent and fov imply
            pivot = _lerp(_pivot(a), _pivot(b), f)
            state["position"] = _position_from_pivot(state, pivot)

        return state


class Track:
    """An ordered sequence of segments, animating one aspect of the scene."""

    def __init__(self, name="track"):
        self.name = name
        self.segments = []

    def __len__(self):
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, i):
        return self.segments[i]

    @property
    def duration(self):
        return self.segments[-1].end if self.segments else 0.0

    def append(self, segment):
        """Add a segment to the end of this track."""
        segment.start = self.duration
        self.segments.append(segment)
        return segment

    def clear(self):
        self.segments = []

    def segment_at(self, t):
        """Return ``(segment, local_time)`` active at time `t`, or ``(None, 0)``."""
        if not self.segments:
            return None, 0.0
        for seg in self.segments:
            if t < seg.end:
                return seg, max(t - seg.start, 0.0)
        last = self.segments[-1]
        return last, last.duration  # past the end: hold the final state

    def apply(self, viewer, t):
        segment, local_t = self.segment_at(t)
        if segment is not None:
            segment.apply(viewer, local_t)

    def __repr__(self):
        return (
            f"<Track '{self.name}' with {len(self)} segment(s), {self.duration:.2f}s>"
        )


class CameraTrack(Track):
    """The camera's timeline."""

    def __init__(self):
        super().__init__(name="camera")

    @property
    def end_state(self):
        """State the camera is left in at the end of this track."""
        return self.segments[-1].end_state if self.segments else None


class Animation:
    """A timeline of camera moves that can be played or rendered to a video.

    Segments are appended one after the other; each starts where the previous
    one ended, so a timeline reads much like the animation itself:

    >>> anim = oc.Animation(v)                              # doctest: +SKIP
    >>> anim.orbit(duration=6)          # one turn around the scene
    >>> anim.move_to("XZ", duration=2)  # swing round to the lateral view
    >>> anim.hold(1)                    # ... and stay there for a second
    >>> anim.render("tour.mp4")

    Parameters
    ----------
    viewer :    Viewer
                Viewer to animate.
    fps :       int
                Frames per second used when rendering (and the rate the
                preview aims for, capped by ``viewer.max_fps``).

    """

    def __init__(self, viewer, fps=30):
        self.viewer = viewer
        self.fps = int(fps)
        self.camera = CameraTrack()
        #: All tracks on this animation. Camera-only for now, but a track
        #: animating e.g. object visibility would simply be added here.
        self.tracks = [self.camera]

        self._start_state = None  # where the timeline picks up from
        self._playback = None  # the function registered with the viewer

    def __repr__(self):
        return (
            f"<Animation {self.duration:.2f}s @ {self.fps} fps, "
            f"{len(self.camera)} camera segment(s)>"
        )

    @property
    def duration(self):
        """Length of the animation in seconds."""
        return max((track.duration for track in self.tracks), default=0.0)

    @property
    def n_frames(self):
        """Number of frames this animation renders to."""
        return frame_count(self.duration, self.fps)

    @property
    def current_state(self):
        """Camera state the next segment will start from."""
        state = self.camera.end_state
        if state is not None:
            return state
        if self._start_state is None:
            self._start_state = self.viewer.get_view()
        return self._start_state

    def clear(self):
        """Remove all segments."""
        for track in self.tracks:
            track.clear()
        self._start_state = None
        return self

    def start_at(self, view=None):
        """Set the view the timeline starts from, without animating to it.

        Parameters
        ----------
        view :  str | dict, optional
                A view as accepted by `Viewer.set_view` (e.g. ``"XY"``) or a
                camera state. Defaults to the viewer's current view.

        """
        if self.camera.segments:
            raise ValueError(
                "`start_at` only works on an empty timeline - use `move_to` to "
                "jump to a view mid-animation."
            )
        self._start_state = self._as_state(view)
        return self

    def hold(self, duration):
        """Stay on the current view for `duration` seconds."""
        self.camera.append(Hold(self.current_state, duration))
        return self

    def orbit(
        self,
        objects=None,
        *,
        turns=1,
        duration=6,
        axis="up",
        recenter=True,
        scale=1.0,
        easing="linear",
    ):
        """Circle the camera around object(s).

        Parameters
        ----------
        objects :   str | list of str | visual(s), optional
                    What to orbit around, given as object ID(s) as shown in the
                    legend (or the visuals themselves). Defaults to everything
                    on the viewer.
        turns :     float
                    Number of turns. Negative values orbit the other way.
        duration :  float
                    Seconds the orbit takes.
        axis :      "up" | "x" | "y" | "z" | "view" | (3, ) vector
                    Axis to rotate about. The default, ``"up"``, is the scene's
                    up direction (the camera's ``reference_up``) and gives a
                    turntable rotation from whatever view you start at. Prefix
                    with ``-`` to flip it.
        recenter :  bool
                    If True (default), frame `objects` before orbiting: the
                    pivot is the centre of their bounding box and the camera is
                    moved to fit them in view. If False, keep the current view
                    and orbit around whatever the camera is already looking at
                    - use this to animate exactly what you see on screen.
        scale :     float
                    Only with ``recenter=True``: how much of the viewport the
                    objects fill. Values > 1 zoom out.
        easing :    str | callable
                    See `octarine.anim_utils.EASINGS`. Note that anything but
                    ``"linear"`` will make a full turn start and/or end slowly.

        """
        state = self.current_state
        if recenter:
            pivot, radius = _bounding_sphere(self.viewer, objects)
            state = _framed_state(self.viewer, pivot, radius, scale=scale)
        elif objects is not None:
            pivot, _ = _bounding_sphere(self.viewer, objects)
        else:
            pivot = _pivot(state)

        self.camera.append(
            Orbit(state, pivot, axis, turns=turns, duration=duration, easing=easing)
        )
        return self

    def move_to(self, view=None, *, duration=2, easing="in_out", path="arc"):
        """Move the camera to another view.

        Parameters
        ----------
        view :      str | dict, optional
                    Target view - anything `Viewer.set_view` takes (``"XY"``,
                    ``"-XZ"``, ...) or a camera state as returned by
                    `Viewer.get_view`. Defaults to the viewer's current view,
                    which is what makes "fly to where I am looking now" work
                    from the GUI.
        duration :  float
                    Seconds the move takes.
        easing :    str | callable
                    Defaults to ``"in_out"`` so the camera accelerates out of
                    and decelerates into the move.
        path :      "arc" | "linear"
                    ``"arc"`` swings around the point being looked at,
                    ``"linear"`` moves the camera in a straight line.

        """
        self.camera.append(
            Transition(
                self.current_state,
                self._as_state(view),
                duration,
                easing=easing,
                path=path,
            )
        )
        return self

    def zoom(self, factor=2, *, duration=2, easing="in_out"):
        """Zoom in (`factor` > 1) or out (`factor` < 1) on the current view."""
        if factor <= 0:
            raise ValueError(f"Zoom factor must be > 0, got {factor}")
        state = _copy_state(self.current_state)
        pivot = _pivot(state)
        for key in ("width", "height", "depth"):
            state[key] = state[key] / factor
        state["position"] = _position_from_pivot(state, pivot)
        self.camera.append(
            Transition(self.current_state, state, duration, easing=easing)
        )
        return self

    def _as_state(self, view):
        """Resolve a view specifier into a camera state."""
        if isinstance(view, dict):
            return view
        # `None` gives the current view, a name (e.g. "XY") what that view
        # would be - neither moves the camera
        return self.viewer.get_view(view)

    def set_time(self, t):
        """Apply the animation's state at time `t` (in seconds)."""
        for track in self.tracks:
            track.apply(self.viewer, t)
        return self

    def times(self, fps=None):
        """Times (in seconds) of the frames this animation renders to.

        The last frame stops one frame short of the end so that a full turn
        loops seamlessly.

        """
        fps = int(fps or self.fps)
        return [i / fps for i in range(frame_count(self.duration, fps))]

    def play(self, loop=False, speed=1.0):
        """Play the animation in the viewer.

        Playback is driven by the viewer's animation loop and follows the wall
        clock, so it runs at the same speed no matter the frame rate (see
        `Viewer.max_fps` if it looks choppy).

        Parameters
        ----------
        loop :  bool
                Whether to start over when the end is reached.
        speed : float
                Playback speed multiplier.

        """
        if not self.duration:
            raise ValueError("Nothing to play - this animation is empty.")

        self.stop()
        start = time.perf_counter()

        def _tick():
            t = (time.perf_counter() - start) * speed
            if t >= self.duration:
                if loop:
                    t = t % self.duration
                else:
                    self.set_time(self.duration)
                    self.stop()
                    return
            self.set_time(t)

        self._playback = _tick
        self.viewer.add_animation(_tick, on_error="log")
        return self

    def stop(self):
        """Stop playback (see `Animation.play`)."""
        if self._playback is not None:
            self.viewer.remove_animation(self._playback)
            self._playback = None
        return self

    @property
    def playing(self):
        """Whether the animation is currently being played in the viewer."""
        return self._playback is not None

    def recorder(self, filename=None, **kwargs):
        """Return a `Recorder` that renders this animation frame by frame.

        Use this instead of `Animation.render` when you need to keep control
        between frames - the controls panel drives one of these off a timer to
        keep the GUI responsive while recording.

        """
        return Recorder(self, filename, **kwargs)

    def render(
        self,
        filename=None,
        *,
        fps=None,
        size=None,
        pixel_ratio=None,
        alpha=False,
        supersample=1,
        restore=True,
        progress=True,
    ):
        """Render the animation.

        Parameters
        ----------
        filename :  str | pathlib.Path, optional
                    Where to write to. The format follows the extension:

                     - ``.mp4`` (and other video formats) is written frame by
                       frame and requires ``imageio`` + ``imageio-ffmpeg``
                     - ``.gif`` requires ``imageio``
                     - a path without an extension is treated as a directory
                       and gets a numbered PNG per frame - no extra
                       dependencies, and you can encode it yourself

                    If ``None``, the frames are returned as a list of numpy
                    arrays instead.
        fps :       int, optional
                    Overrides the animation's own frame rate.
        size :      (width, height), optional
                    Size of the video. Defaults to the current canvas size.
                    Note that the canvas is resized for the recording.
        pixel_ratio : float, optional
                    Factor by which to scale the canvas size, see
                    `Viewer.screenshot`. You probably want ``1`` when passing
                    an explicit `size`.
        alpha :     bool
                    Whether to keep the background transparent. Only useful for
                    PNG sequences - videos are written without an alpha channel
                    either way.
        supersample : int
                    Anti-aliasing quality, see `Viewer.screenshot`. The cost is
                    paid per frame, hence the conservative default.
        restore :   bool
                    Whether to put the camera back where it was afterwards.
        progress :  bool | callable
                    Show a progress bar (requires ``tqdm``), or a callable
                    ``f(frame, n_frames)`` called after every frame.

        Returns
        -------
        list of numpy arrays
                    If `filename` is ``None``.
        pathlib.Path
                    The file (or directory) written to, otherwise.

        """
        rec = self.recorder(
            filename,
            fps=fps,
            size=size,
            pixel_ratio=pixel_ratio,
            alpha=alpha,
            supersample=supersample,
            restore=restore,
        )

        callback = _progress_callback(progress, rec.n_frames)
        try:
            while rec.step():
                if callback:
                    callback(rec.frame, rec.n_frames)
        except BaseException:
            rec.cancel()
            raise
        finally:
            if callback:
                callback.close()
        return rec.finish()


def _progress_callback(progress, n_frames):
    """Make a ``f(i, n)`` progress callback with a `close` method."""
    if not progress:
        return None

    if callable(progress):

        def user_callback(i, n):
            progress(i, n)

        user_callback.close = lambda: None
        return user_callback

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    bar = tqdm(total=n_frames, desc="Rendering", leave=False)

    def update(i, n):
        bar.update(i - bar.n)

    update.close = bar.close
    return update


class Recorder:
    """Renders an animation one frame at a time.

    Instantiating a recorder starts the recording (it resizes the canvas and
    opens the output file); each `step` renders and writes the next frame and
    returns ``False`` once there are none left; `finish` closes the output and
    returns it. `Animation.render` simply runs this to completion - drive it
    yourself if you need to do something between frames (e.g. keep a GUI
    responsive).

    """

    def __init__(
        self,
        animation,
        filename=None,
        *,
        fps=None,
        size=None,
        pixel_ratio=None,
        alpha=False,
        supersample=1,
        restore=True,
    ):
        if not animation.duration:
            raise ValueError("Nothing to render - this animation is empty.")

        self.animation = animation
        self.viewer = animation.viewer
        self.fps = int(fps or animation.fps)
        self.times = animation.times(self.fps)
        self.frame = 0
        self.alpha = alpha
        self.pixel_ratio = pixel_ratio
        self.supersample = supersample
        self.restore = restore
        self.done = False

        # Playback and recording would fight over the camera
        animation.stop()

        self._state_before = self.viewer.get_view()
        # Resize once for the whole recording rather than for every frame
        self._size_before = tuple(self.viewer.size) if size else None
        if size:
            self.viewer.size = size

        # Park the viewer's own draw loop. We force a draw per frame ourselves,
        # and anything driving us from an event loop (the controls panel does,
        # to stay responsive) would otherwise let the canvas paint the scene in
        # between - at up to `max_fps`, each one re-allocating the render
        # targets we just resized for the recording.
        self._max_fps_before = None
        try:
            # N.B. the scheduler hands this back as a float but only takes ints
            self._max_fps_before = int(self.viewer.max_fps)
            self.viewer.max_fps = 1
        except AttributeError:
            pass  # offscreen canvases have no scheduler to park

        try:
            self.writer = _make_writer(filename, self.fps, alpha=alpha)
        except BaseException:
            self._restore()
            raise

    @property
    def n_frames(self):
        return len(self.times)

    def step(self):
        """Render and write the next frame.

        Returns ``True`` if a frame was written and ``False`` if there was
        nothing left to do.

        """
        if self.done or self.frame >= self.n_frames:
            return False

        self.animation.set_time(self.times[self.frame])
        image = self.viewer.screenshot(
            filename=None,
            alpha=self.alpha,
            pixel_ratio=self.pixel_ratio,
            supersample=self.supersample,
        )
        self.writer.append(np.asarray(image))
        self.frame += 1
        return True

    def finish(self):
        """Close the output and return it (a path, or the list of frames)."""
        if not self.done:
            self.done = True
            result = self.writer.close()
            self._restore()
            self._result = result
        return self._result

    def cancel(self):
        """Abort the recording and discard whatever was written."""
        if not self.done:
            self.done = True
            self._result = None
            try:
                self.writer.abort()
            finally:
                self._restore()

    def _restore(self):
        if self._size_before:
            self.viewer.size = self._size_before
            self._size_before = None
        if self._max_fps_before is not None:
            self.viewer.max_fps = self._max_fps_before
            self._max_fps_before = None
        if self.restore and self._state_before is not None:
            self.viewer.set_view(self._state_before)
            self._state_before = None


def _make_writer(filename, fps, alpha=False):
    """Pick a writer for this output path."""
    if filename is None:
        return _ListWriter()

    filename = Path(filename).expanduser()
    suffix = filename.suffix.lower()

    if suffix in VIDEO_SUFFIXES:
        return _VideoWriter(filename, fps)
    if not suffix:
        return _PNGSequenceWriter(filename, alpha=alpha)
    raise ValueError(
        f"Don't know how to write '{suffix}' files. Use one of "
        f"{VIDEO_SUFFIXES}, or a path without an extension to write a PNG "
        "sequence."
    )


def _import_imageio():
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Writing videos requires imageio. Please install it with "
            "`pip install imageio imageio-ffmpeg`."
        )
    return imageio


class _FrameWriter:
    """Base class for the various ways of getting frames onto disk."""

    def append(self, frame):
        raise NotImplementedError

    def close(self):
        """Finalize and return whatever was written."""
        raise NotImplementedError

    def abort(self):
        """Give up and clean up after ourselves."""
        self.close()


class _ListWriter(_FrameWriter):
    """Keeps the frames in memory (`filename=None`)."""

    def __init__(self):
        self.frames = []

    def append(self, frame):
        self.frames.append(frame)

    def close(self):
        return self.frames

    def abort(self):
        self.frames = []


def _drop_alpha(frame):
    """Videos are RGB; alpha would either be ignored or misread as black."""
    return frame[..., :3] if frame.ndim == 3 and frame.shape[-1] == 4 else frame


def _crop_to_even(frame):
    """Trim to even dimensions - the video codecs insist on it.

    imageio would otherwise rescale the whole frame with a warning; losing a
    row of pixels is the lesser evil.

    """
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


class _VideoWriter(_FrameWriter):
    """Streams frames into a video file (mp4, gif & friends).

    Streaming rather than collecting the frames first is what keeps memory
    flat: a two-minute 1080p recording is gigabytes of frames.

    """

    def __init__(self, filename, fps):
        imageio = _import_imageio()
        self.filename = filename
        # GIF has neither macro blocks nor an aspect it would rescale - only
        # the video codecs need even dimensions
        self.crop = filename.suffix.lower() != ".gif"
        if self.crop:
            # `macro_block_size=1` stops ffmpeg from silently rescaling frames
            # whose dimensions are not a multiple of 16 - we make sure they are
            # even (see `_crop_to_even`), which is all any sane player needs
            kwargs = dict(macro_block_size=1, quality=8)
        else:
            kwargs = dict(loop=0)  # ... whereas a GIF should loop forever

        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            try:
                self.writer = imageio.get_writer(str(filename), fps=fps, **kwargs)
            except TypeError:  # a plugin that takes none of the above
                self.writer = imageio.get_writer(str(filename), fps=fps)
        except Exception as e:
            raise RuntimeError(
                f"Failed to open '{filename}' for writing: {e}\nWriting mp4 "
                "files requires ffmpeg - `pip install imageio-ffmpeg`."
            )

    def append(self, frame):
        frame = _drop_alpha(frame)
        self.writer.append_data(_crop_to_even(frame) if self.crop else frame)

    def close(self):
        self.writer.close()
        return self.filename

    def abort(self):
        self.writer.close()
        self.filename.unlink(missing_ok=True)


class _PNGSequenceWriter(_FrameWriter):
    """Writes one numbered PNG per frame into a directory."""

    def __init__(self, directory, alpha=False, prefix="frame"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.prefix = prefix
        self.written = []

    def append(self, frame):
        if not self.alpha:
            frame = _drop_alpha(frame)
        filename = self.directory / f"{self.prefix}_{len(self.written):05d}.png"
        utils.write_png(frame, filename)
        self.written.append(filename)

    def close(self):
        return self.directory

    def abort(self):
        for filename in self.written:
            filename.unlink(missing_ok=True)
        self.written = []
