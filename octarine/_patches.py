"""Monkey-patches for upstream issues that are fixed but not yet released.

Each patch checks whether the upstream fix is already present and does
nothing if so, so these can stay in place until we bump the minimum versions.
"""

import sys
import logging
import threading
import functools

from contextlib import contextmanager


logger = logging.getLogger("octarine")


def patch_rendercanvas_asyncadapter():
    """Port of https://github.com/pygfx/rendercanvas/pull/250 (fixes rendercanvas#211).

    Since rendercanvas 2.5.0, the running async lib is detected via the global
    asyncgen hooks. In IPython with ``%gui qt``, the Qt loop runs inside the
    input-hook of prompt_toolkit's asyncio loop, so the hooks are asyncio's while
    rendercanvas' asyncadapter tasks are stepped from a Qt timer. ``sleep()`` then
    awaits an asyncio future, which the adapter rejects ("Incompatible awaitable
    result <Future pending>"), cancelling the scheduler and freezing the canvas.

    The fix: mark the adapter task being stepped in a thread-local, and have the
    detection check that before looking at the hooks.

    TODO: remove once we require a rendercanvas release that includes #250.
    """
    from rendercanvas.utils import asyncadapter, asyncs
    from rendercanvas.core import coreutils, loop as loop_module

    if hasattr(asyncadapter, "get_current_task"):
        return  # fixed upstream

    class _ThreadLocal(threading.local):
        current_task = None

    _thread_local = _ThreadLocal()

    def get_current_task():
        """Get the task that is currently being stepped in this thread, or None."""
        return _thread_local.current_task

    @contextmanager
    def detached():
        """Hide the task that is currently being stepped."""
        old_task = _thread_local.current_task
        _thread_local.current_task = None
        try:
            yield
        finally:
            _thread_local.current_task = old_task

    # Mark the task while it is being stepped. Tasks step via a lambda that
    # calls task.step(), which resolves on the class, so existing tasks pick
    # this up too.
    original_step = asyncadapter.Task.step

    @functools.wraps(original_step)
    def step(self):
        old_task = _thread_local.current_task
        _thread_local.current_task = self
        try:
            return original_step(self)
        finally:
            _thread_local.current_task = old_task

    asyncadapter.Task.step = step

    def _task_call_soon_threadsafe(task):
        # The PR passes the loop's call_soon_threadsafe to the Task. Rather than
        # patching BaseLoop._rc_add_task, recover it from the bound
        # _rc_call_later method that the loop already passes in.
        loop = getattr(task._call_later, "__self__", None)
        return getattr(loop, "call_soon_threadsafe", None)

    original_detect_lib = asyncs.detect_current_async_lib
    original_detect_cst = asyncs.detect_current_call_soon_threadsafe

    @functools.wraps(original_detect_lib)
    def detect_current_async_lib():
        if _thread_local.current_task is not None:
            return "rendercanvas.utils.asyncadapter"
        return original_detect_lib()

    @functools.wraps(original_detect_cst)
    def detect_current_call_soon_threadsafe():
        task = _thread_local.current_task
        if task is not None:
            func = _task_call_soon_threadsafe(task)
            if func is not None:
                return func
        return original_detect_cst()

    # sleep(), precise_sleep() and Event look these up as module globals at
    # call time, so replacing the module attributes reaches them.
    asyncs.detect_current_async_lib = detect_current_async_lib
    asyncs.detect_current_call_soon_threadsafe = detect_current_call_soon_threadsafe

    # Closing async generators happens synchronously from inside a task (when
    # the loop stops), but must not count as part of it; otherwise sleep() and
    # Event() yield to the adapter and the agen cannot be closed.
    original_close_agen = coreutils.close_agen

    @functools.wraps(original_close_agen)
    def close_agen(agen):
        with detached():
            return original_close_agen(agen)

    coreutils.close_agen = close_agen
    loop_module.close_agen = close_agen  # imported by name

    # Expose the same API as the upstream fix, so that e.g. wgpu versions that
    # know about it (see below) find it.
    asyncadapter.get_current_task = get_current_task
    asyncadapter.detached = detached

    # wgpu has its own copy of the hooks-only detection, used for GPUPromise
    # (awaiting, .then()). Newer wgpu versions consult get_current_task()
    # themselves; for older ones we patch the detection in the same way.
    wgpu_async = sys.modules.get("wgpu._async")
    if wgpu_async is None:
        try:
            import wgpu._async as wgpu_async
        except ImportError:
            wgpu_async = None
    if wgpu_async is not None and not hasattr(
        wgpu_async, "_get_current_asyncadapter_task"
    ):
        wgpu_detect_lib = wgpu_async.detect_current_async_lib
        wgpu_detect_cst = wgpu_async.detect_current_call_soon_threadsafe

        @functools.wraps(wgpu_detect_lib)
        def wgpu_detect_current_async_lib():
            if _thread_local.current_task is not None:
                return "rendercanvas.utils.asyncadapter"
            return wgpu_detect_lib()

        @functools.wraps(wgpu_detect_cst)
        def wgpu_detect_current_call_soon_threadsafe():
            task = _thread_local.current_task
            if task is not None:
                func = _task_call_soon_threadsafe(task)
                if func is not None:
                    return func
            return wgpu_detect_cst()

        wgpu_async.detect_current_async_lib = wgpu_detect_current_async_lib
        wgpu_async.detect_current_call_soon_threadsafe = (
            wgpu_detect_current_call_soon_threadsafe
        )

    logger.debug("Patched rendercanvas asyncadapter detection (rendercanvas#250).")


def apply_all():
    for patch in (patch_rendercanvas_asyncadapter,):
        try:
            patch()
        except Exception as e:  # never let a patch break importing octarine
            logger.warning(f"Failed to apply {patch.__name__}: {e}")
