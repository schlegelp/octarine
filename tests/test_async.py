import os
import sys
import time
import asyncio
import threading

from rendercanvas.base import BaseCanvasGroup, BaseRenderCanvas
from rendercanvas.raw import RawLoop
from rendercanvas.utils import asyncs

import wgpu._async as wgpu_async


class FooCanvasGroup(BaseCanvasGroup):
    pass


class FooCanvas(BaseRenderCanvas):
    _rc_canvas_group = FooCanvasGroup(None)

    def __init__(self):
        super().__init__()
        self._final_canvas_init()


def test_sleep_when_another_loop_owns_the_asyncgen_hooks():
    # Regression test for rendercanvas#211, which stopped us from releasing for
    # a while. In IPython with '%gui qt' the Qt loop runs inside asyncio's
    # input-hook, so asyncio owns the asyncgen hooks while rendercanvas steps
    # its tasks from a Qt timer. rendercanvas detects the running async lib
    # from those hooks, so it used to conclude "asyncio", and sleep() then
    # awaited an asyncio future that the asyncadapter cannot handle, killing
    # the scheduler and freezing the canvas.
    #
    # rendercanvas 3.0.1 fixes this by installing its own hooks for the
    # duration of every callback. We keep the test to catch a regression, and
    # because wgpu has its own copy of the detection.

    leeway = 0.20 if os.getenv("CI") else 0

    # A (non-running) asyncio loop to hold the asyncgen hooks with, standing in
    # for the one that IPython is running
    asyncio_loop = asyncio.new_event_loop()

    libs = []
    funcs = []
    wgpu_libs = []
    wgpu_funcs = []
    times = []

    def steal_the_hooks():
        sys.set_asyncgen_hooks(
            firstiter=asyncio_loop._asyncgen_firstiter_hook,
            finalizer=asyncio_loop._asyncgen_finalizer_hook,
        )

    async def coro():
        # Take the hooks again while the loop is running, from outside this
        # task, the way the IPython loop does whenever it resumes
        loop.call_soon(steal_the_hooks)
        await asyncs.sleep(0.01)

        libs.append(asyncs.detect_current_async_lib())
        funcs.append(asyncs.detect_current_call_soon_threadsafe())
        wgpu_libs.append(wgpu_async.detect_current_async_lib())
        wgpu_funcs.append(wgpu_async.GPUPromise("test", None)._call_soon_threadsafe)
        times.append(time.perf_counter())
        await asyncs.sleep(0.05)
        times.append(time.perf_counter())
        loop.stop()

    loop = RawLoop()
    FooCanvas.select_loop(loop)
    _canvas = FooCanvas()
    loop.add_task(coro)

    # Failsafe: when this breaks, the coro never resumes
    failsafe = threading.Timer(2.0, loop.call_soon_threadsafe, [loop.stop])
    failsafe.start()

    # Hand the hooks to asyncio before we start, like the loop that is already
    # running in IPython would have
    prev_hooks = sys.get_asyncgen_hooks()
    steal_the_hooks()
    try:
        loop.run()
    finally:
        failsafe.cancel()
        sys.set_asyncgen_hooks(*prev_hooks)
        asyncio_loop.close()

    # Our own loop is stepping the task, so it is the current async lib
    assert libs == ["rendercanvas.utils.asyncadapter"]
    assert funcs == [loop.call_soon_threadsafe]
    assert wgpu_libs == ["rendercanvas.utils.asyncadapter"]
    assert wgpu_funcs == [loop.call_soon_threadsafe]

    # And the sleep must actually have worked
    assert len(times) == 2
    assert 0.04 < times[1] - times[0] < 0.08 + leeway
