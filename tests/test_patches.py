import os
import sys
import time
import asyncio
import threading

import octarine  # noqa: F401 - applies the patches

from rendercanvas.base import BaseCanvasGroup, BaseRenderCanvas
from rendercanvas.raw import RawLoop
from rendercanvas.utils import asyncs


class FooCanvasGroup(BaseCanvasGroup):
    pass


class FooCanvas(BaseRenderCanvas):
    _rc_canvas_group = FooCanvasGroup(None)

    def __init__(self):
        super().__init__()
        self._final_canvas_init()


def test_sleep_when_another_loop_owns_the_asyncgen_hooks():
    # Regression test for rendercanvas#211, adapted from rendercanvas#250.
    #
    # In IPython with '%gui qt', the Qt loop runs inside asyncio's input-hook,
    # so the asyncgen hooks are asyncio's while rendercanvas' asyncadapter
    # steps the tasks. sleep() must still use the adapter, otherwise it awaits
    # an asyncio future that the adapter cannot handle.
    import wgpu._async as wgpu_async

    leeway = 0.20 if os.getenv("CI") else 0

    # A (non-running) asyncio loop to steal the asyncgen hooks with
    asyncio_loop = asyncio.new_event_loop()

    libs = []
    funcs = []
    wgpu_libs = []
    wgpu_funcs = []
    times = []

    async def coro():
        sys.set_asyncgen_hooks(
            firstiter=asyncio_loop._asyncgen_firstiter_hook,
            finalizer=asyncio_loop._asyncgen_finalizer_hook,
        )
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

    # Failsafe: without the fix the coro never resumes
    failsafe = threading.Timer(2.0, loop.call_soon_threadsafe, [loop.stop])
    failsafe.start()

    prev_hooks = sys.get_asyncgen_hooks()
    try:
        loop.run()
    finally:
        failsafe.cancel()
        sys.set_asyncgen_hooks(*prev_hooks)
        asyncio_loop.close()

    assert libs == ["rendercanvas.utils.asyncadapter"]
    assert funcs == [loop.call_soon_threadsafe]
    assert wgpu_libs == ["rendercanvas.utils.asyncadapter"]
    assert wgpu_funcs == [loop.call_soon_threadsafe]

    assert len(times) == 2
    assert 0.04 < times[1] - times[0] < 0.08 + leeway
