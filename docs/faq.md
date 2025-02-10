# FAQ

If you do have a question, head over to [Discussion](https://github.com/schlegelp/octarine/discussions) and ask away!

In the meantime, here are a couple things that may help you troubleshoot:

<details>
<summary>
When trying to open the GUI controls, I get an error message like this:

```python
ImportError:
    Importing PySide6 disabled by IPython, which has
    already imported an Incompatible QT Binding: pyqt6
```
</summary>

This message indicates that <code>IPython</code> has started a main event
loop that's incompatible with <code>PySide6</code>, the library we use to build
the controls GUI.

There are two explanations: either you didn't run the right <code>%magic</code>
command (or used the equivalent start-up option or config) to initialize
the main event loop for <code>PySide6</code>, or you have also have <code>PyQt5</code> or (more
likely) <code>PyQt6</code> installed in which case <code>IPython</code> latches on to
those instead of <code>PySide6</code>.

Bottom line: make sure you only have <code>PySide6</code> installed as window
manager and then run this %magic command at start-up:

```python
%gui qt6
```

</details>


<details>
<summary>
My Octarine-based app/script uses a lot of CPU.
</summary>

By default, <code>octarine</code> will re-render the scene at every frame even if
nothing has changed - i.e. the camera hasn't moved and objects haven't changed.

You can change the render trigger to something a bit more resource-friendly:

<ul>
  <li> <code>continuous</code> is the default, greedy mode </li>
  <li> <code>active_window</code> pauses the rendering when the window is not active </li>
  <li> <code>reactive</code> tries to only re-render the scene if something has changed </li>
<ul>


```python
import octarine as oc

v = oc.Viewer()

# Set to render reactively
v.render_trigger = "reactive"

# Set to render only when window is active
v.render_trigger = "active_window"
```

We may change the default render trigger in the future.

</details>



