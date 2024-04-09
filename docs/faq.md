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

This message indicates that `IPython` has started a main event
loop that's incompatible with `PySide6`, the library we use to build
the controls GUI.

There are two explanations: either you didn't run the right %magic
command (or used the equivalent start-up option or config) to initialize
the main event loop for `PySide6`, or you have also have `PyQt5` or (more
likely) `PyQt6` installed in which case `IPython` latches on to
those instead of `PySide6`.

Bottom line: make sure you only have `PySide6` installed as window
manager and then run this %magic command at start-up:

```python
%gui qt6
```

</details>



