# Installation

`Octarine` is published as a [Python package] and can be installed with
`pip`, ideally by using a [virtual environment]. Open up a terminal and install
`Octarine` with:

=== "Full Install"

    The full install should set you up for using `Octarine` in
    both shell (`IPython`) and Jupyter:

    ``` sh
    pip install "octarine3d[all]"
    ```

    It includes `PySide6`, `jupyter_rfb` and `sidecar` as additional dependencies.


=== "Minimal"

    If you opt for the minimal install, you will still have to
    install at least one [Window Manager](#window-managers):

    ``` sh
    pip install octarine3d
    ```

=== "Dev"

    To install the latest version from Github:

    ``` sh
    pip install git+https://github.com/schlegelp/octarine.git
    ```

!!! tip

    If you don't have prior experience with Python, check out
    [Using Python's pip to Manage Your Projects' Dependencies], which is a
    really good introduction on the mechanics of Python package management and
    helps you troubleshoot if you run into errors.

  [Python package]: https://pypi.org/project/octarine3d/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [Markdown]: https://python-markdown.github.io/
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/


### Window Managers

`pygfx` (and hence `Octarine`) requires a Window manager to work. If you
have opted for the `[all]` install option, you should already have
`PySide6` and `jupyter-rbf` and you don't need to worry about this.

If instead you decided to pick your a window manager on your own, here are
the ones supported by [`wgpu-py`](https://github.com/pygfx/wgpu-py):

- qt: `PySide6`, `PyQt6`, `PySide2`, `PyQt5` all work but I recommend `PySide6` (see below)
- `glfw`: a lightweight GUI for the desktop
- `jupyter-rfb`: only needed if you plan on using `Octarine` in Jupyter

=== "PySide6"

    ``` sh
    pip install PySide6
    ```

=== "PyQt6"

    ``` sh
    pip install PyQt6
    ```

=== "PyQt5"

    ``` sh
    pip install PyQt5
    ```

=== "glfw"

    ``` sh
    pip install glfw
    ```

=== "jupyter-rbf"

    ``` sh
    pip install jupyter-rfb
    ```

Please note that at this point, `Octarine`'s controls panel requires `PySide6`. So if you want GUI controls you have to use `PySide6`.

## What next?

<div class="grid cards" markdown>

-   :material-eye-arrow-right:{ .lg .middle } __Viewer Basics__

    ---

    Learn about using `Octarine` in different environments.

    [:octicons-arrow-right-24: The Basics](intro.md)

-   :material-cube:{ .lg .middle } __Objects__

    ---

    Check out the guide on different object types.

    [:octicons-arrow-right-24: Adding Objects](objects.md)

-   :material-format-font:{ .lg .middle } __Animations__

    ---

    Add movement to the viewer.

    [:octicons-arrow-right-24: Animations](animations.md)

-   :material-camera-control:{ .lg .middle } __Control__

    ---

    Learn how to control the viewer, adjust colors, etc.

    [:octicons-arrow-right-24: Controls](controls.md)

</div>