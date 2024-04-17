# Plugins

In the [Custom Converters](converters.md) tutorial you learned how to generate and
register new converter functions. Here we will show you how to package converters
into installable plugins for `Octarine`.

By convention, `Octarine` plugins are called `octarine-{name}-plugin`. Here, we will
generate an example plugin for the `Point3d` dummy class we used to illustrate converters.

### Project layout

Just to orient you, this is the structure of our example plugins - a bog-standard Python project:

```
.
├── octarine_point3d_plugin
│   ├── __init__.py
│   └── converters.py
├── setup.py
├── requirements.txt
└── README.md
```

### Adding functionality

First we need to define a registration function to add new functionality to `Octarine`.
Here, we will do so directly in the `__init__.py` but you can also define it elsewhere
and import it at top-level of the namespace.

```python
# __init__.py

import octarine as oc

# Import converters (see the "Custom Converters" Tutorial for details)
from .converters import convert_point3d, add_point3d


def register_plugin():
    """Register Point3d converter and method with Octarine."""
    # Register the new converter
    oc.register_converter(Point3d, convert_point3d)

    # Monkey-patch the Viewer class
    oc.Viewer.add_point3d = add_point3d

```

!!! note

    Consider delaying imports for large (read: slow to import) libraries until
    you actually need them. For example by putting the `import` statements inside
    the converter functions. That way `Octarine` itself will not be bogged down by
    the plugin and remain quick to import.

### Entry Point

Next we need to make `Octarine` aware of the new plugin. For that we are using `setuptools`'
entry points and is as simple as modifying the call to `setup()` in the `setup.py`:


```python
# setup.py
from setuptools import setup

...

setup(
    name="octarine-point3d-plugin",
    ...
    entry_points={
        "octarine.plugins": [
            "octarine_point3d_plugin = octarine_point3d_plugin:register_plugin",
        ]
    },
)
```

The important thing here is that we define an entry point for `octarine.plugins` which makes
the plugin discoverable by `Octarine`. The entry point itself then tells us which
module to import (`octarine_point3d_plugin`) and which function (`octarine_point3d_plugin:register_plugin`)
to run in order to teach `Octarine` how to deal with new data. It is not enforced but
by convention that function should be called `register_plugin()`.

And that's it! Publish the plugin to PyPI and everyone who installs it will be able to
use `Point3d` with `Octarine`.

### Concluding Remarks

The plugin system is currently rather simple but you can still do a lot with it - in particular
if you use [monkey patching](https://en.wikipedia.org/wiki/Monkey_patch) to modify the behavior
of the `Viewer` class.

See [`octarine-navis-plugin`](https://github.com/navis-org/octarine-navis-plugin) for an example of
an `Octarine` plugin in the wild.
