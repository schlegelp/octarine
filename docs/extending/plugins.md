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

### Entry Point

Each plugin must register as a Plugin via a setuptools entry_points. That's done in `setup.py`
via the `entry_point` parameter.


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

### Registration

Next, we need to define our registration function. We will do so directly in the `__init__.py`
but you can also define it elsewhere and import it at top-level of the namespace.

```python
# __init__.py

import octarine as oc

# Import converters (see the Custom Converters Tutorial for details)
from .converters import convert_point3d, add_point3d


def register_plugin():
    """Register Point3d converter and method with Octarine."""
    oc.register_converter(Point3d, convert_point3d)
    oc.Viewer.add_point3d = add_point3d

```

And that's it! Publish the plugin to PyPI and everyone who installs it will be able to
use `Point3d` with `Octarine`.

### Concluding Remarks

Consider delaying imports for large (read: slow to import) libraries until you actually need them by
putting the `import` statements inside the converter functions. That way `Octarine` itself will remain
quick to import.

See [`octarine-navis-plugin`](https://github.com/navis-org/octarine-navis-plugin) for an example of
an `Octarine` plugin in the wild.
