import importlib

from importlib_metadata import entry_points

registered_plugins = []


def register_plugins():
    """Register a plugin."""
    # Find all plugins that defined an entry point
    discovered_plugins = entry_points(group="octarine.plugins")

    # Go over each of the plugins
    for plugin in discovered_plugins:
        # Import the module
        try:
            module = importlib.import_module(plugin.module)
        except BaseException as e:
            print(f"Error importing plugin {plugin.name}: {e}")
            continue

        # Get the function to register the plugin
        register_func = getattr(module, plugin.value.split(":")[-1], None)

        # If the function is not found, print an error
        if register_func is None:
            print(
                f"Registration function {plugin.value.split(':')[-1]} not found for plugin {plugin.name}."
            )
            continue

        # Otherwise, register the plugin
        try:
            register_func()
        except BaseException as e:
            print(f"Error registering plugin {plugin.name}: {e}")

        # Add the plugin to the list of registered plugins
        registered_plugins.append(plugin.name)


register_plugins()
