# Extending `Octarine`

In [Adding Objects](objects.md) you learned how to use the built-in
object types. But what if you have want to visualize something not currently supported by `Octarine`?

You have effectively two options:

**Option 1** is to generate the `pygfx` visual (e.g. a `pygfx.Mesh` or a `pygfx.Line`)
yourself and use the generic [octarine.Viewer.add][]`()` method to add them to the scene.

The above is good enough for the odd one-off but what if you want
to use the `Octarine` viewer for your specialised data on a regular
basis? Easy: you extend `Octarine`'s functionality to include your
data!

**Option 2** is to write [Custom Converter](converters.md) functions for your data and register them with
`Octarine`. You can then package these converters as [Plugins](plugins.md)
