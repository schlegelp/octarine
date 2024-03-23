![cocoa](docs/_static/octarine_logo_banner.png)
<p align="center">
<i>
Octarine is the eighth color of the Discworld's spectrum, which is described as the color of magic itself. Only wizards and cats can see it.
</i>
</p>

# Octarine
This project is intended to be for 3D what [`fastplotlib`](https://github.com/fastplotlib/fastplotlib) is for 2D:
a high-performance, easy-to-use 3D viewer. `Octarine` is build on top off the excellent
[`pygfx`](https://github.com/pygfx/pygfx) rendering engine which does most of the heavy lifting - we're simply
abstracting away some of the boiler plate code for you.

## Installation 

```bash 
pip install octarine
```

## Quickstart 

```python
# Import octarine
from octarine import Viewer

# Create a Viewer instances 
v = Viewer()


```