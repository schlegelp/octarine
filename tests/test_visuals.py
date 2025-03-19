import time
import pytest
import octarine as oc

import trimesh as tm
import numpy as np

# Set random state
np.random.seed(0)


@pytest.fixture
def mesh():
    return tm.creation.icosphere()


@pytest.fixture()
def line_single():
    return np.random.rand(10, 3)


@pytest.fixture()
def line_stack():
    return [np.random.rand(i, 3) for i in np.random.randint(2, 10, 10)]


@pytest.fixture()
def points():
    return np.random.rand(10, 3)


@pytest.fixture()
def points_colors():
    return np.random.rand(10, 3), np.random.rand(10, 3)


def test_adding_generic_objects(mesh, line_single, line_stack, points, points_colors):
    v = oc.Viewer(offscreen=True)

    # Test adding objects generically
    for ob in [mesh, line_single, line_stack, points, points_colors]:
        v.add(ob)
        v.clear()

    v.close()


@pytest.mark.parametrize("color", [None, "red", np.random.rand(3)])
def test_adding_mesh(mesh, color):
    v = oc.Viewer(offscreen=True)
    v.add_mesh(mesh, color=color)
    v.close()


def test_showing_messsage():
    v = oc.Viewer(offscreen=True)
    v.show_message("test", color="red")
    v.show_message(None, color="red")
    v.show_message("test", color="red", duration=1)
    time.sleep(2)
    v.close()
