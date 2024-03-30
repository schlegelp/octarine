from ipywidgets import VBox, Widget
from IPython.display import display

try:
    from sidecar import Sidecar
except ImportError:
    Sidecar = None


class JupyterOutput(VBox):
    """Wrap Viewer (and potentially a toolbar) in Vbox for display."""

    def __init__(
        self,
        viewer,
        #make_toolbar: bool,
        use_sidecar: bool,
        sidecar_kwargs: dict
    ):
        """

        Parameters
        ----------
        viewer  :       Viewer
                        The Viewer instance to display.
        use_sidecar :   bool
                        Whether to use Sidecar for display. Will throw an
                        error if Sidecar is not installed.
        sidecar_kwargs: dict
                        optional kwargs passed to Sidecar

        """
        self.viewer = viewer
        self.sidecar = None

        self.use_sidecar = use_sidecar

        if self.use_sidecar and Sidecar is None:
            raise ImportError("Sidecar extention is not installed: `pip install sidecar`\n"
                              "Note that you will have to restart Jupyter and deep-reload "
                              "the page after installation.")

        self.output = (viewer.canvas, )

        # if not make_toolbar:  # just stack canvas and the additional widgets, if any
        #     self.output = (frame.canvas, *add_widgets)

        # if make_toolbar:  # make toolbar and stack canvas, toolbar, add_widgets
        #     self.toolbar = IpywidgetToolBar(frame)
        #     self.output = (frame.canvas, self.toolbar, *add_widgets)

        if use_sidecar:  # instantiate sidecar if desired
            self.sidecar = Sidecar(**sidecar_kwargs)

        # stack all objects in the VBox
        super().__init__(self.output)

    def _repr_mimebundle_(self, *args, **kwargs):
        """
        This is what jupyter hook into when this output context instance is returned at the end of a cell.
        """
        if self.use_sidecar:
            with self.sidecar:
                return display(VBox(self.output))
        else:
            # just display VBox contents in cell output
            return super()._repr_mimebundle_(*args, **kwargs)

    def close(self, close_viewer=True):
        """Closes the output context, cleanup all the stuff"""
        self._is_closed = True  #  we need this to avoid recursion in close

        if close_viewer:
            self.viewer.close()

        # if self.toolbar is not None:
        #     self.toolbar.close()

        if self.sidecar is not None:
            self.sidecar.close()

        super().close()  # ipywidget VBox cleanup