from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .texture import Texture


class SurfaceTexture:
    """
    Texture acquired from a Surface.

    Wraps a :class:`Texture` that can be rendered to and then presented.
    """

    def __init__(self, inner: Any, texture: "Texture", suboptimal: bool) -> None:
        self._inner = inner
        self.texture = texture
        self.suboptimal = suboptimal

    def present(self) -> None:
        """Presents the texture to the surface."""
        if hasattr(self._inner, "present"):
            self._inner.present()
        else:
            raise NotImplementedError("Backend does not support present")
