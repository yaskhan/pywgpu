from typing import Optional, Any


class TextureView:
    """
    Handle to a texture view.

    Created with :meth:`Texture.create_view`.
    """

    def __init__(self, inner: Any, texture: Any) -> None:
        self._inner = inner
        self._texture = texture

    @property
    def texture(self) -> Any:
        """The texture this view was created from."""
        return self._texture
