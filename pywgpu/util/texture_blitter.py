# Texture blitting utilities

class TextureBlitter:
    """
    Helper for blitting textures (copying/scaling/format conversion).
    """
    def __init__(self, device):
        self.device = device

    def blit(self, source, destination):
        if hasattr(self, '_inner') and hasattr(self._inner, 'blit'):
            return self._inner.blit(source, destination)
        else:
            raise NotImplementedError("Backend does not support blit")
