class Tracker:
    """
    Tracks usage of resources (buffers, textures).
    """
    def __init__(self) -> None:
        self.resources = []

    def track(self, resource) -> None:
        self.resources.append(resource)
