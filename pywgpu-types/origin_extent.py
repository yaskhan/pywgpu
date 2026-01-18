from pydantic import BaseModel

class Origin3d(BaseModel):
    x: int = 0
    y: int = 0
    z: int = 0

class Extent3d(BaseModel):
    width: int
    height: int = 1
    depth_or_array_layers: int = 1
