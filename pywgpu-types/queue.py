from typing import Optional, Any, Union, List
from pydantic import BaseModel
from .origin_extent import Origin3d


class ImageDataLayout(BaseModel):
    offset: int = 0
    bytes_per_row: Optional[int] = None
    rows_per_image: Optional[int] = None


class ImageCopyBuffer(BaseModel):
    buffer: Any  # Buffer
    layout: ImageDataLayout


class ImageCopyTexture(BaseModel):
    texture: Any  # Texture
    mip_level: int = 0
    origin: Union[Origin3d, List[int], Any] = Origin3d()
    aspect: str = "all"
