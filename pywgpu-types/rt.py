from typing import Optional, List, Any
from pydantic import BaseModel

class BlasDescriptor(BaseModel):
    label: Optional[str] = None
    # Add actual BLAS fields (geometries etc) later

class TlasDescriptor(BaseModel):
    label: Optional[str] = None
    max_instances: int
