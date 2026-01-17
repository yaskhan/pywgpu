from typing import Optional
from pydantic import BaseModel
from enum import Enum

class QueryType(Enum):
    OCCLUSION = 0
    TIMESTAMP = 1
    PIPELINE_STATISTICS = 2

class QuerySetDescriptor(BaseModel):
    label: Optional[str] = None
    type: QueryType
    count: int
    pipeline_statistics: Optional[list] = None
