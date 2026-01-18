from typing import Optional, List
from pydantic import BaseModel
from enum import Enum

class CooperativeScalarType(Enum):
    F32 = 'f32'
    F16 = 'f16'
    I32 = 'i32'
    U32 = 'u32'

class CooperativeMatrixProperties(BaseModel):
    m_size: int
    n_size: int
    k_size: int
    ab_type: CooperativeScalarType
    cr_type: CooperativeScalarType
    saturating_accumulation: bool
