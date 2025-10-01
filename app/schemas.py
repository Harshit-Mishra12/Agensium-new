from pydantic import BaseModel
from typing import List, Dict

class SchemaInput(BaseModel):
    dataset: List[Dict]

class SchemaOutput(BaseModel):
    fields: Dict
    row_count: int

class DedupInput(BaseModel):
    items: List[str]

class DedupOutput(BaseModel):
    original_count: int
    unique_count: int
    items: List[str]
