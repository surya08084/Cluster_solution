from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PipelineStep:
    id: str
    params: Optional[Dict] = None


@dataclass
class PipelineSpec:
    dataset_id: str
    steps: List[PipelineStep]