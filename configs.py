from dataclasses import dataclass, field
from typing import List


@dataclass
class DoLaConfig:
    mode: str = 'dola'
    candidate_premature_layers: List[int] = field(default_factory=lambda: [])
    mature_layer: int = -1
    post_softmax: bool = True


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 256
    relative_top: float = 0.1
    deterministic: bool = False