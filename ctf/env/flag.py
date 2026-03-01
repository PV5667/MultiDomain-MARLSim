from dataclasses import dataclass

@dataclass
class Flag:
    x: int
    y: int
    disposition: float = 0.0