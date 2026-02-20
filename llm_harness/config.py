from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    dataset: Path
    provider: str
    model: str


@dataclass(slots=True)
class ReportConfig:
    results: Path
