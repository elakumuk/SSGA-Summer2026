"""Append-only research log for AI/LLM usage and design decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ResearchLogEntry:
    stage: str
    llm_used: bool
    tool_or_model: str = ""
    prompt_summary: str = ""
    human_decision: str = ""
    output_used: str = ""
    risk_or_limitation: str = ""
    verification: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)


class ResearchLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: ResearchLogEntry) -> None:
        payload = asdict(entry)
        with self.log_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")

    def log_stage(
        self,
        stage: str,
        *,
        llm_used: bool = False,
        tool_or_model: str = "",
        prompt_summary: str = "",
        human_decision: str = "",
        output_used: str = "",
        risk_or_limitation: str = "",
        verification: str = "",
        **extra: Any,
    ) -> None:
        self.log(
            ResearchLogEntry(
                stage=stage,
                llm_used=llm_used,
                tool_or_model=tool_or_model,
                prompt_summary=prompt_summary,
                human_decision=human_decision,
                output_used=output_used,
                risk_or_limitation=risk_or_limitation,
                verification=verification,
                extra=extra,
            )
        )

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        entries = []
        with self.log_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
