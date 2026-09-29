"""Versioned, content-hashed agent configuration with an atomic active pointer.

A version is an immutable JSON file under configs/versions/. The active
version is named by configs/active.json, which is replaced atomically
(write temp file, os.replace), never edited in place and never a symlink
(Windows and deploy targets differ). Each research run reads the active
version once at its start, so an in-flight request keeps one consistent
config even if a promotion lands mid-run.

Only the agent-editable surface lives here (synthesis instructions for
now). Evaluation rubrics live elsewhere, so a candidate cannot change how
it is graded.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(
    os.getenv("CONFIG_ROOT", Path(__file__).resolve().parent.parent / "configs")
)
VERSIONS = ROOT / "versions"
ACTIVE = ROOT / "active.json"
HISTORY = ROOT / "promotions.jsonl"

BASELINE_INSTRUCTIONS = (
    "Cite inline with [N]. Cite a source for a claim only if its Passage supports that claim.\n"
    "If no passage supports something, say it is uncertain instead of citing."
)


class ConfigVersion(BaseModel):
    version_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    synthesis_instructions: str = Field(min_length=10, max_length=4000)
    parent: str | None = None
    notes: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.synthesis_instructions.encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        return f"{self.version_id}@{self.content_hash[:8]}"


class StalePromotion(RuntimeError):
    pass


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save(version: ConfigVersion) -> ConfigVersion:
    """Versions are immutable: re-saving the same id with different content fails."""
    path = VERSIONS / f"{version.version_id}.json"
    if path.exists():
        existing = get(version.version_id)
        if existing.content_hash != version.content_hash:
            raise ValueError(
                f"version {version.version_id} exists with different content"
            )
        return existing
    _atomic_write(path, version.model_dump_json(indent=2))
    return version


def get(version_id: str) -> ConfigVersion:
    path = VERSIONS / f"{version_id}.json"
    if not path.exists():
        raise KeyError(f"unknown version {version_id}")
    return ConfigVersion.model_validate_json(path.read_text(encoding="utf-8"))


def list_versions() -> list[ConfigVersion]:
    return sorted(
        (get(p.stem) for p in VERSIONS.glob("*.json")), key=lambda v: v.created_at
    )


def ensure_baseline() -> ConfigVersion:
    base = save(ConfigVersion(version_id="v1-baseline", synthesis_instructions=BASELINE_INSTRUCTIONS,
                              notes="M0 evidence-bound synthesis prompt"))  # fmt: skip
    if not ACTIVE.exists():
        _atomic_write(
            ACTIVE,
            json.dumps(
                {"version_id": base.version_id, "content_hash": base.content_hash}
            ),
        )
    return base


def active() -> ConfigVersion:
    """Read once per run. Verifies the pointer still matches the file's content."""
    if not ACTIVE.exists():
        return ensure_baseline()
    ptr = json.loads(ACTIVE.read_text(encoding="utf-8"))
    v = get(ptr["version_id"])
    if v.content_hash != ptr["content_hash"]:
        raise RuntimeError(f"active pointer hash mismatch for {v.version_id}")
    return v


def set_active(version_id: str, *, expected_active: str, actor: str, reason: str,
               evidence: dict | None = None, action: str = "promote") -> dict:  # fmt: skip
    """Compare-and-swap on the active pointer. Refuses if the active version
    is not the one the caller evaluated against (stale promotion)."""
    current = active()
    if current.version_id != expected_active:
        raise StalePromotion(
            f"active is {current.version_id}, caller expected {expected_active}; re-evaluate first"
        )
    target = get(version_id)
    _atomic_write(
        ACTIVE,
        json.dumps(
            {"version_id": target.version_id, "content_hash": target.content_hash}
        ),
    )
    event = {
        "at": datetime.now(timezone.utc).isoformat(), "action": action, "actor": actor,
        "from": current.label, "to": target.label, "reason": reason, "evidence": evidence or {},
    }  # fmt: skip
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event


def history() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [
        json.loads(l)
        for l in HISTORY.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
