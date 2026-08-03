"""Small JSON store with physically separated private and shared planes."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .schema import SharedRule, TenantEpisode


class JsonMemoryStore:
    """Persistence adapter designed to make the isolation boundary inspectable."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.tenants_dir = self.root / "tenants"
        self.shared_dir = self.root / "shared"
        self.tenants_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tenant_key(tenant_id: str) -> str:
        return sha256(tenant_id.encode()).hexdigest()[:24]

    def _tenant_path(self, tenant_id: str) -> Path:
        return self.tenants_dir / f"{self.tenant_key(tenant_id)}.json"

    @staticmethod
    def _read(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def append_private_episode(self, episode: TenantEpisode) -> None:
        path = self._tenant_path(episode.tenant_id)
        payload = self._read(path, {"tenant_id": episode.tenant_id, "episodes": []})
        if payload.get("tenant_id") != episode.tenant_id:
            raise ValueError("tenant store identity mismatch")
        episodes = [item for item in payload["episodes"] if item["episode_id"] != episode.episode_id]
        episodes.append(episode.to_private_dict())
        payload["episodes"] = episodes
        self._write(path, payload)

    def load_private_episodes(self, tenant_id: str) -> list[TenantEpisode]:
        payload = self._read(self._tenant_path(tenant_id), {"episodes": []})
        if payload.get("tenant_id", tenant_id) != tenant_id:
            raise ValueError("tenant store identity mismatch")
        return [TenantEpisode.from_private_dict(item) for item in payload["episodes"]]

    def load_candidate_ledger(self) -> dict[str, Any]:
        return self._read(self.shared_dir / "candidates.json", {"candidates": {}})

    def save_candidate_ledger(self, payload: dict[str, Any]) -> None:
        self._write(self.shared_dir / "candidates.json", payload)

    def load_shared_rules(self) -> list[SharedRule]:
        payload = self._read(self.shared_dir / "rules.json", {"rules": []})
        return [SharedRule.from_dict(item) for item in payload["rules"]]

    def save_shared_rules(self, rules: list[SharedRule]) -> None:
        self._write(self.shared_dir / "rules.json", {"rules": [rule.to_dict() for rule in rules]})
