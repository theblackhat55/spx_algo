from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest_artifacts(model_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return {"ok": False, "reason": "manifest_missing"}

    artifacts = manifest.get("artifacts", {})
    mismatches = []
    missing = []

    for name, meta in artifacts.items():
        p = model_dir / name
        if not p.exists():
            missing.append(name)
            continue
        expected = meta.get("sha256")
        actual = sha256_file(p)
        if expected and actual and expected != actual:
            mismatches.append({"artifact": name, "expected": expected, "actual": actual})

    return {
        "ok": len(missing) == 0 and len(mismatches) == 0,
        "reason": "ok" if len(missing) == 0 and len(mismatches) == 0 else "artifact_mismatch",
        "missing": missing,
        "mismatches": mismatches,
        "manifest": manifest,
    }
