"""Integrity manifest over a release tree."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wireope.utils.atomic import atomic_write_json, iter_files, sha256_file
from wireope.version import RELEASE_SLUG

SKIP_DIRS: tuple[str, ...] = (
    ".git",
    "artefacts",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
)


def manifest_entries(root: Path, exclude: Sequence[Path] = ()) -> list[dict[str, str]]:
    resolved_exclude = {path.resolve() for path in exclude}
    entries: list[dict[str, str]] = []
    for path in iter_files(root, SKIP_DIRS):
        if path.resolve() in resolved_exclude:
            continue
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    entries.sort(key=lambda item: item["path"])
    return entries


def manifest_digest(entries: Sequence[dict[str, str]]) -> str:
    import hashlib

    joined = "\n".join(f"{item['path']}:{item['sha256']}" for item in entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_manifest(root: Path, target: Path, exclude: Sequence[Path] = ()) -> dict[str, Any]:
    excluded = [*exclude, target]
    entries = manifest_entries(root, excluded)
    payload: dict[str, Any] = {
        "root": RELEASE_SLUG,
        "algorithm": "sha256",
        "excluded": sorted({path.resolve().name for path in excluded}),
        "file_count": len(entries),
        "manifest_digest": manifest_digest(entries),
        "files": entries,
    }
    return payload


def write_integrity_manifest(root: Path, target: Path) -> dict[str, Any]:
    payload = build_manifest(root, target)
    atomic_write_json(target, payload)
    return payload


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(manifest_path.read_text())
    recorded = {item["path"]: item["sha256"] for item in payload["files"]}
    live: dict[str, str] = {}
    for item in manifest_entries(root, [manifest_path]):
        live[item["path"]] = item["sha256"]
    missing = sorted(set(recorded) - set(live))
    added = sorted(set(live) - set(recorded))
    changed = sorted(path for path in set(recorded) & set(live) if recorded[path] != live[path])
    return {
        "intact": not missing and not added and not changed,
        "missing": missing,
        "added": added,
        "changed": changed,
        "manifest_digest": manifest_digest(
            sorted(
                ({"path": path, "sha256": digest} for path, digest in live.items()),
                key=lambda item: item["path"],
            )
        ),
        "recorded_digest": payload["manifest_digest"],
    }
