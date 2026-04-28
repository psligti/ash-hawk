from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_eval_manifest(
    *,
    workdir: Path,
    run_id: str,
    scenario_path: str | None,
    scenario_required_files: list[str],
    repetitions: int,
) -> tuple[Path | None, str | None]:
    if scenario_path is None or not scenario_path.strip():
        return None, None
    scenario_file = Path(scenario_path)
    if not scenario_file.is_absolute():
        scenario_file = (workdir / scenario_file).resolve()
    if not scenario_file.exists():
        return None, None

    scenario_hash = hashlib.sha256(scenario_file.read_bytes()).hexdigest()
    payload = {
        "manifest_version": "1.0",
        "scenario_path": str(scenario_file),
        "scenario_hash": scenario_hash,
        "scenario_required_files": sorted(
            {item for item in scenario_required_files if item.strip()}
        ),
        "repeat_repetitions": repetitions,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    manifest_path = workdir / ".ash-hawk" / "thin_runtime" / "runs" / run_id / "eval_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({**payload, "manifest_hash": manifest_hash}, indent=2), encoding="utf-8"
    )
    return manifest_path, manifest_hash


def verify_eval_manifest(
    *,
    manifest_path: str | None,
    manifest_hash: str | None,
    scenario_path: str | None,
) -> tuple[bool, str | None]:
    if manifest_path is None or manifest_hash is None:
        return False, "No eval manifest is available"
    path = Path(manifest_path)
    if not path.exists():
        return False, f"Eval manifest is missing: {manifest_path}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return False, "Eval manifest is malformed"
    stored_hash = raw.get("manifest_hash")
    if stored_hash != manifest_hash:
        return False, "Eval manifest hash mismatch"
    if scenario_path is None or not scenario_path.strip():
        return False, "Scenario path is missing for candidate validation"

    candidate_scenario = Path(scenario_path)
    if not candidate_scenario.is_absolute():
        candidate_scenario = candidate_scenario.resolve()
    if not candidate_scenario.exists():
        return False, f"Candidate scenario path is missing: {candidate_scenario}"

    candidate_hash = hashlib.sha256(candidate_scenario.read_bytes()).hexdigest()
    if raw.get("scenario_hash") != candidate_hash:
        return False, "Candidate scenario drifted from the frozen eval manifest"
    return True, None
