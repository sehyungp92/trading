"""Runtime deployment metadata artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .lineage import LineageContext, lineage_to_payload, redact_config, stable_hash


_ALLOWED_METADATA_SOURCES = {
    "live_bot_runtime_deployment_metadata_v1",
    "vps_live_bot_runtime_deployment_metadata_v1",
}
_ALLOWED_ENVIRONMENTS = {"live_bot", "vps", "paper_vps", "production_vps"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _clean_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "y", "clean"}:
        return True
    if cleaned in {"0", "false", "no", "n", "dirty"}:
        return False
    return None


def _worktree_clean(repo_root: Path, env: Mapping[str, str]) -> bool:
    override = _clean_bool(env.get("SOURCE_CONTROL_WORKTREE_CLEAN"))
    if override is not None:
        return override
    status = _run_git(repo_root, "status", "--porcelain")
    return status == ""


def _normalise_remote(remote: str) -> str:
    value = remote.strip()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    if value.endswith(".git"):
        value = value[:-4]
    return value


def _repo_url(repo_root: Path, env: Mapping[str, str]) -> str:
    explicit = env.get("SOURCE_CONTROL_ORIGIN") or env.get("REPO_URL") or env.get("GITHUB_REPOSITORY_URL")
    remote = explicit or _run_git(repo_root, "config", "--get", "remote.origin.url")
    if not remote:
        return ""
    return _normalise_remote(remote)


def _file_sha256(path: Path) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return ""


def _host_fingerprint(env: Mapping[str, str]) -> str:
    explicit = env.get("RUNTIME_HOST_FINGERPRINT")
    if explicit:
        return explicit
    return stable_hash(
        "host_",
        {
            "computer": env.get("COMPUTERNAME", ""),
            "user_domain": env.get("USERDOMAIN", ""),
            "runner": env.get("RUNNER_NAME", ""),
        },
    )


def _bridge_id(lineage: Mapping[str, Any], explicit: str = "") -> str:
    if explicit:
        return explicit
    family_id = str(lineage.get("family_id") or "").strip()
    if family_id:
        return f"trading_{family_id}_family"
    strategy_id = str(lineage.get("strategy_id") or "").strip()
    return strategy_id or "trading_default_bridge"


def _metadata_source(emission_environment: str, env: Mapping[str, str]) -> str:
    explicit = env.get("DEPLOYMENT_METADATA_SOURCE", "")
    if explicit in _ALLOWED_METADATA_SOURCES:
        return explicit
    if emission_environment in {"vps", "paper_vps", "production_vps"}:
        return "vps_live_bot_runtime_deployment_metadata_v1"
    return "live_bot_runtime_deployment_metadata_v1"


def _emission_environment(env: Mapping[str, str]) -> str:
    explicit = env.get("EMISSION_ENVIRONMENT") or env.get("DEPLOYMENT_EMISSION_ENVIRONMENT")
    if explicit in _ALLOWED_ENVIRONMENTS:
        return explicit
    mode = (env.get("TRADING_MODE") or env.get("TRADING_ENV") or "").strip().lower()
    if mode == "paper":
        return "paper_vps"
    if mode == "live":
        return "production_vps"
    return "live_bot"


def build_deployment_metadata(
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    bridge_id: str = "",
    repo_root: str | Path | None = None,
    effective_config: Mapping[str, Any] | None = None,
    strategy_plugin_contract_path: str | Path | None = None,
    runtime_entrypoint: str = "",
    runtime_started_at_utc: str = "",
    runtime_instance_id: str = "",
    dry_run: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the approval-grade runtime deployment metadata payload.

    The artifact is evidence, not a gate.  A dirty local checkout therefore
    produces ``source_control_worktree_clean = false`` instead of pretending it
    is approval-ready.
    """
    env = env or os.environ
    root = Path(repo_root) if repo_root is not None else _repo_root()
    lineage_payload = lineage_to_payload(lineage)
    bridge = _bridge_id(lineage_payload, bridge_id)
    emitted_at = _utc_now()
    emission_environment = _emission_environment(env)
    repo_url = _repo_url(root, env)
    lineage_sha = str(lineage_payload.get("code_sha") or "")
    deployed_sha = (
        env.get("DEPLOYED_COMMIT_SHA")
        or env.get("SOURCE_CONTROL_COMMIT_SHA")
        or (lineage_sha if lineage_sha != "unknown" else "")
        or _run_git(root, "rev-parse", "HEAD")
        or "unknown"
    )
    source_sha = env.get("SOURCE_CONTROL_COMMIT_SHA") or deployed_sha
    worktree_clean = _worktree_clean(root, env)

    contract_path = Path(
        strategy_plugin_contract_path
        or env.get("STRATEGY_PLUGIN_CONTRACT_PATH")
        or f"trading_assistant_backtest/contracts/{bridge}/strategy_plugin_contract.json"
    )
    contract_hash = env.get("STRATEGY_PLUGIN_CONTRACT_HASH") or _file_sha256(
        contract_path if contract_path.is_absolute() else root / contract_path
    )
    config_hash = stable_hash(
        "cfg_",
        {
            "config_version": lineage_payload.get("config_version", ""),
            "effective_config": redact_config(dict(effective_config or {})),
        },
        length=32,
    )
    is_dry_run = dry_run if dry_run is not None else _clean_bool(env.get("DRY_RUN"))
    if is_dry_run is None:
        is_dry_run = False

    instance_id = runtime_instance_id or env.get("RUNTIME_INSTANCE_ID") or stable_hash(
        "runtime_",
        {
            "bridge_id": bridge,
            "deployment_id": lineage_payload.get("deployment_id", ""),
            "code_sha": deployed_sha,
        },
    )
    started_at = (
        runtime_started_at_utc
        or env.get("LIVE_RUNTIME_STARTED_AT_UTC")
        or env.get("RUNTIME_STARTED_AT_UTC")
        or emitted_at
    )

    return {
        "metadata_source": _metadata_source(emission_environment, env),
        "emission_environment": emission_environment,
        "repo_url": repo_url,
        "source_control_origin": repo_url,
        "deployed_commit_sha": deployed_sha,
        "source_control_commit_sha": source_sha,
        "source_control_worktree_clean": worktree_clean,
        "bot_id": "trading",
        "portfolio_id": str(lineage_payload.get("family_id") or lineage_payload.get("portfolio_id") or ""),
        "strategy_id": bridge,
        "config_hash": config_hash,
        "strategy_version": str(lineage_payload.get("strategy_version") or ""),
        "config_version": str(lineage_payload.get("config_version") or ""),
        "deployment_id": str(lineage_payload.get("deployment_id") or ""),
        "telemetry_schema_version": "trading_live_shadow_contract_v1",
        "strategy_plugin_contract_path": str(contract_path).replace("\\", "/"),
        "strategy_plugin_contract_hash": contract_hash,
        "emitted_at_utc": emitted_at,
        "live_runtime_started_at_utc": started_at,
        "runtime_entrypoint": runtime_entrypoint or env.get("RUNTIME_ENTRYPOINT", ""),
        "runtime_instance_id": instance_id,
        "runtime_host_fingerprint": _host_fingerprint(env),
        "dry_run": bool(is_dry_run),
        "approval_ready": bool(
            repo_url
            and repo_url.startswith("https://github.com/")
            and source_sha == deployed_sha
            and deployed_sha not in {"", "unknown"}
            and worktree_clean
            and contract_hash
        ),
    }


def write_deployment_metadata(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    bridge_id: str = "",
    repo_root: str | Path | None = None,
    effective_config: Mapping[str, Any] | None = None,
    strategy_plugin_contract_path: str | Path | None = None,
    runtime_entrypoint: str = "",
    runtime_started_at_utc: str = "",
    runtime_instance_id: str = "",
    dry_run: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    metadata = build_deployment_metadata(
        lineage,
        bridge_id=bridge_id,
        repo_root=repo_root,
        effective_config=effective_config,
        strategy_plugin_contract_path=strategy_plugin_contract_path,
        runtime_entrypoint=runtime_entrypoint,
        runtime_started_at_utc=runtime_started_at_utc,
        runtime_instance_id=runtime_instance_id,
        dry_run=dry_run,
        env=env,
    )
    bridge = str(metadata["strategy_id"])
    out_dir = Path(data_dir) / "deployments" / bridge
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "deployment_metadata.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)
    return path
