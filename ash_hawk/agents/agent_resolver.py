"""Shared agent path resolution for CLI commands.

Provides a unified function to resolve an agent reference (either a filesystem
path or a short name) into an absolute, symlink-resolved directory path.  Used
by the ``run``, ``thin``, and ``improve`` CLI sub-commands so that each one does
not need to duplicate its own lookup logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ash_hawk.agents.source_workspace import infer_agent_name_from_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResolution:
    """Result of resolving an agent reference.

    Attributes:
        path: Absolute, symlink-resolved path to the agent directory.
        name: Human-readable agent name (file stem or the short name used on
            the command line).
        resolved_from: How the path was resolved — ``"cli_path"`` when the
            reference was an existing filesystem path, ``"name_lookup"`` when
            it was found via a conventional directory search.
    """

    path: Path
    name: str
    resolved_from: Literal["cli_path", "name_lookup"]


class AgentResolutionError(ValueError):
    """Raised when an agent reference cannot be resolved to an existing path."""


_DAWN_KESTREL_AGENT_DIR = Path(".dawn-kestrel") / "agents"
_OPENCODE_AGENT_DIR = Path(".opencode") / "agent"


def resolve_agent_path(agent_ref: str, workdir: Path) -> AgentResolution:
    """Resolve an agent reference to an absolute, symlink-resolved path.

    The function tries, in order:

    1. **Direct path** — if ``agent_ref`` points to an existing file or
       directory on disk (resolved relative to *workdir* when the path is
       relative, or as-is when absolute), return it.
    2. **Dawn Kestrel registry** — check
       ``<workdir>/.dawn-kestrel/agents/<agent_ref>``.
    3. **OpenCode agent directory** — check
       ``<workdir>/.opencode/agent/<agent_ref>.md`` and return its parent
       joined with *agent_ref* (matching the convention used by the thin
       runner).

    If none of the above succeeds an :class:`AgentResolutionError` is raised.

    Args:
        agent_ref: Either a filesystem path or a short agent name supplied on
            the command line.
        workdir: Base directory used to resolve relative paths and to search
            for conventional agent locations.

    Returns:
        An :class:`AgentResolution` with the resolved absolute path.

    Raises:
        AgentResolutionError: If *agent_ref* cannot be resolved.
    """
    # --- 1. Direct filesystem path (wins over name lookup) ----------------
    direct = Path(agent_ref)
    if not direct.is_absolute():
        direct = workdir / direct
    if direct.exists():
        resolved = direct.resolve()
        return AgentResolution(
            path=resolved,
            name=infer_agent_name_from_path(resolved),
            resolved_from="cli_path",
        )

    # --- 2. Dawn Kestrel registry lookup ----------------------------------
    dawn_path = (workdir / _DAWN_KESTREL_AGENT_DIR / agent_ref).resolve()
    if dawn_path.exists():
        return AgentResolution(
            path=dawn_path,
            name=agent_ref,
            resolved_from="name_lookup",
        )

    # --- 3. Adapter registry (agent source path) -------------------------
    _normalized = agent_ref.replace("-", "_")
    try:
        from ash_hawk.scenario.registry import get_default_adapter_registry

        registry = get_default_adapter_registry()
        adapter = registry.get(agent_ref) or registry.get(_normalized)
        if adapter is not None and hasattr(adapter, "agent_source_path"):
            src_path = adapter.agent_source_path()
            if src_path is not None and src_path.is_dir():
                return AgentResolution(
                    path=src_path,
                    name=agent_ref,
                    resolved_from="name_lookup",
                )
    except Exception:
        logger.debug("Adapter registry lookup failed for %s", agent_ref, exc_info=True)

    # --- 4. OpenCode agent directory lookup --------------------------------
    opencode_md = (workdir / _OPENCODE_AGENT_DIR / f"{agent_ref}.md").resolve()
    if opencode_md.exists():
        opencode_dir = opencode_md.parent / agent_ref
        return AgentResolution(
            path=opencode_dir,
            name=agent_ref,
            resolved_from="name_lookup",
        )

    # --- Nothing found ----------------------------------------------------
    raise AgentResolutionError(
        f"Cannot resolve agent '{agent_ref}': not a valid path and not found "
        f"in .dawn-kestrel/agents/ or .opencode/agent/ under {workdir}"
    )
