from __future__ import annotations

HIGH_SIGNAL_FILENAMES = {
    "agent.md",
    "agent_config.yaml",
}

HIGH_SIGNAL_SUFFIXES = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
)

LOW_SIGNAL_FILENAMES = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "MODULES.md",
    "README.md",
    "README_old.md",
    "Makefile",
}


def rank_workspace_targets(files: list[str]) -> list[str]:
    def _priority(name: str) -> tuple[int, int, str]:
        lowered = name.lower()
        if name in HIGH_SIGNAL_FILENAMES:
            return (0, 0, name)
        if name.endswith(HIGH_SIGNAL_SUFFIXES):
            return (1, 0, name)
        if name in LOW_SIGNAL_FILENAMES or lowered in {
            item.lower() for item in LOW_SIGNAL_FILENAMES
        }:
            return (3, 0, name)
        return (2, 0, name)

    return sorted(files, key=_priority)


def preferred_workspace_target(files: list[str]) -> str | None:
    ranked = rank_workspace_targets(files)
    return ranked[0] if ranked else None
