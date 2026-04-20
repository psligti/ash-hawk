from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from ash_hawk.thin_runtime.agents import AgentRegistry
from ash_hawk.thin_runtime.console import ThinRuntimeConsoleReporter, mute_console_logging
from ash_hawk.thin_runtime.context import RuntimeContextAssembler
from ash_hawk.thin_runtime.defaults import build_default_catalog
from ash_hawk.thin_runtime.hooks import HookDispatcher, HookRegistry
from ash_hawk.thin_runtime.memory import ThinRuntimeMemoryManager
from ash_hawk.thin_runtime.models import RuntimeGoal, ThinRuntimeCatalog, ThinRuntimeExecutionResult
from ash_hawk.thin_runtime.persistence import ThinRuntimePersistence
from ash_hawk.thin_runtime.runner import AgenticLoopRunner
from ash_hawk.thin_runtime.skills import SkillRegistry
from ash_hawk.thin_runtime.tools import ToolRegistry
from ash_hawk.types import ToolPermission, ToolSurfacePolicy


class ThinRuntimeHarness:
    def __init__(
        self,
        catalog: ThinRuntimeCatalog | None = None,
        workdir: Path | None = None,
        policy: ToolSurfacePolicy | None = None,
        storage_root: Path | None = None,
        console_output: bool = True,
    ) -> None:
        self.catalog = catalog or build_default_catalog()
        self.workdir = workdir or Path.cwd()
        self.console_output = console_output
        self.policy = policy or ToolSurfacePolicy(
            allowed_tools=[tool.name for tool in self.catalog.tools],
            default_permission=ToolPermission.ALLOW,
        )
        self.agents = AgentRegistry(self.catalog.agents)
        self.skills = SkillRegistry(self.catalog.skills)
        self.tools = ToolRegistry(self.catalog.tools)
        self.hooks = HookDispatcher(HookRegistry(self.catalog.hooks))
        self.console_reporter = (
            ThinRuntimeConsoleReporter(self.hooks) if self.console_output else None
        )
        self.memory = ThinRuntimeMemoryManager(self.catalog.memory_scopes)
        self.memory.hydrate_defaults()
        self.persistence = ThinRuntimePersistence(storage_root=storage_root)
        for scope_name, values in self.persistence.load_memory_snapshot().items():
            self.memory.write_scope(scope_name, values)
        self.context = RuntimeContextAssembler()
        self.runner = AgenticLoopRunner(
            agents=self.agents,
            skills=self.skills,
            tools=self.tools,
            hooks=self.hooks,
            memory=self.memory,
            persistence=self.persistence,
            context=self.context,
            workdir=self.workdir,
            policy=self.policy,
        )

    def execute(
        self,
        goal: RuntimeGoal,
        *,
        agent_name: str = "coordinator",
        requested_skills: list[str] | None = None,
        tool_execution_order: list[str] | None = None,
        scenario_path: str | None = None,
    ) -> ThinRuntimeExecutionResult:
        log_context = mute_console_logging() if self.console_output else nullcontext()
        with log_context:
            execution = self.runner.run(
                goal,
                agent_name=agent_name,
                requested_skills=requested_skills,
                tool_execution_order=tool_execution_order,
                scenario_path=scenario_path,
            )
        if self.console_reporter is not None:
            self.console_reporter.print_run_summary(execution, self.persistence)
        return execution


def create_default_harness(
    workdir: Path | None = None,
    policy: ToolSurfacePolicy | None = None,
    storage_root: Path | None = None,
    console_output: bool = True,
) -> ThinRuntimeHarness:
    return ThinRuntimeHarness(
        workdir=workdir,
        policy=policy,
        storage_root=storage_root,
        console_output=console_output,
    )
