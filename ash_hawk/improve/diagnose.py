# type-hygiene: skip-file
from __future__ import annotations

import importlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash_hawk.types import EvalTrial

logger = logging.getLogger(__name__)

DIAGNOSIS_PROMPT = """You are analyzing a failed evaluation trial. Diagnose why it failed.

Trial ID: {trial_id}
Task ID: {task_id}
Agent Response: {agent_response}
Grader Results: {grader_results}
Error Trace: {error_trace}

Provide your diagnosis as JSON:
{{
    "failure_summary": "one-line summary",
    "root_cause": "detailed root cause analysis",
    "suggested_fix": "concrete fix suggestion",
    "target_files": ["file1.py", "file2.py"],
    "confidence": 0.8
}}"""


@dataclass
class Diagnosis:
    trial_id: str
    failure_summary: str
    root_cause: str
    suggested_fix: str
    target_files: list[str] = field(default_factory=list)
    confidence: float = 0.0


async def diagnose_failures(
    failures: list[EvalTrial],
    trace_dir: Path | None = None,
) -> list[Diagnosis]:
    diagnoses: list[Diagnosis] = []
    for failure in failures:
        try:
            diagnosis = await _diagnose_single(failure, trace_dir)
            if diagnosis is not None:
                diagnoses.append(diagnosis)
        except Exception:
            logger.warning("Diagnosis failed for trial %s", failure.id, exc_info=True)
    return diagnoses


async def _diagnose_single(trial: EvalTrial, trace_dir: Path | None) -> Diagnosis | None:
    transcript = trial.result.transcript if trial.result else None
    agent_response = ""
    grader_results_str = "[]"
    error_trace = ""

    if transcript is not None:
        agent_response = str(transcript.agent_response or "")[:2000]
        if trial.result and trial.result.grader_results:
            grader_results_str = json.dumps([r.model_dump() for r in trial.result.grader_results])[
                :2000
            ]
        error_trace = str(transcript.error_trace or "")[:2000]

    prompt = DIAGNOSIS_PROMPT.format(
        trial_id=trial.id,
        task_id=trial.task_id,
        agent_response=agent_response,
        grader_results=grader_results_str,
        error_trace=error_trace,
    )

    response = await _call_llm(prompt)
    if response is None:
        return None

    return _parse_diagnosis_response(trial.id, response)


async def _call_llm(prompt: str) -> str | None:
    try:
        settings_module = importlib.import_module("dawn_kestrel.core.settings")
        client_module = importlib.import_module("dawn_kestrel.llm.client")
    except ImportError:
        logger.warning("dawn-kestrel not installed, returning stub diagnosis")
        return None

    try:
        settings = settings_module.get_settings()
        provider = settings.get_default_provider().value
        model = settings.get_default_model(provider)
        api_key_secret = settings.get_api_key_for_provider(provider)
        api_key = api_key_secret.get_secret_value() if api_key_secret else None

        LLMClient = getattr(client_module, "LLMClient")
        client = LLMClient(provider_id=provider, model=model, api_key=api_key)

        result = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )
        return result.choices[0].message.content if result.choices else None
    except Exception:
        logger.warning("LLM call failed", exc_info=True)
        return None


def _parse_diagnosis_response(trial_id: str, response: str) -> Diagnosis | None:
    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if not json_match:
        logger.warning("No JSON found in diagnosis response for %s", trial_id)
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in diagnosis response for %s", trial_id)
        return None

    required_fields = ["failure_summary", "root_cause", "suggested_fix"]
    if not all(f in data for f in required_fields):
        logger.warning("Missing required fields in diagnosis for %s", trial_id)
        return None

    return Diagnosis(
        trial_id=trial_id,
        failure_summary=str(data["failure_summary"]),
        root_cause=str(data["root_cause"]),
        suggested_fix=str(data["suggested_fix"]),
        target_files=data.get("target_files", []),
        confidence=float(data.get("confidence", 0.0)),
    )
