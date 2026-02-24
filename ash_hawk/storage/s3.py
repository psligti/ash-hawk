"""S3-compatible storage backend for Ash Hawk using aiobotocore."""

from __future__ import annotations

import json
from typing import Any

import pydantic as pd
from aiobotocore.config import AioConfig
from aiobotocore.session import get_session

from ash_hawk.storage import StoredTrial
from ash_hawk.types import (
    EvalRunSummary,
    EvalSuite,
    EvalTrial,
    RunEnvelope,
    ToolSurfacePolicy,
    TrialEnvelope,
)


class S3Config(pd.BaseModel):
    """Configuration for S3-compatible storage.

    Supports any S3-compatible service including AWS S3, MinIO, Ceph,
    DigitalOcean Spaces, etc.
    """

    bucket: str = pd.Field(description="S3 bucket name")
    prefix: str = pd.Field(default="", description="Optional prefix/path within bucket")
    endpoint_url: str | None = pd.Field(
        default=None,
        description="Custom endpoint URL for S3-compatible services (e.g., MinIO)",
    )
    region_name: str = pd.Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str | None = pd.Field(default=None, description="AWS access key ID")
    aws_secret_access_key: str | None = pd.Field(default=None, description="AWS secret access key")

    model_config = pd.ConfigDict(extra="forbid")


def _dump_model(model: pd.BaseModel) -> dict[str, Any]:
    """Recursively serialize a Pydantic model to a dict."""
    model_type = type(model)
    result: dict[str, Any] = {}
    computed_fields = model_type.model_computed_fields.keys()
    for field_name in model_type.model_fields.keys():
        if field_name in computed_fields:
            continue
        value = getattr(model, field_name)
        if isinstance(value, pd.BaseModel):
            result[field_name] = _dump_model(value)
        elif isinstance(value, dict):
            result[field_name] = {
                k: _dump_model(v) if isinstance(v, pd.BaseModel) else v for k, v in value.items()
            }
        elif isinstance(value, list):
            result[field_name] = [
                _dump_model(item) if isinstance(item, pd.BaseModel) else item for item in value
            ]
        else:
            result[field_name] = value
    return result


class S3Storage:
    """S3-compatible storage backend using aiobotocore.

    Object key structure:
        {prefix}/{suite_id}/suite.json
        {prefix}/{suite_id}/runs/{run_id}/envelope.json
        {prefix}/{suite_id}/runs/{run_id}/trials/{trial_id}.json
        {prefix}/{suite_id}/runs/{run_id}/summary.json
    """

    def __init__(self, config: S3Config) -> None:
        self._config = config
        self._session = get_session()
        self._client: Any = None

    def _build_key(self, *parts: str) -> str:
        """Build an S3 key from parts, optionally including prefix."""
        if self._config.prefix:
            return f"{self._config.prefix}/{'/'.join(parts)}"
        return "/".join(parts)

    def _suite_key(self, suite_id: str) -> str:
        return self._build_key(suite_id, "suite.json")

    def _envelope_key(self, suite_id: str, run_id: str) -> str:
        return self._build_key(suite_id, "runs", run_id, "envelope.json")

    def _trial_key(self, suite_id: str, run_id: str, trial_id: str) -> str:
        return self._build_key(suite_id, "runs", run_id, "trials", f"{trial_id}.json")

    def _summary_key(self, suite_id: str, run_id: str) -> str:
        return self._build_key(suite_id, "runs", run_id, "summary.json")

    def _runs_prefix(self, suite_id: str) -> str:
        return self._build_key(suite_id, "runs") + "/"

    async def _get_client(self) -> Any:
        """Get or create the S3 client."""
        if self._client is None:
            config = AioConfig(region_name=self._config.region_name)
            kwargs: dict[str, Any] = {"config": config}
            if self._config.endpoint_url:
                kwargs["endpoint_url"] = self._config.endpoint_url
            if self._config.aws_access_key_id:
                kwargs["aws_access_key_id"] = self._config.aws_access_key_id
            if self._config.aws_secret_access_key:
                kwargs["aws_secret_access_key"] = self._config.aws_secret_access_key

            self._client = await self._session.create_client("s3", **kwargs).__aenter__()
        return self._client

    async def close(self) -> None:
        """Close the S3 client connection."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def _read_json(self, key: str) -> dict[str, Any] | None:
        """Read and parse JSON from S3. Returns None if not found."""
        client = await self._get_client()
        try:
            response = await client.get_object(Bucket=self._config.bucket, Key=key)
            async with response["Body"] as stream:
                content = await stream.read()
            return json.loads(content)
        except client.exceptions.NoSuchKey:
            return None
        except client.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise

    async def _write_json(self, key: str, data: dict[str, Any]) -> None:
        """Write JSON data to S3."""
        client = await self._get_client()
        content = json.dumps(data, indent=2)
        await client.put_object(
            Bucket=self._config.bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="application/json",
        )

    async def save_suite(self, suite: EvalSuite) -> None:
        """Save an evaluation suite definition."""
        await self._write_json(self._suite_key(suite.id), _dump_model(suite))

    async def load_suite(self, suite_id: str) -> EvalSuite | None:
        """Load an evaluation suite by ID."""
        data = await self._read_json(self._suite_key(suite_id))
        if data is None:
            return None
        return EvalSuite.model_validate(data)

    async def save_run_envelope(self, suite_id: str, envelope: RunEnvelope) -> None:
        """Save a run envelope."""
        await self._write_json(self._envelope_key(suite_id, envelope.run_id), envelope.model_dump())

    async def load_run_envelope(self, suite_id: str, run_id: str) -> RunEnvelope | None:
        """Load a run envelope."""
        data = await self._read_json(self._envelope_key(suite_id, run_id))
        if data is None:
            return None
        return RunEnvelope.model_validate(data)

    async def save_trial(
        self,
        suite_id: str,
        run_id: str,
        trial: EvalTrial,
        envelope: TrialEnvelope,
        policy: ToolSurfacePolicy,
    ) -> None:
        """Save a trial with its envelope and policy."""
        stored = {
            "trial": _dump_model(trial),
            "envelope": _dump_model(envelope),
            "policy": _dump_model(policy),
        }
        await self._write_json(self._trial_key(suite_id, run_id, trial.id), stored)

    async def load_trial(self, suite_id: str, run_id: str, trial_id: str) -> StoredTrial | None:
        """Load a stored trial with all associated data."""
        data = await self._read_json(self._trial_key(suite_id, run_id, trial_id))
        if data is None:
            return None
        return StoredTrial(
            trial=EvalTrial.model_validate(data["trial"]),
            envelope=TrialEnvelope.model_validate(data["envelope"]),
            policy=ToolSurfacePolicy.model_validate(data["policy"]),
        )

    async def list_runs(self, suite_id: str) -> list[str]:
        """List all run IDs for a suite."""
        client = await self._get_client()
        prefix = self._runs_prefix(suite_id)
        run_ids: set[str] = set()

        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract run_id from key: {prefix}/{suite_id}/runs/{run_id}/...
                parts = key[len(prefix) :].split("/")
                if len(parts) >= 2:
                    run_ids.add(parts[0])

        return sorted(run_ids)

    async def list_suites(self) -> list[str]:
        """List all suite IDs."""
        client = await self._get_client()
        suite_ids: set[str] = set()

        # List objects and look for suite.json files
        prefix = self._config.prefix + "/" if self._config.prefix else ""
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Check if this is a suite.json file
                if key.endswith("/suite.json"):
                    # Extract suite_id: {prefix}/{suite_id}/suite.json
                    parts = key.replace("/suite.json", "").split("/")
                    if parts:
                        suite_id = parts[-1]
                        suite_ids.add(suite_id)

        return sorted(suite_ids)

    async def save_summary(self, suite_id: str, run_id: str, summary: EvalRunSummary) -> None:
        """Save a run summary."""
        await self._write_json(self._summary_key(suite_id, run_id), _dump_model(summary))

    async def load_summary(self, suite_id: str, run_id: str) -> EvalRunSummary | None:
        """Load a run summary."""
        data = await self._read_json(self._summary_key(suite_id, run_id))
        if data is None:
            return None
        return EvalRunSummary.model_validate(data)
