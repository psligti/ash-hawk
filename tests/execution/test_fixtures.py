"""Tests for fixture resolution."""

from pathlib import Path

import pytest

from ash_hawk.scenario.fixtures import FixtureError, FixtureResolver
from ash_hawk.types import EvalSuite, EvalTask


class TestFixtureResolverInit:
    def test_init_with_path_and_suite(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")

        resolver = FixtureResolver(suite_path, suite)

        assert resolver.suite_dir == tmp_path

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")

        resolver = FixtureResolver(str(suite_path), suite)

        assert resolver.suite_dir == tmp_path


class TestFixtureResolverResolvePath:
    def test_resolve_relative_path(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "subdir" / "suite.yaml"
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        result = resolver.resolve_path("fixtures/data.json")

        assert result == tmp_path / "subdir" / "fixtures" / "data.json"

    def test_resolve_absolute_path(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        abs_path = "/absolute/path/file.txt"
        result = resolver.resolve_path(abs_path)

        assert result == Path(abs_path)


class TestFixtureResolverResolveTaskFixtures:
    def test_resolve_empty_fixtures(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(id="task-1", input="test")
        result = resolver.resolve_task_fixtures(task)

        assert result == {}

    def test_resolve_single_fixture(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input="test",
            fixtures={"data_file": "fixtures/data.json"},
        )
        result = resolver.resolve_task_fixtures(task)

        assert result == {"data_file": tmp_path / "fixtures" / "data.json"}

    def test_resolve_multiple_fixtures(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input="test",
            fixtures={
                "data_file": "fixtures/data.json",
                "config_file": "config/settings.yaml",
                "output_dir": "output/",
            },
        )
        result = resolver.resolve_task_fixtures(task)

        assert result["data_file"] == tmp_path / "fixtures" / "data.json"
        assert result["config_file"] == tmp_path / "config" / "settings.yaml"
        assert result["output_dir"] == tmp_path / "output"


class TestFixtureResolverInjectFixtures:
    def test_inject_fixtures_string_input(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input="plain string input",
            fixtures={"data_file": "fixtures/data.json"},
        )

        result = resolver.inject_fixtures(task)

        assert result.input == "plain string input"

    def test_inject_fixtures_dict_input_no_refs(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={"prompt": "test prompt", "other": "value"},
            fixtures={"data_file": "fixtures/data.json"},
        )

        result = resolver.inject_fixtures(task)

        assert result.input == {"prompt": "test prompt", "other": "value"}

    def test_inject_fixtures_dict_input_with_refs(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={
                "prompt": "Read the file",
                "file_path": "$data_file",
            },
            fixtures={"data_file": "fixtures/data.json"},
        )

        result = resolver.inject_fixtures(task)

        expected_path = str(tmp_path / "fixtures" / "data.json")
        assert result.input["file_path"] == expected_path

    def test_inject_fixtures_nested_dict(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={
                "prompt": "test",
                "config": {
                    "input_file": "$data_file",
                    "nested": {"output": "$output_dir"},
                },
            },
            fixtures={
                "data_file": "fixtures/data.json",
                "output_dir": "output/",
            },
        )

        result = resolver.inject_fixtures(task)

        expected_data = str(tmp_path / "fixtures" / "data.json")
        expected_output = str(tmp_path / "output")
        assert result.input["config"]["input_file"] == expected_data
        assert result.input["config"]["nested"]["output"] == expected_output

    def test_inject_fixtures_list_values(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={
                "files": ["$file1", "$file2", "static.txt"],
            },
            fixtures={
                "file1": "a.txt",
                "file2": "b.txt",
            },
        )

        result = resolver.inject_fixtures(task)

        expected_file1 = str(tmp_path / "a.txt")
        expected_file2 = str(tmp_path / "b.txt")
        assert result.input["files"][0] == expected_file1
        assert result.input["files"][1] == expected_file2
        assert result.input["files"][2] == "static.txt"

    def test_inject_fixtures_embedded_variable(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={
                "prompt": "Find code in $codebase_path and analyze $config_file",
            },
            fixtures={
                "codebase_path": "src/",
                "config_file": "config.yaml",
            },
        )

        result = resolver.inject_fixtures(task)

        expected_codebase = str(tmp_path / "src")
        expected_config = str(tmp_path / "config.yaml")
        assert (
            result.input["prompt"]
            == f"Find code in {expected_codebase} and analyze {expected_config}"
        )

    def test_inject_fixtures_word_boundary(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={
                "prompt": "$codebase and $codebase_path should differ",
            },
            fixtures={
                "codebase": "src/",
            },
            # Note: codebase_path is NOT a fixture, so $codebase_path should not be replaced
        )

        result = resolver.inject_fixtures(task)

        expected_codebase = str(tmp_path / "src")
        # $codebase should be replaced, but $codebase_path should NOT (word boundary)
        assert result.input["prompt"] == f"{expected_codebase} and $codebase_path should differ"


class TestFixtureResolverValidateFixtures:
    def test_validate_existing_fixtures(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        fixture_file = tmp_path / "data.json"
        fixture_file.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input="test",
            fixtures={"data_file": "data.json"},
        )

        errors = resolver.validate_fixtures(task)

        assert errors == []

    def test_validate_missing_fixtures(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input="test",
            fixtures={"data_file": "nonexistent.json"},
        )

        errors = resolver.validate_fixtures(task)

        assert len(errors) == 1
        assert "data_file" in errors[0]
        assert "not found" in errors[0]

    def test_validate_mixed_fixtures(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        existing_file = tmp_path / "exists.txt"
        existing_file.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input="test",
            fixtures={
                "exists": "exists.txt",
                "missing": "missing.txt",
            },
        )

        errors = resolver.validate_fixtures(task)

        assert len(errors) == 1
        assert "missing" in errors[0]


class TestFixtureResolverGetWorkingDir:
    def test_default_to_suite_dir(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "subdir" / "suite.yaml"
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        suite_path.touch()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(id="task-1", input="test")

        result = resolver.get_working_dir(task)

        assert result == tmp_path / "subdir"

    def test_from_fixtures(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input="test",
            fixtures={"working_dir": "workspace"},
        )

        result = resolver.get_working_dir(task)

        assert result == work_dir

    def test_from_input_dict(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={"prompt": "test", "working_dir": "workspace"},
        )

        result = resolver.get_working_dir(task)

        assert result == work_dir

    def test_from_input_dict_with_fixture_ref(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suite.yaml"
        suite_path.touch()

        work_dir = tmp_path / "workspace"
        work_dir.mkdir()

        suite = EvalSuite(id="test", name="Test Suite")
        resolver = FixtureResolver(suite_path, suite)

        task = EvalTask(
            id="task-1",
            input={"prompt": "test", "working_dir": "$workspace"},
            fixtures={"workspace": "workspace"},
        )

        result = resolver.get_working_dir(task)

        assert result == work_dir
