# B2. Fixtures: resolution + $injection

Fixtures let tasks reference files/dirs without hardcoding absolute paths.

## How it works

- `EvalTask.fixtures` stores fixture name → relative/absolute path.
- `FixtureResolver` resolves paths relative to suite directory (`ash_hawk/execution/fixtures.py`).
- `$fixture_name` placeholders are substituted recursively in `task.input`.

## Pattern

```yaml
tasks:
  - id: t1
    input:
      prompt: "Use $test_file and edit code in $project_dir"
    fixtures:
      test_file: ./fixtures/tests/test_x.py
      project_dir: ./fixtures/project/
```

At runtime those placeholders become resolved absolute paths.

## Quality checks

- Use `validate_fixtures()` to detect missing files early.
- Keep fixture names semantic (`test_file`, `repo_fixture`, `expected_dir`).

## Skill links

- Back to suites basics: [B1](01_suites_101.md)
- Next: grader strategy: [B3](05_graders_and_weights.md)
