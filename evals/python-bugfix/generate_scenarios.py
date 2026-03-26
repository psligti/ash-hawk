#!/usr/bin/env python3
"""Generate scenario files for all python-bugfix fixtures."""

from pathlib import Path
import shutil

BASE_DIR = Path("/Users/parkersligting/develop/pt/ash-hawk/evals/python-bugfix")
TEMPLATE = BASE_DIR / "python-bugfix.scenario.yaml"

fixtures = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name not.startswith(("0", "2") and "-" in d.name])

print(f"Found {len(fixtures)} fixtures")

for fixture_dir in fixtures:
    fixture_name = fixture_dir.name
    scenario_file = fixture_dir / f"{fixture_name}.scenario.yaml"
    
    shutil.copy(TEMPLATE, scenario_file)
    
    content = scenario_file.read_text()
    content = content.replace("id: python-bugfix-base", f"id: {fixture_name}")
    content = content.replace("description: Base scenario", f"description: Fix {fixture_name.replace('-', ' ').title()}")
    content = content.replace("solution_file: ./src/solution.py", f"solution_file: ./{fixture_dir.name}/src/solution.py")
    content = content.replace("test_file: ./test_bug.py", f"test_file: ./{fixture_dir.name}/test_bug.py")
    
    scenario_file.write_text(content)
    print(f"Created scenario: {scenario_file.name}")

print("Done!")
