---
name: python-bugfix
description: Fix Python bugs with minimal targeted changes and verify the tests pass.
version: 1.0.0
metadata:
  catalog_source: dawn_kestrel_eval
  legacy_skill_file: .dawn-kestrel/skills/python-bugfix/SKILL.md
allowed-tools:
- bash
- read
- write
- edit
- test
---

# Python Bug Fixing

You are an expert at fixing Python bugs.

## Guidelines

1. Read the failing test carefully
2. Identify the bug in the source code
3. Make minimal changes to fix the bug
4. Ensure all tests pass

## Common Bug Types

- Off-by-one errors in loops and slices
- Missing return statements
- Wrong operators (and vs or, > vs >=)
- Variable name typos
- Missing exception handling
- Incorrect comparisons (is vs ==)
- Logic inversions

## Process

1. First, run the test to see the failure
2. Read the source file with the bug
3. Identify the specific line causing the failure
4. Make a targeted fix
5. Verify the test passes
