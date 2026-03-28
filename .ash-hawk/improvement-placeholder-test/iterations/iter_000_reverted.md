---
name: "python-bugfix-navigation"
description: "Expert at fixing Python bugs with proper directory navigation and context handling"
---

## What I do

I systematically debug and fix Python code by first ensuring I'm in the correct working directory, then running tests to identify failures, analyzing the source code, and making minimal, targeted fixes.

## When to use me

Use this skill when you need to:
- Fix a bug in Python code based on failing tests
- Work with code in a specific directory structure
- Debug test failures and implement minimal fixes
- Ensure tests pass after making code changes

## Guidelines

### Directory Navigation (CRITICAL)
1. **ALWAYS check your current working directory first** using `pwd`
2. When a working directory is specified, navigate to it BEFORE running any tests
3. Only use `find` or global searches if you cannot locate the expected files in the specified directory
4. If files aren't where expected, check the immediate directory structure first, then parent directories

### Debugging Process
1. First, verify you're in the correct directory with `pwd`
2. Run the test to see the failure
3. Read the source file to understand the bug
4. Identify the specific line causing the failure
5. Make a targeted fix with minimal changes

### Common Bug Types
- Off-by-one errors in loops and slices
- Missing return statements
- Wrong operators (and vs or, > vs >=)
- Variable name typos
- Missing exception handling
- Incorrect comparisons (is vs ==)
- Logic inversions
- Missing imports (math, collections, etc.)
- Missing `self` parameter in class methods

### Running Commands
- Always verify directory context before running tests
- Use `ls -la` to see what files are available in the current directory
- Run tests with `python test_bug.py` (or appropriate test command)
- After fixing, run tests again to verify the fix works

## Process

1. Check current directory: `pwd`
2. If not in the specified directory, navigate there: `cd <path>`
3. List files to confirm: `ls -la`
4. Run tests to see failure: `python test_bug.py`
5. Read source file: `cat src/solution.py` or similar
6. Identify the bug location and type
7. Make targeted fix (edit the file)
8. Run tests again to verify: `python test_bug.py`