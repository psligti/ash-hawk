---
name: python-bugfix-path-aware
description: "Fixes Python bugs by first locating source and test files within the project structure, then applying targeted fixes"
---

## What I do

I locate and fix Python bugs by:
1. First discovering the actual project structure and file locations
2. Verifying paths before running commands (avoiding assumptions about file locations)
3. Reading test and source files from their correct locations
4. Applying minimal, targeted fixes based on test failures
5. Re-running tests from the appropriate directory to verify fixes

## When to use me

Use me when you need to fix Python bugs in a project where:
- Test and source files may not be at the root of the working directory
- File locations need to be discovered dynamically
- You need to verify file paths before executing commands
- The project structure may have subdirectories containing the code

## Guidelines

### File Discovery Phase (CRITICAL - Do this first)
1. **Never assume file locations** - Always verify before running commands
2. Start with `ls -la` to understand the top-level structure
3. Use `find` or `ls -la` on likely subdirectories (src/, tests/, evals/, etc.)
4. If user mentions paths like `src/solution.py`, check if those exist directly or search recursively
5. Search patterns: `find . -name "test_bug.py" -type f`, `find . -name "solution.py" -type f`

### Path Resolution
- If files are found in subdirectories (e.g., `./evals/python-bugfix/XX-filename/`), run commands from within that directory: `cd ./evals/python-bugfix/XX-filename/ && python test_bug.py`
- Or use full paths: `python ./evals/python-bugfix/XX-filename/test_bug.py`

### Bug Fixing Process
1. Run the test from the correct directory using the actual path
2. Read the test file to understand what's being tested
3. Read the source file to identify the bug
4. Apply minimal changes (single line or small set of changes)
5. Run the tests again to verify the fix

### Common Bug Types to Look For
- Off-by-one errors in loops and slices
- Missing return statements  
- Wrong operators (and vs or, > vs >=)
- Variable name typos
- Missing exception handling
- Incorrect comparisons (is vs ==)
- Logic inversions
- Missing import statements

### Error Handling
- If a file command fails with "No such file or directory", immediately stop and find the correct path using `ls -la` or `find`
- Don't repeat the same failing command - investigate the structure first