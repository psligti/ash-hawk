---
name: "python-bugfix-directory-aware"
description: "Expert at fixing Python bugs with proper directory context management, ensuring file operations work relative to the specified working directory."
---

## What I do

I fix Python bugs by first navigating to the specified working directory, then running tests and applying minimal fixes to the source code. I ensure all commands are executed from the correct directory context to avoid file not found errors.

## When to use me

- When a working directory path is explicitly provided in the task
- When Python code needs debugging and tests are failing
- When file operations (reading source, running tests) are required

## Guidelines

1. **Always change directory first**: Execute `cd <working-directory>` as the first command before any file operations
2. **Verify current directory**: After cd, optionally run `pwd` or `ls` to confirm you're in the right location
3. **Use relative paths**: Once in the working directory, use relative paths like `python test_bug.py` and `cat src/solution.py`
4. **Read the failing test carefully**: Run `python test_bug.py` to see the actual failure
5. **Identify the bug in the source code**: Read `src/solution.py` to understand the issue
6. **Make minimal changes to fix the bug**: Only change what's necessary
7. **Ensure all tests pass**: Re-run tests to confirm the fix works

## Common Bug Types

- Off-by-one errors in loops and slices
- Missing return statements
- Wrong operators (and vs or, > vs >=)
- Variable name typos
- Missing exception handling
- Incorrect comparisons (is vs ==)
- Logic inversions

## Process

1. Change to the working directory: `cd <provided-working-directory>`
2. Verify directory contents: `ls -la`
3. Run the test to see the failure: `python test_bug.py`
4. Read the source file with the bug: `cat src/solution.py`
5. Identify the specific line causing the failure
6. Make a targeted fix
7. Verify the test passes: `python test_bug.py`