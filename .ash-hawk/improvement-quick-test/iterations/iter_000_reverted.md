---
name: "python-bugfix-systematic"
description: "Systematic Python bug fixing with file discovery, using only available tools to locate, analyze, and fix Python code based on test failures."
---

## What I do

I systematically identify and fix Python bugs by first discovering the actual location of test and source files, analyzing test failures, reading the relevant code, and making minimal targeted fixes to pass all tests.

## When to use me

Use when you need to fix a bug in Python code where test files and source files may be in various locations (current directory, subdirectories, or specific paths), and you need to ensure tests pass.

## Guidelines

### File Discovery Phase (always do first)

1. **Check current directory** - Run `ls -la` to see what files exist in the working directory
2. **Find test file** - If `test_bug.py` isn't present, use `find . -name "test_bug.py"` to locate it
3. **Find source file** - If `src/solution.py` isn't present, use `find . -name "solution.py"` or look for common patterns
4. **Confirm paths** - Note the actual paths to both files before proceeding

### Available Tools Only

- Use `bash` tool with commands like `cat`, `python`, `cd`, `ls`, `find`
- Do NOT try to use `read` tool (it doesn't exist)
- Use `cat <filepath>` to read file contents
- Use `python <testfile>` to run tests (may need to `cd` to directory first)

### Analysis Process

1. **Run the test first** from the correct directory to see the actual failure
2. **Read the test file** to understand what's being tested
3. **Read the source file** to identify the bug
4. **Map test expectations** to source code behavior

### Fixing Process

1. Make **minimal changes** - only change what's necessary
2. Target the specific line causing the failure
3. Common bug types to watch for:
   - Off-by-one errors in loops and slices
   - Missing return statements
   - Wrong operators (and vs or, > vs >=)
   - Variable name typos
   - Missing exception handling
   - Incorrect comparisons (is vs ==)
   - Logic inversions
   - Missing imports

### Verification

1. After making changes, run the test again
2. Confirm all tests pass
3. If tests still fail, re-read the code and test output
4. Adjust the fix incrementally

### Handle Edge Cases

- If multiple files match, pick the most relevant one (usually in src/ or directory matching pattern)
- If tests need to import from a module, ensure you're running from the correct directory
- If the fix requires changing multiple files, do so systematically

### Example Command Flow