# Complete Evaluation Example

This example demonstrates all three grader layers in ash-hawk.

## Structure

```
complete-eval/
├── suites/
│   └── complete-example.yaml    # Suite definition with 3 tasks
├── fixtures/
│   ├── src/
│   │   └── models.py            # User model (provided to agent)
│   ├── tests/
│   │   ├── test_calculate_average.py
│   │   └── test_user_repository.py
│   └── expected/
│       └── user_repository.py   # Expected implementation
└── README.md
```

## Tasks

### Task 1: String Match (First Layer Only)

Simple deterministic grading - checks if response contains "4".

### Task 2: Test + LLM Judge (First + Second Layer)

- 60% weight: Test runner executes pytest
- 40% weight: LLM judge evaluates code quality

### Task 3: Composite (All Three Layers)

- 40% weight: Test runner
- 20% weight: Static analysis (ruff + mypy)
- 40% weight: LLM judge for production readiness

## Running

```bash
ash-hawk run examples/complete-eval/suites/complete-example.yaml
```

## Creating Fixtures

### Test Files

Test files should:
1. Import the agent's generated code
2. Test all specified requirements
3. Include edge cases
4. Use pytest fixtures for setup

### Expected Outputs (Optional)

Reference implementations that demonstrate:
1. Correct functionality
2. Good code structure
3. Proper error handling
4. Type hints and docstrings
