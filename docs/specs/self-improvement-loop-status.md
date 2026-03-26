# Self-Improvement Loop Implementation Status

## ✅ Completed

### Infrastructure (Commits: e4f2a3d, faaaf8e, 6620b6f, 204ea33)

1. **FixtureSplitter** (`ash_hawk/improvement/fixture_splitter.py`)
   - Deterministic 70/30 train/holdout split using MD5 hashing
   - Seed-based reproducibility
   - Handles edge cases (empty, single fixture)

2. **GuardrailChecker** (`ash_hawk/improvement/guardrails.py`)
   - Max consecutive holdout drops: 3 (default)
   - Max reverts: 5 (default)
   - Plateau detection: 5 cycles at 0.02 threshold
   - Tracks iteration history

3. **cycle_runner.py** Updates
   - `heldout_scenarios: list[Path] | None` parameter
   - `guardrail_config: GuardrailConfig | None` parameter
   - Integrated GuardrailChecker for early stopping
   - Reports heldout scores during iterations

4. **Evaluation Fixtures** (`evals/python-bugfix/`)
   - 20 realistic bug fixtures
   - Bug types: off-by-one, missing returns, wrong operators, typos, missing imports, exception handling, method errors, logic errors, data structure bugs
   - CompositeGrader: TestRunner (0.7) + DiffConstraints (0.2) + Efficiency (0.1)

5. **Test Coverage**
   - 27 tests passing
   - 100% coverage of FixtureSplitter and GuardrailChecker
   - Integration tests for self-improvement loop

## 🔍 Known Issue: Rate Limiting with Concurrent Agents

When running multiple agents concurrently, the LLM provider may rate limit, causing stream cancellations:

```
LLM stream cancelled during collection
transient failure after 74.13s: LLM stream cancelled before completion
```

**Root Cause**: Too many concurrent agents hitting provider rate limits.

**Solution**: The `LLMClient` uses tenacity for retries with exponential backoff, acting as a "poor man's queue". Increase retries to handle rate limiting:

```bash
# Increase from default 10 to 20-30 for heavy concurrent usage
export ASH_HAWK_AUTO_RESEARCH_LLM_MAX_RETRIES=20

# Then run
uv run ash-hawk auto-research run -s evals/scenarios/*.scenario.yaml
```

**Configuration**:
- Default: 10 retries
- Max: 50 retries  
- Env var: `ASH_HAWK_AUTO_RESEARCH_LLM_MAX_RETRIES`

**Future**: A proper request queue is planned to serialize agent execution with configurable workers.

## 🚀 Usage

```bash
# Run improvement cycle
uv run python scripts/run_improvement_cycle.py

# Or via CLI
ash-hawk auto-research run \
  --scenarios evals/python-bugfix/*/scenario.yaml \
  --iterations 100 \
  --threshold 0.02
```

## 📊 Verification

```python
from ash_hawk.improvement.fixture_splitter import FixtureSplitter
from ash_hawk.improvement.guardrails import GuardrailConfig, GuardrailChecker
from pathlib import Path

# Split fixtures
scenarios = list(Path('evals/python-bugfix').glob('*/scenario.yaml'))
splitter = FixtureSplitter(seed=42, train_ratio=0.7)
split = splitter.split(scenarios)
# Train: 14, Holdout: 6

# Configure guardrails
config = GuardrailConfig(
    max_consecutive_holdout_drops=3,
    max_reverts=5,
    plateau_window=5,
    plateau_threshold=0.02,
)
checker = GuardrailChecker(config)

# Track iterations
checker.record_iteration(score=0.5, applied=True)
if checker.should_stop():
    print(f"Stopping: {checker.stop_reason}")
```

## Next Steps

To achieve +20pt improvement on heldout:
1. Fix workdir issue in adapter
2. Run full 100-iteration cycle
3. Verify improvement on holdout set
