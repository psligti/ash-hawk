# F5: Calibration Framework

**Status**: IMPLEMENTED  
**Priority**: P2 (High Value)  
**Source**: iron-rook + bolt-merlin calibration infrastructure

## Problem Statement

LLM judge scores may not correlate well with actual outcomes. Without calibration:
- Pass thresholds are set arbitrarily
- Some judges are overconfident, others underconfident
- No visibility into prediction reliability

## Solution

A calibration framework that computes:
1. **Expected Calibration Error (ECE)** - Gap between predicted and actual
2. **Brier Score** - Mean squared error of predictions
3. **Reliability Diagram** - Per-bin accuracy visualization

## API

```python
from ash_hawk.calibration import CalibrationRunner, GroundTruth, CalibrationResult

# Define ground truth
ground_truths = [
    GroundTruth(task_id="task-001", expected_passed=True),
    GroundTruth(task_id="task-002", expected_passed=False, notes="Edge case"),
    GroundTruth(task_id="task-003", expected_passed=True),
]

# Run calibration
runner = CalibrationRunner()
runner.add_results_from_ground_truth(eval_results, ground_truths)
result: CalibrationResult = runner.compute()

# Metrics
result.ece           # Expected Calibration Error (0.0 = perfect)
result.brier_score   # Brier score (0.0 = perfect)
result.pass_rate     # Actual pass rate
result.mean_predicted  # Average predicted score
result.mean_actual   # Average actual outcome (0.0-1.0)
result.per_bin_accuracy  # Reliability diagram data
result.disagreement_tasks  # Tasks where prediction != actual
result.is_well_calibrated  # True if ECE < 0.1
```

## Ground Truth Format

```json
{
  "labels": [
    {"task_id": "task-001", "expected_passed": true},
    {"task_id": "task-002", "expected_passed": false, "notes": "Edge case"}
  ]
}
```

## Usage Patterns

### Per-Agent Calibration

```python
# Calibrate security agent specifically
security_results = {t: r for t, r in all_results.items() if t.startswith("sec-")}
security_gt = [gt for gt in all_gt if "security" in gt.task_id]

runner = CalibrationRunner()
runner.add_results_from_ground_truth(security_results, security_gt)
result = runner.compute()

# Set agent-specific threshold
security_threshold = result.mean_predicted  # Adjust based on calibration
```

### Disagreement Detection

```python
disagreement = runner.check_disagreement(threshold=0.3)
for task_id in disagreement:
    print(f"Task {task_id} prediction differs significantly from ground truth")
```

## CLI Usage

```bash
# Run calibration
uv run ash-hawk calibrate ./evals/suites/security.yaml \
    --ground-truth ./evals/ground-truth/security.json \
    --output ./reports/calibration.json

# Compare to baseline
uv run ash-hawk calibrate ./evals/suites/security.yaml \
    --ground-truth ./evals/ground-truth/security.json \
    --baseline ./reports/calibration-baseline.json
```

## Files

- `ash_hawk/calibration/__init__.py` - Exports
- `ash_hawk/calibration/types.py` - Data types
- `ash_hawk/calibration/ece.py` - ECE computation
- `ash_hawk/calibration/brier.py` - Brier score computation
- `ash_hawk/calibration/runner.py` - CalibrationRunner class
- `tests/calibration/` - Test coverage

## Interpretation

| ECE | Interpretation |
|-----|----------------|
| < 0.05 | Excellent calibration |
| 0.05 - 0.10 | Good calibration |
| 0.10 - 0.15 | Fair calibration |
| > 0.15 | Poor calibration |

| Brier Score | Interpretation |
|-------------|----------------|
| < 0.10 | Excellent predictions |
| 0.10 - 0.20 | Good predictions |
| 0.20 - 0.30 | Fair predictions |
| > 0.30 | Poor predictions |

## Testing

```bash
uv run pytest tests/calibration/ -v
```
