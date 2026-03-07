# F1: Judge Output Normalizer

**Status**: IMPLEMENTED  
**Priority**: P1 (Critical)  
**Source**: iron-rook `judge_normalizer.py` (434 lines)

## Problem Statement

LLM-based judges return responses in inconsistent formats across different providers (OpenAI, Anthropic, local models). This causes:
- Evaluation failures when expected fields are missing
- Incorrect score extraction from nested JSON
- Need for provider-specific parsing logic in each grader

## Solution

A unified normalizer that handles all common LLM output formats and extracts consistent `NormalizedJudgeOutput`.

## Supported Formats

```python
# Format 1: Direct
{"score": 0.8, "passed": true, "reasoning": "..."}

# Format 2: Nested in answer
{"answer": {"overall_score": 0.8, "reasoning": "..."}}

# Format 3: Alternative nested
{"answer": {"score": 0.8}}

# Format 4: Overall assessment
{"answer": {"overall_assessment": 0.8}}
{"answer": {"overall_assessment": {"score": 0.8}}}

# Format 5: Dimension scores
{"factual_accuracy": {"score": 0.8}, "logical_soundness": {"score": 0.7}}

# Format 6: Text patterns
"8/10" or "80%" or "score: 0.8"

# Format 7: Markdown code blocks
```json
{"score": 0.8}
```
```

## API

```python
from ash_hawk.graders.judge_normalizer import normalize_judge_output, NormalizedJudgeOutput

# Usage
result: NormalizedJudgeOutput = normalize_judge_output(
    raw_output,  # str, dict, or Any
    pass_threshold=0.7  # Optional: threshold for deriving passed from score
)

# Result fields
result.score        # float [0.0, 1.0]
result.passed       # bool
result.reasoning    # str
result.issues       # list[str]
result.strengths    # list[str]
result.breakdown    # dict[str, float] | None
```

## Integration

### In LLMJudgeGrader

```python
# Before
score = output.get("score", 0.0)
passed = output.get("passed", score >= self.threshold)

# After
normalized = normalize_judge_output(output, pass_threshold=self.threshold)
score = normalized.score
passed = normalized.passed
reasoning = normalized.reasoning
```

### In EvalSuite YAML

No changes required - normalizer is used internally by graders.

## Files

- `ash_hawk/graders/judge_normalizer.py` - Main implementation
- `tests/graders/test_judge_normalizer.py` - Test coverage

## Testing

```bash
uv run pytest tests/graders/test_judge_normalizer.py -v
```

## Metrics

- Handles 7+ distinct output formats
- 100% test coverage for all format paths
- Zero dependencies (pure Python)

## Future Enhancements

- [ ] Add support for XML output formats
- [ ] Add confidence interval extraction
- [ ] Add multi-turn conversation parsing
