# Correctness Judge Prompt

Evaluates whether the agent's response is factually and logically correct.

## Metadata

- **Version**: 1.0.0
- **Category**: correctness
- **Description**: Judges the factual accuracy and logical soundness of responses

## Prompt Template

```
You are an expert evaluator judging the CORRECTNESS of an AI agent's response.

Your task is to evaluate whether the agent's response is factually accurate and logically sound.

## Task Input
{task_input}

## Expected Output (if provided)
{expected_output}

## Agent's Response
{agent_response}

## Full Transcript Context
{transcript_context}

## Evaluation Criteria

Evaluate the response on these dimensions:

1. **Factual Accuracy** (weight: 0.4)
   - Are stated facts verifiable and accurate?
   - Are claims supported by evidence when required?
   - Are there any factual errors or hallucinations?

2. **Logical Soundness** (weight: 0.3)
   - Is the reasoning logical and coherent?
   - Are conclusions properly derived from premises?
   - Are there any logical fallacies?

3. **Completeness** (weight: 0.3)
   - Does the response address all parts of the task?
   - Is the solution complete and not partial?
   - Are edge cases considered?

## Output Format

You MUST respond with a valid JSON object matching this schema:

```json
{{
  "score": <float between 0.0 and 1.0>,
  "passed": <boolean>,
  "reasoning": "<brief explanation of the score>",
  "breakdown": {{
    "factual_accuracy": <float 0.0-1.0>,
    "logical_soundness": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>
  }},
  "issues": ["<list of specific issues found, if any>"],
  "strengths": ["<list of strengths in the response, if any>"]
}}
```

Respond with ONLY the JSON object, no additional text.
```

## Example Output

```json
{
  "score": 0.85,
  "passed": true,
  "reasoning": "The response is mostly correct with accurate facts and sound logic, but misses an edge case.",
  "breakdown": {
    "factual_accuracy": 0.9,
    "logical_soundness": 0.85,
    "completeness": 0.8
  },
  "issues": ["Did not handle the empty input edge case"],
  "strengths": ["Clear explanation", "Accurate technical details"]
}
```
