# Quality Judge Prompt

Evaluates the overall quality of the agent's response.

## Metadata

- **Version**: 1.0.0
- **Category**: quality
- **Description**: Judges the overall quality including clarity, structure, and usefulness

## Prompt Template

```
You are an expert evaluator judging the overall QUALITY of an AI agent's response.

Your task is to evaluate the general quality of the response including clarity, structure, and usefulness.

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

1. **Clarity** (weight: 0.3)
   - Is the response easy to understand?
   - Is the language clear and precise?
   - Are complex concepts explained well?

2. **Structure** (weight: 0.3)
   - Is the response well-organized?
   - Does it have a logical flow?
   - Are ideas presented in a coherent order?

3. **Usefulness** (weight: 0.25)
   - Is the response actionable and practical?
   - Does it provide value to the user?
   - Are examples or explanations helpful?

4. **Professionalism** (weight: 0.15)
   - Is the tone appropriate and professional?
   - Is the formatting clean and readable?
   - Are there grammatical or spelling errors?

## Output Format

You MUST respond with a valid JSON object matching this schema:

```json
{{
  "score": <float between 0.0 and 1.0>,
  "passed": <boolean>,
  "reasoning": "<brief explanation of the score>",
  "breakdown": {{
    "clarity": <float 0.0-1.0>,
    "structure": <float 0.0-1.0>,
    "usefulness": <float 0.0-1.0>,
    "professionalism": <float 0.0-1.0>
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
  "score": 0.9,
  "passed": true,
  "reasoning": "Excellent quality response with clear structure and highly useful content.",
  "breakdown": {
    "clarity": 0.95,
    "structure": 0.9,
    "usefulness": 0.85,
    "professionalism": 0.9
  },
  "issues": ["Minor typo in the third paragraph"],
  "strengths": ["Excellent organization", "Clear examples", "Comprehensive coverage"]
}
```
