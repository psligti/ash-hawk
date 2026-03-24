---
name: goal-tracking
description: Track and manage multi-step goals with explicit progress monitoring
license: MIT
compatibility: opencode
metadata:
  audience: agents
  workflow: evaluation
---

## What I do

Track progress on multi-step goals:
- Break down complex objectives into discrete, trackable components
- Maintain explicit progress state (e.g., "Component 1/4 complete")
- Verify each component exists before marking complete
- Report final status as "GOAL COMPLETE: X/4 components" or show which are missing

- Do NOT just to the implicitly - show me:
- Component N/M: [verifying/creating/completing/done]

- Component N/M: [status]

- Repeat until all components are done

- Only mark the goal complete when ALL components are verified

## When to use me

Use this skill when:
- Implementing a feature that requires exactly N components
- Working on a task with multiple discrete steps
- Debugging issues that require systematic verification
- Any work requiring explicit progress tracking

## Guidelines

1. **Always track component count upfront**
   Start by stating: "This task has N components: [1, 2, 3, 4...]"

2. **Show progress before each component**
   Use explicit markers like:
   - "Component X/N: starting"
   - "Component X/N: creating"
   - "Component X/N: verifying"
   - "Component X/N: complete"

3. **Verify before marking complete**
   - Use bash/read/edit tools to check file/directory existence
   - Verify the tool worked successfully
   - Check for expected outputs

   DO NOT skip verification steps.

4. **Mark complete when all components are verified**
   Only use explicit "GOAL COMPLETE" status after verification.

5. **Revert partial progress on failure**
   If a component fails verification, go back and fix it before moving forward.
   Do NOT claim a component is complete until verified.
