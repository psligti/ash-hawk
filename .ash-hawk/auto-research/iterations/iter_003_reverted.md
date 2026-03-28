---
name: "moderate-broadcaster-filter"
description: "Downgrades technical accounts that behave solely as one-way broadcasters (no reciprocity) to the MODERATE category."
targets_categories:
  - "MODERATE"
---

## What I do

I identify and penalize accounts that demonstrate technical capability or share high-value resources (repos, code, papers) but fail to engage in two-way communication. I distinguish between "Builders in Public" (who interact) and "Broadcasters" (who only announce). I verify that the presence of technical content does not automatically qualify an account for HIGH_VALUE if it lacks social reciprocity.

## When to use me

Apply this logic when an account shows strong evidence of technical output (GitHub links, technical threads) but has:
- Very low reply-to-tweet ratio
- No evidence of amplifying others
- Follower count significantly higher than engagement/interaction depth
- A timeline consisting almost entirely of links to their own work

## Guidelines

1. **Check for Broadcasting Signals**: Look for accounts that post their own work but do not reply to comments, quote-tweet others with added value, or engage with accounts smaller than themselves.
2. **Score Capping**: If an account has "High Tech" signals but "Low Reciprocity" signals, cap the score at **MODERATE** (max 69%).
3. **Differentiation**: A HIGH_VALUE account must *give* to the community (insights, amplification, replies), not just *show* their work.
4. **Signal Weighting**: Reduce the weight of "Technical Content" signals if "Reciprocity" signals are below a threshold.