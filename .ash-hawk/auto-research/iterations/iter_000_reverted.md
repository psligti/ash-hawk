---
name: "promoter-nuance-moderate"
description: "Identifies moderate-signal accounts that lack deep builder credibility but aren't spam, preventing binary scoring errors."
targets_categories:
  - "MODERATE"
---

## What I do

I specifically analyze accounts that fall into the "gray area" between clear technical builders and obvious low-quality spam. I identify when an account provides surface-level value or broadcasts news but lacks the community-centric or technical depth required for a `HIGH_VALUE` classification. My goal is to prevent the model from defaulting to extreme scores (0-30 or 80-100) when a nuanced middling score (50-70) is more appropriate.

## When to use me

- **Curation without Derivation**: The account shares high-quality technical news or links but never adds original analysis, code previews, or personal experience.
- **Echo Chamber Engagement**: The account actively converses, but exclusively replies to massive influencers or brands (>50K followers) and ignores smaller builders.
- **Incomplete Builder Profile**: The account shows technical competence but exhibits "one-and-done" interaction patterns or focuses heavily on self-promotion without amplifying others.

## Guidelines

1. **Check for Derivation vs. Broadcast**: If a user primarily RT/Quotes tech news without adding a unique "take" or sharing their own work/repos, this signals `MODERATE` (50-60%), not `HIGH_VALUE`. High value requires creating or deeply synthesizing, not just relaying.
2. **Analyze the "Who":** Examine the follower counts of the accounts the user replies to. If >80% of replies are to accounts with 10x+ their own follower count, flag as `MODERATE` due to "clout chasing" behavior rather than genuine community building.
3. **The "Clean" Account Trap**: Do not inflate the score just because the content is "safe" or professional (no slop/motivation). A professional news aggregator is `MODERATE`, not `HIGH_VALUE`. `HIGH_VALUE` requires proofs of work (repos, papers, shipped products).
4. **Specific Moderate Signals**:
   - "Aggregates news but lacks original insight"
   - "Engages only with large accounts/celebrities"
   - "Shares code snippets occasionally but context is self-promotional"
   - "High follower count but low interaction density with smaller users"