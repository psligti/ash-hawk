---
name: "ai-slop-detection"
description: "Distinguishes between genuine technical discourse and low-effort 'AI slop' or engagement farming to properly classify accounts as LOW value."
targets_categories:
  - "LOW"
---

## What I do

I analyze tweet content for superficial patterns often found in "Nice-thought" accounts that should be classified as **LOW** or **SKIP**. Specifically, I detect formulaic threads, generic motivational quotes paired with buzzwords, and engagement baiting. I verify if technical claims are backed by specific evidence (code, screenshots, papers) or if they are vague platitudes.

## When to use me

Apply this skill when evaluating an account that frequently posts about AI/tech trends but lacks specific technical depth or unique insights. This is crucial when the account has a moderate-to-high follower count, which might otherwise trigger a default assumption of credibility, or when the account uses "engagement farming" language.

## Guidelines

- **Flag Formulaic Content**: Identify threads that follow a rigid "hot take" -> "generic advice" -> "CTA" structure without substance.
- **Detect Buzzword Salad**: Penalize tweets high in buzzwords (e.g., "game-changer," "unleash," "paradigm shift") but low in technical specificity.
- **Check for Reciprocity**: Heavily weight the lack of engagement with smaller accounts or lack of amplification of others' work.
- **Evidence Verification**: If an account claims to have "shipped" something or broken news, look for links, repos, or proof. Absence of proof in >3 recent "claim" tweets indicates LOW value.
- **Volume vs. Value**: A high frequency of posts with no interaction (replies/quotes) from the user usually indicates a broadcast bot rather than a community member.