---
name: "moderate-signal-originality"
description: "Distinguish knowledge creators from knowledge transmitters to prevent over-scoring content aggregators as HIGH_VALUE."
targets_categories:
  - "MODERATE"
---

## What I do

I evaluate the "additive value" of an account's content to distinguish between **High Value Creators** and **Moderate Curators**. I specifically look for "The Reporter vs. Builder" signal:

*   **Creators (High Value):** Generate original insights, share personal implementation logs, use "I" statements ("I built," "I tried," "I learned"), and add context to shared links.
*   **Transmitters (Moderate):** Aggregate industry news, share links to papers/repos without personal commentary/implementation, or act as a "news ticker" for the niche.

## When to use me

Apply this skill when an account posts relevant, high-quality technical topics (passing the "slop" filter) but lacks evidence of personal experience or distinct opinion. Specifically, use this when:
*   The timeline consists mostly of retweets/links to external content (e.g., "New paper from OpenAI: [link]").
*   The engagement consists of short affirmations ("Great thread", "This") on popular posts without adding new information.
*   The account acts as a reliable information source but shows no signs of being a relationship target for amplification.

## Guidelines

1.  **Check for "I" Statements:** Look for evidence of "hands-on" work. If the user shares a repo, do they show *their* usage of it, or just existence? No usage = MODERATE.
2.  **Analyze Link Commentary:** If sharing a link/article, is there specific analysis? If the tweet is just the headline + link, it is MODERATE. If it includes a counter-point or specific takeaway, it is HIGH_VALUE.
3.  **Evaluate Aggregation:** Accounts that effectively act as "News Feeds" (e.g., "Daily AI Updates") are **MODERATE**, not LOW. They have utility (information) but do not offer "Engagement potential" (reciprocal relationship).
4.  **The "Parrot" Check:** Does the user merely repeat the sentiments of larger influencers? If their timeline is an echo chamber of top-tier accounts with no unique spin, downgrade to MODERATE.