---
name: "moderate-reciprocity-check"
description: "Identifies accounts that offer technical value but lack reciprocal engagement behaviors, correctly classifying them as MODERATE instead of HIGH_VALUE or SKIP."
targets_categories:
  - "MODERATE"
---

## What I do

I evaluate accounts that display "High Value" signals (sharing code, repos, technical insights) but fail to demonstrate "Community Value" signals (amplifying others, engaging small accounts, multi-turn conversation). My specific function is to detect the "Broadcast-Only Expert" or "Clout Chaser"—users who provide intellectual value but do not contribute to the reciprocal ecosystem of the niche. I downgrade these accounts from HIGH_VALUE to MODERATE by heavily penalizing one-sided engagement patterns.

## When to use me

Use me when an account possesses undeniable technical merit (e.g., they ship code, share papers, or write deep technical threads) but their engagement profile feels cold, self-promotional, or insular. specifically when the agent is wavering between giving a HIGH score because of content quality versus a LOW score because of social behavior.

## Guidelines

1.  **The Reply-Follower Ratio**: Analyze who the account replies to. If >80% of replies are directed at accounts with >50K followers (verified users or big names), this is a **MODERATE** signal. They are clout chasing, not community building.
2.  **Amplification Balance**: Compare the volume of self-replies/threads vs. retweets/quotes of others. If the account almost exclusively amplifies their own content ("Just shipped v2...", "Thread on how I..."), cap the score at **65%**.
3.  **The "Thread Farm" Check**: If technical content is consistently gated behind "Subscribe to read more" or "Follow for part 2", treat it as engagement farming. This forces a **MODERATE** or **SKIP** classification regardless of the thread's technical depth.
4.  **Multi-turn Verification**: Look for evidence of back-and-forth conversation. If the account only posts initial replies and never responds to responses, they are a "One-and-Done" broadcaster. This disqualifies them from HIGH_VALUE and places them firmly in **MODERATE** or **LOW**.
5.  **Scoring Logic**:
    *   Strong Tech Content + Zero Community Reciprocity = **60-70% (MODERATE)**
    *   Strong Tech Content + High Clout Chasing = **50-60% (MODERATE)**
    *   Do not score >70% if the account does not demonstrably interact with smaller creators.