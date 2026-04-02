# `ash-hawk improve` Command Flow

## Overview

The `improve` command runs iterative auto-research cycles that discover agent skill/policy/tool/agent files, generate LLM-proposed improvements, evaluate them against scenario suites, and keep only changes that produce a measurable score increase.

Two modes exist:
- **`run`** — single-target improvement cycle
- **`enhanced-run`** — multi-target parallel improvement with intent analysis, knowledge promotion, lever search, and skill cleanup

---

## Top-Level Command Registration

```mermaid
flowchart TD
    CLI["ash-hawk CLI<br/>(click.Group)"] -->|subcommand| RUN["run"]
    CLI -->|subcommand: improve| AR["auto-research<br/>(click.Group)"]
    CLI -->|subcommand| THIN["thin"]

    AR -->|subcommand: run| CYCLE["Single-target<br/>improvement cycle"]
    AR -->|subcommand: enhanced-run| ECYCLE["Enhanced<br/>multi-target cycle"]

    style CLI fill:#1a1a2e,color:#e0e0e0
    style AR fill:#16213e,color:#e0e0e0
```

**Entry point**: `ash_hawk/cli/main.py` registers `auto_research` as `improve`.

---

## Mode 1: `improve run` — Single-Target Cycle

```mermaid
flowchart TD
    START(["ash-hawk improve run<br/>-s scenario.yaml"]) --> PARSE["Parse CLI options<br/>(iterations, threshold,<br/>rate-limit, thin-bridge,<br/>candidate-target-updates)"]
    PARSE --> VALIDATE["_validate_auto_research_scenarios()<br/>Ensure YAML has: schema_version,<br/>id, description, sut, inputs, graders<br/>Disallow: tools, budget, expectations"]
    VALIDATE --> LLM["_create_llm_client()<br/>From dawn-kestrel settings"]
    LLM --> DISCOVER["Discover Improvement Target"]

    subgraph DISCOVER["Target Discovery"]
        direction TB
        D1["Search order:<br/>Skill → Tool → Policy → Agent"]
        D2["_find_skill_file()<br/>.dawn-kestrel/skills/*/SKILL.md"]
        D3["_find_primary_tool()<br/>.dawn-kestrel/tools/*/TOOL.md"]
        D4["_find_policy_file()<br/>.dawn-kestrel/policies/*/POLICY.md"]
        D5["_find_agent_file()<br/>.dawn-kestrel/agents/*/AGENT.md"]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    DISCOVER --> INJECTOR["Create DawnKestrelInjector<br/>+ ImprovementTarget"]
    INJECTOR --> BASELINE

    subgraph BASELINE["Baseline Evaluation"]
        direction TB
        B1["run_scenarios_async()<br/>or ThinScenarioRunner"]
        B2["Collect transcripts<br/>+ category_scores"]
        B1 --> B2
    end

    BASELINE --> LOOP

    subgraph LOOP["Iteration Loop (N iterations)"]
        direction TB
        I1["Evaluate for transcripts<br/>(cache on iter 0)"]
        I2["_filter_valid_transcripts()<br/>TranscriptValidityGrader"]
        I3["LLM: generate_improvement()<br/>Focus on weakest category"]
        I4["Save proposed content<br/>to target file"]
        I5["Evaluate proposal<br/>against scenarios"]
        I6{"delta >= threshold?"}

        I1 --> I2 --> I3 --> I4 --> I5 --> I6
        I6 -->|Yes: KEEP| I7["Update current_score<br/>Reset consecutive_failures"]
        I6 -->|No: REVERT| I8["Restore original content<br/>Delete if renamed target"]

        I7 --> CONV{"Converged?"}
        I8 --> CONV
    end

    CONV -->|No: next iteration| LOOP
    CONV -->|Yes| SAVE["Save CycleResult JSON<br/>.ash-hawk/auto-research/cycles/"]
    LOOP -->|Max iterations| SAVE

    SAVE --> REPORT["Print final report:<br/>Baseline → Final score,<br/>kept/reverted counts"]

    style LOOP fill:#0a1628,color:#e0e0e0
    style BASELINE fill:#1a1a2e,color:#e0e0e0
```

---

## Single Iteration Detail

```mermaid
flowchart TD
    subgraph ITER["run_iteration()"]
        direction TB
        T0["Get transcripts<br/>(cached or fresh eval)"]
        T0 --> T1

        subgraph FILTER["Transcript Filtering"]
            direction LR
            T1["TranscriptValidityGrader<br/>.grade()"] --> T2["Valid transcripts"]
            T1 --> T3["Error signals"]
        end

        T2 --> GEN

        subgraph GEN["LLM Improvement Generation"]
            direction TB
            G1["Build prompt with:<br/>• Current content (6K chars)<br/>• Category scores table<br/>• Transcript excerpts (8K chars)<br/>• Failed proposals (avoid repeats)<br/>• Error signals<br/>• Existing skills (avoid dups)"]
            G2["_call_llm()<br/>temperature = 0.3 + 0.1 * failures<br/>(capped at 1.0)"]
            G3["_extract_content()<br/>Parse markdown from response"]
            G1 --> G2 --> G3
        end

        G3 --> SAVE

        subgraph SAVE["Apply & Evaluate"]
            direction TB
            S1["extract_skill_name()<br/>Parse YAML frontmatter"]
            S2["target.save_content(improved)"]
            S3["_run_evaluation()<br/>Score proposal"]
            S4["Compute delta = score_after - score_before"]
            S1 --> S2 --> S3 --> S4
        end

        S4 --> DECIDE{"delta >= threshold?"}
        DECIDE -->|Yes| KEEP["Return IterationResult<br/>applied=True"]
        DECIDE -->|No| REVERT["Restore original content<br/>Return IterationResult<br/>applied=False"]
    end

    style GEN fill:#16213e,color:#e0e0e0
    style SAVE fill:#1a1a2e,color:#e0e0e0
```

---

## Hill-Climb Mode (candidate_target_updates > 1)

When `--candidate-target-updates` > 1, the iteration evaluates multiple targets and picks the best:

```mermaid
flowchart TD
    START["Discover all targets<br/>(skills, tools, policies, agents)"]
    START --> DEDUPE["_dedupe_candidates()<br/>Limit to max_candidate_updates"]
    DEDUPE --> PAR

    subgraph PAR["Parallel Candidate Evaluation"]
        direction TB
        C1["Candidate A:<br/>generate + eval"]
        C2["Candidate B:<br/>generate + eval"]
        C3["Candidate C:<br/>generate + eval"]
    end

    PAR --> PICK["Pick best outcome<br/>max by (delta, score_after)"]
    PICK --> DECIDE{"best.delta >= threshold?"}
    DECIDE -->|Yes| APPLY["Save best content<br/>Return applied=True"]
    DECIDE -->|No| REVERT["Revert all<br/>Return applied=False"]

    style PAR fill:#0a1628,color:#e0e0e0
```

---

## Mode 2: `improve enhanced-run` — Multi-Target Cycle

```mermaid
flowchart TD
    START(["ash-hawk improve enhanced-run<br/>-s scenarios/*.yaml"]) --> CONFIG["Build EnhancedCycleConfig<br/>(multi-target, intent analysis,<br/>knowledge promotion, cleanup)"]
    CONFIG --> DISCOVER["TargetDiscovery.discover_all_targets()<br/>Find all skills/tools/policies/agents"]
    DISCOVER --> HAS{"Targets found?"}
    HAS -->|No| ERR["Return error"]
    HAS -->|Yes| INTENT

    subgraph INTENT["Intent Analysis (optional)"]
        direction TB
        IA1["Run baseline evaluation<br/>run_scenarios_async()"]
        IA2["IntentAnalyzer.analyze_transcripts()<br/>Extract tool usage patterns,<br/>decision patterns, failure patterns"]
        IA1 --> IA2
    end

    INTENT --> MULTI

    subgraph MULTI["Multi-Target Parallel Improvement"]
        direction TB
        M1["MultiTargetCycleRunner<br/>asyncio.Semaphore(parallel_targets)"]
        M2["For each target:<br/>run_cycle() with explicit_targets"]
        M3["Aggregate into MultiTargetResult<br/>Weighted avg improvement"]
        M1 --> M2 --> M3
    end

    MULTI --> LEVER

    subgraph LEVER["Lever Matrix Search (optional)"]
        direction TB
        L1["LeverMatrixSearch<br/>Explore agent/skills/tools/<br/>context_strategy/prompt_preset space"]
        L2["Find best configuration"]
        L1 --> L2
    end

    LEVER --> PROMO

    subgraph PROMO["Knowledge Promotion"]
        direction TB
        P1["For each applied iteration<br/>across all targets:"]
        P2["KnowledgePromoter.should_promote()<br/>Check: min_improvement,<br/>consecutive_successes,<br/>stability"]
        P3["promote_lesson()<br/>Save to .ash-hawk/lessons/<br/>+ note-lark MCP"]
        P1 --> P2 --> P3
    end

    PROMO --> CLEANUP

    subgraph CLEANUP["Skill Cleanup"]
        direction TB
        CL1["SkillCleaner.get_baseline_skills()"]
        CL2["For each new skill created<br/>during cycle:"]
        CL3["evaluate_skill_effectiveness()<br/>Did it help or hurt?"]
        CL4["Delete low-value skills"]
        CL1 --> CL2 --> CL3 --> CL4
    end

    CLEANUP --> RESULT["EnhancedCycleResult<br/>+ Print tables + Save JSON"]

    style MULTI fill:#0a1628,color:#e0e0e0
    style PROMO fill:#16213e,color:#e0e0e0
    style CLEANUP fill:#1a1a2e,color:#e0e0e0
```

---

## Convergence Detection

```mermaid
flowchart TD
    START["_check_convergence_result()"] --> CD["ConvergenceDetector.check()"]
    CD --> C1{"Score variance < 0.001<br/>over last 5 iterations?"}
    C1 -->|Yes| PLATEAU["PLATEAU<br/>Scores stabilized"]
    C1 -->|No| C2{"No improvement > 0.005<br/>for 10+ iterations?"}
    C2 -->|Yes| NOIMP["NO_IMPROVEMENT<br/>Stuck in local optimum"]
    C2 -->|No| C3{"Score decreased for<br/>3+ consecutive iterations?"}
    C3 -->|Yes| REG["REGRESSION<br/>Getting worse"]
    C3 -->|No| CONTINUE["Not converged<br/>Continue iterating"]

    style PLATEAU fill:#2d6a4f,color:#e0e0e0
    style NOIMP fill:#e76f51,color:#e0e0e0
    style REG fill:#d62828,color:#e0e0e0
    style CONTINUE fill:#457b9d,color:#e0e0e0
```

---

## Knowledge Promotion Flow

```mermaid
flowchart TD
    START["Applied iteration found"] --> SP["should_promote()"]
    SP --> C1{"delta >=<br/>min_improvement (0.05)?"}
    C1 -->|No| SKIP1["Skip"]
    C1 -->|Yes| C2{"iteration.applied?"}
    C2 -->|No| SKIP2["Skip"]
    C2 -->|Yes| C3{"consecutive_successes >= 3?"}
    C3 -->|No| SKIP3["Skip"]
    C3 -->|Yes| C4{"Recent regression<br/>< max_regression (0.02)?"}
    C4 -->|No| SKIP4["Skip"]
    C4 -->|Yes| PROMOTE["promote_lesson()"]

    PROMOTE --> LOCAL["_save_local()<br/>.ash-hawk/lessons/{id}.json"]
    LOCAL --> NL{"note-lark enabled?"}
    NL -->|Yes| MCP["promote_to_note_lark()<br/>memory_structured()"]
    NL -->|No| DONE["Done"]
    MCP --> DONE

    style PROMOTE fill:#2d6a4f,color:#e0e0e0
    style SKIP1 fill:#e0e0e0,color:#333
    style SKIP2 fill:#e0e0e0,color:#333
    style SKIP3 fill:#e0e0e0,color:#333
    style SKIP4 fill:#e0e0e0,color:#333
```

---

## Skill Cleanup Decision

```mermaid
flowchart TD
    START["New skill created during cycle"] --> EVAL["evaluate_skill_effectiveness()"]
    EVAL --> C1{"Appeared in applied<br/>iteration with positive delta?"}
    C1 -->|Yes| KEEP["Keep skill"]
    C1 -->|No| C2{"Associated with<br/>regression iteration?"}
    C2 -->|Yes| C3{"remove_negative_impact<br/>enabled?"}
    C2 -->|No| C4{"remove_unused<br/>enabled?"}
    C3 -->|Yes| DELETE["Delete skill"]
    C3 -->|No| KEEP2["Keep skill"]
    C4 -->|Yes| DELETE2["Delete skill"]
    C4 -->|No| KEEP3["Keep skill"]

    style KEEP fill:#2d6a4f,color:#e0e0e0
    style KEEP2 fill:#2d6a4f,color:#e0e0e0
    style KEEP3 fill:#2d6a4f,color:#e0e0e0
    style DELETE fill:#d62828,color:#e0e0e0
    style DELETE2 fill:#d62828,color:#e0e0e0
```

---

## Evaluation Paths

```mermaid
flowchart TD
    EVAL["_run_evaluation()"] --> BRIDGE{"thin_bridge<br/>enabled?"}
    BRIDGE -->|Yes| THIN["ThinScenarioRunner<br/>run_with_grading()<br/>per-scenario parallelism<br/>via asyncio.Semaphore"]
    BRIDGE -->|No| STANDARD["run_scenarios_async()<br/>Standard scenario runner"]

    THIN --> SCORE["Mean score +<br/>transcripts +<br/>category_scores"]
    STANDARD --> SCORE

    style EVAL fill:#16213e,color:#e0e0e0
```

---

## Module Map

```mermaid
flowchart LR
    subgraph CLI["CLI Layer"]
        main["main.py"]
        cli["cli.py"]
        ecli["enhanced_cli.py"]
    end

    subgraph CORE["Core Runners"]
        cr["cycle_runner.py<br/>run_cycle()"]
        ecr["enhanced_cycle_runner.py<br/>run_enhanced_cycle()"]
        mtr["multi_target_runner.py<br/>MultiTargetCycleRunner"]
    end

    subgraph SUPPORT["Support Modules"]
        llm["llm.py<br/>generate_improvement()"]
        conv["convergence.py<br/>ConvergenceDetector"]
        kp["knowledge_promotion.py<br/>KnowledgePromoter"]
        sc["skill_cleanup.py<br/>SkillCleaner"]
        td["target_discovery.py<br/>TargetDiscovery"]
        ia["intent_analyzer.py<br/>IntentAnalyzer"]
        lm["lever_matrix.py<br/>LeverMatrixSearch"]
    end

    subgraph TYPES["Types"]
        types["types.py<br/>CycleResult, IterationResult,<br/>EnhancedCycleConfig, ..."]
    end

    main --> cli
    main --> ecli
    cli --> cr
    ecli --> ecr
    ecr --> mtr
    mtr --> cr
    cr --> llm
    cr --> conv
    ecr --> kp
    ecr --> sc
    ecr --> td
    ecr --> ia
    ecr --> lm

    cr -.-> types
    ecr -.-> types
    mtr -.-> types
    conv -.-> types
    kp -.-> types

    style CLI fill:#1a1a2e,color:#e0e0e0
    style CORE fill:#0a1628,color:#e0e0e0
    style SUPPORT fill:#16213e,color:#e0e0e0
    style TYPES fill:#2d2d44,color:#e0e0e0
```

---

## Key File Paths

| Artifact | Location |
|---|---|
| Cycle results | `.ash-hawk/auto-research/cycles/cycle_{agent}_{timestamp}.json` |
| Iteration artifacts | `.ash-hawk/auto-research/iterations/iter_XXX_{kept\|reverted}.md` |
| Enhanced results | `.ash-hawk/enhanced-auto-research/` |
| Promoted lessons | `.ash-hawk/lessons/{lesson_id}.json` |
| Skill content | `.dawn-kestrel/skills/{name}/SKILL.md` |
| Tool content | `.dawn-kestrel/tools/{name}/TOOL.md` |
| Policy content | `.dawn-kestrel/policies/{name}/POLICY.md` |
| Agent content | `.dawn-kestrel/agents/{name}/AGENT.md` |
