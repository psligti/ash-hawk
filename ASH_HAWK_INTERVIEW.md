# Ash Hawk Strategic Review Interview

**Purpose**: Identify waste, incomplete features, strategy gaps, and opportunities in the Ash Hawk evaluation harness.

**Instructions**: Answer questions inline. Mark with ✅ for "done/working well", ❌ for "not needed/cut", 🔄 for "needs work", and add notes as needed.

---

## Section 1: Strategy & Direction (Questions 1-15)

### 1.1 Core Purpose

1. **What problem is Ash Hawk uniquely solving that existing tools don't?**
   - [Your answer]

2. **Who are your primary users?** (Select all that apply)
   - [ ] Internal team evaluating their own agents
   - [ ] External customers running evals on your agents
   - [ ] Researchers benchmarking different models
   - [ ] CI/CD pipelines for regression testing
   - [ ] Other: ___________

3. **What's the single most valuable metric ash-hawk provides today?**
   - [Your answer]

4. **If you had to cut 50% of features, what would you absolutely keep?**
   - [Your answer]

### 1.2 Competitive Positioning

5. **How does ash-hawk compare to these alternatives?** (Better/Same/Worse/Not comparable)
   - pytest + custom scripts: _______
   - LangSmith evaluations: _______
   - Promptfoo: _______
   - Inspect AI: _______
   - AgentBench: _______

6. **What's ash-hawk's "unfair advantage" over alternatives?**
   - [Your answer]

### 1.3 Strategic Direction

7. **Is ash-hawk meant to be** (choose one):
   - [ ] A general-purpose evaluation framework (like pytest for agents)
   - [ ] A specialized tool for your specific agent evaluation needs
   - [ ] A research tool for experimentation
   - [ ] A production-grade CI/CD testing tool

8. **Should ash-hawk support evaluation of agents you DON'T control?** (e.g., third-party APIs)
   - [ ] Yes, critical
   - [ ] Nice to have
   - [ ] No, only evaluating your own agents

9. **What's the 6-month vision for ash-hawk?**
   - [Your answer]

10. **What's preventing ash-hawk from being used more widely right now?**
    - [Your answer]

11. **Is ash-hawk a product or infrastructure?** (Choose one)
    - [ ] Product (users actively choose to use it)
    - [ ] Infrastructure (invisible backbone, users don't think about it)

12. **Should ash-hawk have a web UI/dashboard?**
    - [ ] Yes, critical for adoption
    - [ ] Nice to have eventually
    - [ ] No, CLI-only is fine

13. **What's the biggest risk to ash-hawk's success?**
    - [Your answer]

14. **If ash-hawk disappeared tomorrow, what would users miss most?**
    - [Your answer]

15. **Is ash-hawk over-engineered, under-engineered, or about right?**
    - [ ] Over-engineered (too many features, too complex)
    - [ ] Under-engineered (missing critical features)
    - [ ] About right
    - Notes: ___________

---

## Section 2: Framework Architecture (Questions 16-30)

### 2.1 Dual Framework Issue

16. **Ash Hawk has two parallel evaluation frameworks:**
    - **Suite-based evals** (`ash-hawk run suite.yaml`) — traditional eval suites
    - **Scenario-based evals** (`ash-hawk scenario run`) — newer scenario framework

    **Which framework should be the primary direction?**
    - [ ] Suite-based only (deprecate scenarios)
    - [ ] Scenario-based only (deprecate suites)
    - [ ] Keep both (explain why): ___________
    - [ ] Merge them into one unified framework

17. **If merging, what's the migration path for existing suite.yaml files?**
    - [Your answer]

18. **Is the scenario framework production-ready?**
    - [ ] Yes, fully ready
    - [ ] Mostly ready, needs polish
    - [ ] Still experimental
    - [ ] Should be deprecated

19. **Which framework do actual users use today?**
    - [ ] Mostly suites
    - [ ] Mostly scenarios
    - [ ] Both equally
    - [ ] Neither much

### 2.2 Grader Ecosystem

20. **Ash Hawk has 19+ grader types. Which are ACTUALLY used?** (Check all that are valuable)

    **Deterministic Graders:**
    - [ ] `string_match` — exact/substring matching
    - [ ] `regex` — pattern matching
    - [ ] `json_schema` — structure validation
    - [ ] `test_runner` — execute test files
    - [ ] `static_analysis` — lint/type checks
    - [ ] `tool_call` — verify specific tool usage
    - [ ] `transcript` — transcript analysis
    - [ ] `diff_constraints` — diff/patch validation

    **LLM-Based Graders:**
    - [ ] `llm_judge` — general LLM evaluation
    - [ ] `llm_rubric` — rubric-based grading
    - [ ] `rubric_guard` — rubric enforcement

    **Trace Analysis Graders:**
    - [ ] `trace_schema` — trace structure validation
    - [ ] `trace_content` — trace content analysis
    - [ ] `budget_compliance` — token/time budget enforcement
    - [ ] `verify_before_done` — completion verification
    - [ ] `evidence_required` — evidence checking
    - [ ] `ordering` — action ordering validation

    **Other Graders:**
    - [ ] `composite` — weighted aggregation
    - [ ] `aggregation` — result aggregation
    - [ ] `human` — manual review
    - [ ] `cheat_detection` — detect gaming attempts

21. **Which graders should be DEPRECATED?** (List and explain)
    - [Your answer]

22. **Which graders are CRITICAL but UNDERDEVELOPED?**
    - [Your answer]

23. **Is the grader registry entry point system (`ash_hawk.graders`) being used by external plugins?**
    - [ ] Yes, actively
    - [ ] Yes, but rarely
    - [ ] No, but planned
    - [ ] No, remove it

24. **Should graders be able to:**
    - Modify transcripts before other graders see them?
      - [ ] Yes, needed
      - [ ] No, read-only
    - Emit their own metrics/telemetry?
      - [ ] Yes, needed
      - [ ] No, unified logging
    - Access external resources (APIs, databases)?
      - [ ] Yes, needed
      - [ ] No, sandboxed

25. **Are composite graders with weights actually useful, or over-engineered?**
    - [ ] Critical feature
    - [ ] Useful but could simplify
    - [ ] Over-engineered
    - Notes: ___________

### 2.3 Storage Backends

26. **Ash Hawk supports 4 storage backends. Which are actually used?**
    - [ ] File storage (local filesystem)
    - [ ] SQLite
    - [ ] PostgreSQL
    - [ ] S3

27. **Which backends should be REMOVED?**
    - [Your answer]

28. **Is the storage abstraction worth the complexity, or should it be file-only?**
    - [ ] Keep all backends
    - [ ] File + SQLite only
    - [ ] File-only is sufficient
    - Notes: ___________

29. **Do you need real-time storage queries, or is batch reporting sufficient?**
    - [ ] Real-time queries needed
    - [ ] Batch reporting is fine

30. **Should storage results be exportable to external systems?** (e.g., BigQuery, Snowflake)
    - [ ] Yes, critical
    - [ ] Nice to have
    - [ ] No need

---

## Section 3: Feature Completeness (Questions 31-50)

### 3.1 Phase 2 Roadmap

31. **PHASE2.md lists uncompleted features. Which are STILL WANTED?**

    **conftest.yaml Loading:**
    - [ ] Still wanted
    - [ ] No longer needed

    **pyproject.toml Configuration:**
    - [ ] Still wanted
    - [ ] No longer needed

    **Suite Discovery (-k, -m filtering):**
    - [ ] Still wanted
    - [ ] No longer needed

32. **If keeping conftest.yaml, what's the priority?** (1-5, 5=highest)
    - Priority: ___
    - Notes: ___________

33. **Is pytest-like discovery (-k pattern, -m markers) critical for your workflow?**
    - [ ] Critical
    - [ ] Nice to have
    - [ ] Not needed

34. **What Phase 2 feature would provide the MOST value?**
    - [Your answer]

### 3.2 Fast Evals

35. **Are "Fast Evals" (lightweight regression tests without full agent execution) actually used?**
    - [ ] Yes, frequently
    - [ ] Yes, occasionally
    - [ ] Rarely
    - [ ] Never
    - [ ] Didn't know they existed

36. **Should Fast Evals be expanded or deprecated?**
    - [ ] Expand with more features
    - [ ] Keep as-is
    - [ ] Deprecate

37. **What's missing from Fast Evals that would make them more useful?**
    - [Your answer]

### 3.3 Calibration

38. **Is the calibration module (ECE, Brier scores) actively used?**
    - [ ] Yes, critical for grader tuning
    - [ ] Yes, occasionally
    - [ ] No
    - [ ] Didn't know it existed

39. **Does calibration actually change how you configure graders?**
    - [ ] Yes, we adjust thresholds based on calibration
    - [ ] Sometimes
    - [ ] No, it's informational only
    - [ ] Don't use it

40. **Should calibration be:**
    - [ ] Automatic after every run
    - [ ] Manual CLI command (current)
    - [ ] Integrated into reports
    - [ ] All of the above

41. **What calibration features are MISSING?**
    - [Your answer]

### 3.4 Reporting

42. **Which report formats are actually used?**
    - [ ] HTML reports
    - [ ] JSON export
    - [ ] CLI summary output
    - [ ] Gap scorecard
    - [ ] CI threshold enforcement

43. **Is the HTML report valuable enough to maintain?**
    - [ ] Yes, critical
    - [ ] Yes, but could simplify
    - [ ] No, JSON-only is fine

44. **Do you need trend analysis across multiple runs?**
    - [ ] Yes, critical
    - [ ] Nice to have
    - [ ] No need

45. **Should ash-hawk integrate with external reporting tools?** (Grafana, Datadog, etc.)
    - [ ] Yes
    - [ ] No
    - Notes: ___________

### 3.5 Templates

46. **Are task templates (coding, conversational, research) used?**
    - [ ] Yes, frequently
    - [ ] Yes, occasionally
    - [ ] Rarely
    - [ ] Never

47. **Should templates be expanded or removed?**
    - [ ] Expand
    - [ ] Keep as-is
    - [ ] Remove

48. **What template types are MISSING?**
    - [Your answer]

### 3.6 Agent Adapters

49. **Which agent adapters are actually used?**
    - [ ] dawn-kestrel SDK
    - [ ] mock_adapter
    - [ ] coding_agent_subprocess
    - [ ] Other: ___________

50. **Should ash-hawk support evaluating:**
    - **Anthropic agents directly via API?**
      - [ ] Yes
      - [ ] No
    - **OpenAI assistants?**
      - [ ] Yes
      - [ ] No
    - **Custom HTTP endpoints?**
      - [ ] Yes
      - [ ] No

---

## Section 4: Metrics & Telemetry (Questions 51-65)

### 4.1 Current Metrics

51. **Are the current metrics sufficient?** (Rate 1-5, 5=very sufficient)
    - Pass/fail rates: ___
    - Token usage tracking: ___
    - Latency metrics: ___
    - Cost tracking: ___
    - Calibration scores: ___
    - Confidence intervals: ___

52. **What metrics are you NOT using?**
    - [Your answer]

53. **What metrics are MISSING that you need?**
    - [Your answer]

54. **Do you need real-time metrics during execution, or is post-run analysis sufficient?**
    - [ ] Real-time metrics needed
    - [ ] Post-run is fine

### 4.2 Telemetry & Observability

55. **Should ash-hawk emit metrics to external systems?**
    - [ ] OpenTelemetry integration
    - [ ] StatsD/DogStatsD
    - [ ] Prometheus
    - [ ] Custom webhooks
    - [ ] None, local logs only

56. **Do you need alerting on evaluation failures?**
    - [ ] Yes, critical for CI
    - [ ] Nice to have
    - [ ] No need

57. **Should ash-hawk log structured events for external analysis?**
    - [ ] Yes, structured JSON logs
    - [ ] Yes, to a database
    - [ ] No, current logging is fine

58. **Is trace replay valuable for debugging?**
    - [ ] Yes, use it frequently
    - [ ] Yes, but rarely
    - [ ] No
    - [ ] Didn't know it existed

### 4.3 Historical Analysis

59. **Do you need to compare runs across time?**
    - [ ] Yes, for regression detection
    - [ ] Yes, for trend analysis
    - [ ] No

60. **Should ash-hawk detect regressions automatically?**
    - [ ] Yes, with configurable thresholds
    - [ ] Yes, but simple diff only
    - [ ] No

61. **What historical queries would be valuable?**
    - [Your answer]

### 4.4 Data Export

62. **Do you need to export results to external systems?**
    - [ ] BigQuery/Snowflake
    - [ ] S3 for archival
    - [ ] CSV/Excel for analysis
    - [ ] None

63. **Is the current JSON export sufficient?**
    - [ ] Yes
    - [ ] No, needs: ___________

64. **Should ash-hawk support streaming results during execution?**
    - [ ] Yes
    - [ ] No

65. **What's the biggest gap in observability right now?**
    - [Your answer]

---

## Section 5: Waste & Dead Code (Questions 66-80)

### 5.1 Unused Features

66. **Which features feel like "nice to have" that nobody uses?**
    - [Your answer]

67. **Are there graders you've never used?**
    - [Your answer]

68. **Are there CLI commands you've never used?**
    - [Your answer]

69. **Are there configuration options that are never changed from defaults?**
    - [Your answer]

70. **What code feels like it was built "just in case"?**
    - [Your answer]

### 5.2 Over-Engineering

71. **Is the codebase too abstract?** (Interfaces for everything, too many layers)
    - [ ] Yes, over-abstracted
    - [ ] About right
    - [ ] No, could use more abstraction

72. **Are there multiple ways to do the same thing?** (Confusing)
    - [Your answer]

73. **What's the most complex part of the codebase that could be simplified?**
    - [Your answer]

74. **Is Pydantic strict mode (`extra="forbid"`) causing more pain than value?**
    - [ ] Yes, too restrictive
    - [ ] No, worth the strictness
    - Notes: ___________

75. **Are the type hints helpful or noisy?**
    - [ ] Helpful, keep them
    - [ ] Somewhat helpful
    - [ ] Noisy, reduce them

### 5.3 Incomplete Features

76. **One test is skipped due to "event loop timing issues" (test_trial.py:308). Is cancellation handling important?**
    - [ ] Yes, fix it
    - [ ] Nice to have
    - [ ] Remove the test

77. **What features are 80% done but never finished?**
    - [Your answer]

78. **What features were started and abandoned?**
    - [Your answer]

79. **Are there TODO comments that have been around for months?**
    - [Your answer]

80. **What would you DELETE if you had no backward compatibility concerns?**
    - [Your answer]

---

## Section 6: User Experience (Questions 81-90)

### 6.1 CLI Experience

81. **Rate the CLI experience** (1-5, 5=excellent)
    - Discovery (finding commands): ___
    - Documentation (help text): ___
    - Error messages: ___
    - Progress feedback: ___
    - Output formatting: ___

82. **What CLI pain points exist?**
    - [Your answer]

83. **Is `ash-hawk init` useful or do people copy existing files?**
    - [ ] Useful, use it
    - [ ] Copy existing files
    - [ ] Start from scratch

84. **Should ash-hawk have an interactive mode (TUI)?**
    - [ ] Yes
    - [ ] No

### 6.2 Learning Curve

85. **How long does it take a new user to run their first eval?**
    - [ ] < 5 minutes
    - [ ] 5-30 minutes
    - [ ] 30 minutes - 2 hours
    - [ ] > 2 hours

86. **What's the most confusing part of ash-hawk?**
    - [Your answer]

87. **Is the documentation sufficient?**
    - [ ] Yes
    - [ ] No, needs: ___________

88. **What questions do users ask most frequently?**
    - [Your answer]

### 6.3 Integration

89. **How easy is it to integrate ash-hawk into CI/CD?**
    - [ ] Very easy
    - [ ] Somewhat easy
    - [ ] Difficult
    - [ ] Haven't tried

90. **What CI/CD features are missing?**
    - [Your answer]

---

## Section 7: Priorities & Roadmap (Questions 91-100)

### 7.1 Top Priorities

91. **What are the TOP 3 things to work on in the next quarter?**
    1. ___________
    2. ___________
    3. ___________

92. **What should NOT be worked on?**
    - [Your answer]

93. **What's the BIGGEST quick win?** (High impact, low effort)
    - [Your answer]

94. **What's the BIGGEST investment?** (High impact, high effort)
    - [Your answer]

### 7.2 Success Criteria

95. **How do you measure if ash-hawk is successful?**
    - [Your answer]

96. **What would make you consider deprecating ash-hawk?**
    - [Your answer]

97. **What's the MINIMUM feature set needed for ash-hawk to be useful?**
    - [Your answer]

98. **If you could only keep ONE grader type, which would it be?**
    - [Your answer]

99. **What's the most underutilized feature that SHOULD be used more?**
    - [Your answer]

100. **Final thoughts: What's missing from this interview?**
    - [Your answer]

---

## Summary Section

After completing, please also provide:

**Top 3 features to KEEP:**
1. ___________
2. ___________
3. ___________

**Top 3 features to CUT:**
1. ___________
2. ___________
3. ___________

**Top 3 features to ADD:**
1. ___________
2. ___________
3. ___________

**Biggest strategic concern:**
- [Your answer]

**Overall health rating** (1-10, 10=healthy, focused, valuable):
- Rating: ___
- Notes: ___________

---

*Interview generated for Ash Hawk v0.1.0*
*Based on codebase analysis of 85 source files, 69 test files, and 19 grader types*
