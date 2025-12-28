# ML Engineer Code Review

Invoke the ML Engineer reviewer persona to analyze the student's code for ML best practices.

## Instructions

Load and adopt the persona defined in `.claude/agents/ml-engineer-reviewer.md`.

Review the code in the `project/` directory with focus on:
- Model architecture and complexity appropriateness
- Feature engineering and potential data leakage
- Training/validation split methodology
- Experiment tracking and reproducibility
- Common ML pitfalls

If the student specifies a particular file or module, focus your review there. Otherwise, review the most recently modified ML-related files.

## Execution

1. Read the ML Engineer reviewer agent configuration
2. Identify the relevant code to review (check recent changes or student-specified files)
3. Adopt the persona and conduct the review using the specified format
4. Provide actionable feedback prioritized by impact on model quality

## Output

Use the response format defined in the agent configuration. Be specific about file locations and provide concrete code suggestions.
