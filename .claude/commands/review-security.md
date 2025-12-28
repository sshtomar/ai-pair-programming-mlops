# Security Review

Invoke the Security Engineer reviewer persona to analyze the student's code for vulnerabilities.

## Instructions

Load and adopt the persona defined in `.claude/agents/security-reviewer.md`.

Review the code in the `project/` directory with focus on:
- Input validation and sanitization
- Injection vulnerabilities (SQL, command, pickle, etc.)
- Secrets management and credential handling
- Dependency vulnerabilities
- API security (auth, rate limiting, CORS)
- Data privacy and PII handling

If the student specifies a particular area of concern, focus there. Otherwise, prioritize externally-facing code and data handling paths.

## Execution

1. Read the Security Engineer reviewer agent configuration
2. Identify security-critical code (API endpoints, data loaders, model serialization)
3. Adopt the persona and conduct the review using the specified format
4. Assess security posture and prioritize vulnerabilities by exploitability and impact

## Output

Use the response format defined in the agent configuration. Provide attack scenarios to illustrate impact. Include specific remediation code, not just descriptions of fixes.
