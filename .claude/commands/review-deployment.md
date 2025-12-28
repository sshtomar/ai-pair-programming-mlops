# SRE Deployment Review

Invoke the SRE reviewer persona to analyze the student's code for production readiness.

## Instructions

Load and adopt the persona defined in `.claude/agents/sre-reviewer.md`.

Review the code in the `project/` directory with focus on:
- Error handling and resilience patterns
- Logging, metrics, and observability
- Resource usage and scaling considerations
- Dependency management and security
- Deployment safety (health checks, graceful shutdown)

If the student specifies a particular service or deployment configuration, focus there. Otherwise, review API endpoints, Dockerfiles, and deployment configurations.

## Execution

1. Read the SRE reviewer agent configuration
2. Identify deployment-relevant code (FastAPI apps, Dockerfiles, configs, etc.)
3. Adopt the persona and conduct the review using the specified format
4. Assess operational readiness with a score and prioritized risks

## Output

Use the response format defined in the agent configuration. Think about the on-call engineer who will maintain this at 3 AM. Provide runbook-style mitigation steps.
