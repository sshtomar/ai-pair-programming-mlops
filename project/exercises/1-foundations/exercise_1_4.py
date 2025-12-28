"""
Exercise 1.4: Docker Basics
Difficulty: ★★★
Topic: Dockerfile validation and container best practices

Instructions:
This exercise has two parts:

PART A - Guided Stub (validate_dockerfile):
Docker containers are the foundation of reproducible ML deployments. But poorly
written Dockerfiles create security risks, slow builds, and large images.

Write a function that analyzes a Dockerfile and reports common issues.

PART B - Debug Exercise (fix the Dockerfile):
A sample Dockerfile is provided with multiple best-practice violations.
Identify and explain the issues, then provide a fixed version.

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

from pathlib import Path


# =============================================================================
# PART A: Guided Stub - Dockerfile Validator
# =============================================================================

class DockerfileIssue:
    """Represents an issue found in a Dockerfile."""

    def __init__(self, line_number: int, severity: str, message: str, suggestion: str):
        """
        Args:
            line_number: 1-indexed line where issue was found (0 for file-level issues)
            severity: "error", "warning", or "info"
            message: Description of the issue
            suggestion: How to fix it
        """
        self.line_number = line_number
        self.severity = severity
        self.message = message
        self.suggestion = suggestion

    def __repr__(self) -> str:
        loc = f"line {self.line_number}" if self.line_number > 0 else "file"
        return f"[{self.severity.upper()}] {loc}: {self.message}"


def validate_dockerfile(dockerfile_path: str | Path) -> list[DockerfileIssue]:
    """
    Analyze a Dockerfile for common issues and best practices violations.

    Checks to implement:
    1. Running as root (no USER instruction)
    2. No .dockerignore file in same directory
    3. Using 'latest' tag for base image
    4. COPY/ADD before installing dependencies (poor layer caching)
    5. Multiple RUN commands that could be combined
    6. No HEALTHCHECK instruction (for production images)
    7. Using ADD when COPY would suffice
    8. Not pinning package versions in apt-get/pip install

    Args:
        dockerfile_path: Path to the Dockerfile

    Returns:
        List of DockerfileIssue objects describing problems found

    Raises:
        FileNotFoundError: If Dockerfile doesn't exist

    Example:
        >>> issues = validate_dockerfile("path/to/Dockerfile")
        >>> for issue in issues:
        ...     print(issue)
        [WARNING] file: No .dockerignore file found
        [ERROR] line 1: Using 'latest' tag for base image
    """
    dockerfile_path = Path(dockerfile_path)

    # TODO: Step 1 - Read the Dockerfile
    # Open and read all lines
    # Let FileNotFoundError propagate naturally

    # TODO: Step 2 - Initialize issues list and tracking variables
    issues: list[DockerfileIssue] = []
    # Track: has_user_instruction, has_healthcheck, run_count, etc.

    # TODO: Step 3 - Check for .dockerignore
    # Look for .dockerignore in the same directory as Dockerfile
    # If missing, add a warning-level issue

    # TODO: Step 4 - Parse each line and check for issues
    # For each line:
    #   - Skip comments (lines starting with #) and empty lines
    #   - Check FROM for :latest or no tag
    #   - Check for USER instruction (set has_user flag)
    #   - Count RUN instructions
    #   - Check ADD vs COPY usage
    #   - Check for unpinned versions in pip/apt-get install

    # TODO: Step 5 - Check file-level issues after parsing
    # If no USER instruction, add warning about running as root
    # If no HEALTHCHECK, add info about production readiness
    # If many consecutive RUN commands, suggest combining them

    # TODO: Step 6 - Return the issues list

    pass  # Remove this when you implement


# =============================================================================
# PART B: Debug Exercise - Fix the Dockerfile
# =============================================================================

BROKEN_DOCKERFILE = '''
# This Dockerfile has multiple issues - can you spot them all?

FROM python:latest

# Copy everything first
COPY . /app
WORKDIR /app

# Install system dependencies
RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y libpq-dev

# Install Python dependencies
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install flask

# Add the model file
ADD model.pkl /app/model.pkl

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
'''

# Issues to identify:
# 1. _______________
# 2. _______________
# 3. _______________
# 4. _______________
# 5. _______________
# 6. _______________
# 7. _______________
# 8. _______________


def get_fixed_dockerfile() -> str:
    """
    Return a fixed version of BROKEN_DOCKERFILE.

    Your fixed version should:
    1. Use a specific Python version tag
    2. Order layers for optimal caching (dependencies before code)
    3. Combine RUN commands to reduce layers
    4. Use COPY instead of ADD for local files
    5. Pin package versions
    6. Create a non-root user
    7. Add a HEALTHCHECK
    8. Clean up apt cache

    Returns:
        String containing the fixed Dockerfile
    """
    # TODO: Write the fixed Dockerfile as a multi-line string
    # Follow Docker best practices for ML applications

    fixed = '''
# TODO: Replace this with your fixed Dockerfile
'''

    return fixed


def explain_dockerfile_issues() -> dict[str, str]:
    """
    Return a dictionary explaining each issue in BROKEN_DOCKERFILE.

    Returns:
        Dict mapping issue name to explanation of why it's problematic

    Example:
        >>> explanations = explain_dockerfile_issues()
        >>> "latest_tag" in explanations
        True
    """
    # TODO: Fill in explanations for each issue
    return {
        "latest_tag": "",  # Why is FROM python:latest bad?
        "copy_before_deps": "",  # Why is copying code before deps bad?
        "multiple_runs": "",  # Why are multiple RUN commands bad?
        "add_vs_copy": "",  # When to use ADD vs COPY?
        "unpinned_versions": "",  # Why pin package versions?
        "running_as_root": "",  # Why not run as root?
        "no_healthcheck": "",  # Why add HEALTHCHECK?
        "apt_cache": "",  # Why clean apt cache?
    }


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
/hint 1 - validate_dockerfile:
Start with the basic structure:
    with open(dockerfile_path) as f:
        lines = f.readlines()

    issues = []
    has_user = False
    has_healthcheck = False

    dockerignore = dockerfile_path.parent / ".dockerignore"
    if not dockerignore.exists():
        issues.append(DockerfileIssue(0, "warning", "No .dockerignore", "Create one"))

/hint 2 - validate_dockerfile:
For line parsing:
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("FROM "):
            if ":latest" in stripped or ":" not in stripped.split()[1]:
                issues.append(DockerfileIssue(i, "error", "Use specific tag", "Pin version"))

        if stripped.startswith("USER "):
            has_user = True

/hint 3 - validate_dockerfile:
Check for unpinned packages:
    if "pip install" in stripped and "==" not in stripped:
        issues.append(DockerfileIssue(i, "warning", "Unpinned pip package", "Add ==version"))

    if "apt-get install" in stripped and "=" not in stripped:
        issues.append(DockerfileIssue(i, "info", "Consider pinning apt packages", "Add =version"))

/hint 1 - get_fixed_dockerfile:
Good structure for ML Dockerfile:
    FROM python:3.11-slim

    # Install deps first (changes less often)
    COPY requirements.txt .
    RUN pip install -r requirements.txt

    # Then copy code (changes more often)
    COPY . .

/hint 2 - get_fixed_dockerfile:
Combine RUN commands:
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            gcc \
            libpq-dev && \
        rm -rf /var/lib/apt/lists/*

/hint 3 - get_fixed_dockerfile:
Complete fixed Dockerfile:
    FROM python:3.11-slim

    WORKDIR /app

    # Install system deps (combined, with cleanup)
    RUN apt-get update && \
        apt-get install -y --no-install-recommends gcc libpq-dev && \
        rm -rf /var/lib/apt/lists/*

    # Install Python deps first for better caching
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY . .

    # Create non-root user
    RUN useradd -m appuser && chown -R appuser:appuser /app
    USER appuser

    EXPOSE 5000

    HEALTHCHECK --interval=30s --timeout=3s \
        CMD curl -f http://localhost:5000/health || exit 1

    CMD ["python", "app.py"]
"""
