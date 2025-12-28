"""
Exercise 3.3: Containerization Deep Dive
Difficulty: ★★★
Topic: Optimizing Docker Images for Production ML

This exercise teaches you to analyze and optimize Docker configurations
for ML applications. You'll learn to identify bloat, remove unused
dependencies, and design efficient multi-stage builds.

Instructions:
1. WRITE & VERIFY: Create optimize_requirements() function (Part A)
2. FIX THIS ISSUE: Analyze and fix a bloated 2GB container (Part B)
3. DESIGN DECISION: Multi-stage build decision function (Part C)

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# =============================================================================
# PART A: Write & Verify - Optimize Requirements
# =============================================================================
# Create a function that analyzes a requirements.txt file and identifies
# unused dependencies based on actual imports in Python source files.
#
# This is critical for reducing Docker image size - every unused dependency
# adds megabytes to your image and increases security surface area.
# =============================================================================


def optimize_requirements(
    requirements_path: str | Path,
    source_dirs: list[str | Path],
    keep_packages: list[str] | None = None,
) -> dict:
    """
    Analyze requirements.txt and identify unused dependencies.

    Scans Python source files for import statements and compares
    against declared dependencies.

    Args:
        requirements_path: Path to requirements.txt file
        source_dirs: List of directories to scan for Python imports
        keep_packages: Packages to keep even if not imported (e.g., pytest)

    Returns:
        dict with keys:
            - used: list of packages that are imported in source
            - unused: list of packages not found in any imports
            - missing: list of imports not in requirements (potential issue!)
            - optimized_requirements: string content for optimized requirements.txt
            - size_reduction_estimate: estimated MB saved (rough: 5MB per unused package)

    Raises:
        FileNotFoundError: If requirements_path doesn't exist
        ValueError: If no source directories provided

    Example:
        >>> result = optimize_requirements(
        ...     "requirements.txt",
        ...     ["src/", "api/"],
        ...     keep_packages=["pytest", "black"]
        ... )
        >>> print(result["unused"])
        ['tensorflow', 'opencv-python']  # Not imported anywhere
        >>> print(result["size_reduction_estimate"])
        500  # Estimated 500MB saved!
    """
    # TODO: Implement the optimization logic
    # 1. Parse requirements.txt (handle versions, extras, comments)
    # 2. Scan source_dirs for .py files
    # 3. Extract all import statements (import X, from X import Y)
    # 4. Map imports to package names (tricky! e.g., PIL -> Pillow)
    # 5. Compare and categorize
    # 6. Generate optimized requirements
    pass


# Common package name -> import name mappings (package installs as different import)
PACKAGE_TO_IMPORT = {
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "python-dateutil": "dateutil",
    "typing-extensions": "typing_extensions",
}

# Reverse mapping
IMPORT_TO_PACKAGE = {v: k for k, v in PACKAGE_TO_IMPORT.items()}


def parse_requirements(requirements_content: str) -> list[dict]:
    """
    Parse requirements.txt content into structured format.

    Args:
        requirements_content: Raw content of requirements.txt

    Returns:
        List of dicts with keys: name, version_spec, extras, line

    Example:
        >>> parse_requirements("pandas>=1.0,<2.0\\nnumpy==1.21.0")
        [
            {"name": "pandas", "version_spec": ">=1.0,<2.0", "extras": [], "line": "pandas>=1.0,<2.0"},
            {"name": "numpy", "version_spec": "==1.21.0", "extras": [], "line": "numpy==1.21.0"}
        ]
    """
    # TODO: Implement requirements parsing
    # Handle: comments (#), empty lines, version specs (>=, ==, ~=, etc.)
    # Handle: extras like package[extra1,extra2]
    # Handle: environment markers like ; python_version >= "3.8"
    pass


def extract_imports(python_content: str) -> set[str]:
    """
    Extract all imported module names from Python source code.

    Args:
        python_content: Raw Python source code

    Returns:
        Set of top-level module names imported

    Example:
        >>> code = '''
        ... import os
        ... import pandas as pd
        ... from sklearn.model_selection import train_test_split
        ... from . import local_module
        ... '''
        >>> extract_imports(code)
        {'os', 'pandas', 'sklearn'}  # Note: local imports excluded
    """
    # TODO: Implement import extraction
    # Use ast module for reliable parsing
    # Handle: import X, import X as Y, from X import Y, from X.Y import Z
    # Exclude: relative imports (from . import X)
    pass


# =============================================================================
# PART B: Fix This Issue - Bloated 2GB Container
# =============================================================================
# Analyze this Dockerfile configuration and identify why the image is 2GB
# when it should be ~200MB. Return specific recommendations.
# =============================================================================


BLOATED_DOCKERFILE = """
# This Dockerfile produces a 2GB image. It should be ~200MB!

FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    gcc \\
    g++ \\
    gfortran \\
    libopenblas-dev \\
    liblapack-dev \\
    git \\
    curl \\
    wget \\
    vim \\
    htop \\
    postgresql-client \\
    redis-tools \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build and cache model
RUN python -c "import torch; print(torch.__version__)"
RUN python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"

# Run the app
CMD ["python", "app.py"]
"""

BLOATED_REQUIREMENTS = """
# Production requirements (not all are needed!)
torch>=1.12.0
transformers>=4.20.0
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.4.0
scikit-learn>=1.0.0
scipy>=1.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.9.0
opencv-python>=4.6.0
pillow>=9.0.0
nltk>=3.7
spacy>=3.4.0
fastapi>=0.85.0
uvicorn>=0.18.0
requests>=2.28.0
aiohttp>=3.8.0
boto3>=1.24.0
google-cloud-storage>=2.5.0
azure-storage-blob>=12.13.0
mlflow>=1.28.0
wandb>=0.13.0
jupyter>=1.0.0
jupyterlab>=3.4.0
pytest>=7.1.0
pytest-cov>=3.0.0
black>=22.6.0
flake8>=5.0.0
mypy>=0.971
pre-commit>=2.20.0
ipython>=8.4.0
tqdm>=4.64.0
rich>=12.5.0
click>=8.1.0
pydantic>=1.10.0
python-multipart>=0.0.5
httpx>=0.23.0
"""


@dataclass
class DockerfileIssue:
    """A specific issue found in Dockerfile analysis."""

    category: Literal[
        "base_image", "apt_packages", "pip_packages", "build_cache", "layer_order", "runtime_bloat"
    ]
    severity: Literal["critical", "major", "minor"]
    description: str
    recommendation: str
    estimated_size_mb: int  # Estimated bloat from this issue


@dataclass
class DockerAnalysisResult:
    """Complete analysis of a Dockerfile."""

    issues: list[DockerfileIssue]
    current_size_estimate_mb: int
    optimized_size_estimate_mb: int
    optimized_dockerfile: str  # Suggested fixed Dockerfile


def analyze_bloated_container(
    dockerfile_content: str, requirements_content: str, app_description: str
) -> DockerAnalysisResult:
    """
    Analyze Dockerfile and requirements for bloat, return fixes.

    Args:
        dockerfile_content: The Dockerfile to analyze
        requirements_content: The requirements.txt content
        app_description: What the app actually does (to identify unused deps)
            e.g., "FastAPI sentiment classifier using scikit-learn"

    Returns:
        DockerAnalysisResult with issues found and optimized Dockerfile

    Example:
        >>> result = analyze_bloated_container(
        ...     BLOATED_DOCKERFILE,
        ...     BLOATED_REQUIREMENTS,
        ...     "FastAPI sentiment classifier using scikit-learn"
        ... )
        >>> len(result.issues) > 5
        True
        >>> result.optimized_size_estimate_mb < 500
        True
    """
    # TODO: Implement the analysis
    # Look for these common issues:
    # 1. Base image (python:3.11 is ~900MB, python:3.11-slim is ~120MB)
    # 2. Unused apt packages (vim, htop, etc. not needed in production)
    # 3. Unused pip packages (torch AND tensorflow? jupyter in prod?)
    # 4. No multi-stage build (build deps left in final image)
    # 5. Cached model files (BERT tokenizer downloads ~400MB)
    # 6. Development tools in production (pytest, black, mypy)
    pass


# =============================================================================
# PART C: Design Decision - Multi-Stage Build
# =============================================================================
# Decide when to use multi-stage builds and generate appropriate configuration.
# =============================================================================


@dataclass
class BuildRequirements:
    """Requirements for a container build."""

    has_compiled_dependencies: bool  # C extensions, Cython, etc.
    has_model_artifacts: bool  # Pre-trained models to include
    artifact_size_mb: int
    needs_gpu: bool
    base_image_preference: Literal["debian", "alpine", "distroless"]
    security_priority: Literal["standard", "high", "critical"]


@dataclass
class MultiStageBuildConfig:
    """Configuration for a multi-stage Docker build."""

    use_multi_stage: bool
    num_stages: int
    stages: list[dict]  # List of stage configs
    reasoning: list[str]
    estimated_final_size_mb: int
    dockerfile_template: str


def design_multi_stage_build(requirements: BuildRequirements) -> MultiStageBuildConfig:
    """
    Design a multi-stage build configuration based on requirements.

    Args:
        requirements: BuildRequirements dataclass with build constraints

    Returns:
        MultiStageBuildConfig with stages, reasoning, and Dockerfile template

    Example:
        >>> reqs = BuildRequirements(
        ...     has_compiled_dependencies=True,
        ...     has_model_artifacts=True,
        ...     artifact_size_mb=100,
        ...     needs_gpu=False,
        ...     base_image_preference="debian",
        ...     security_priority="high"
        ... )
        >>> config = design_multi_stage_build(reqs)
        >>> config.use_multi_stage
        True
        >>> config.num_stages >= 2
        True
        >>> "builder" in config.dockerfile_template
        True
    """
    # TODO: Implement multi-stage build design
    # Consider:
    # 1. Compiled deps -> need build stage with gcc/build-essential
    # 2. Model artifacts -> might need separate stage to download/prepare
    # 3. GPU -> need nvidia/cuda base image
    # 4. Security priority high -> use distroless or minimal final stage
    # 5. Alpine base -> might have compatibility issues with some packages
    pass


# =============================================================================
# Utility functions for analysis
# =============================================================================

# Package size estimates in MB (rough, for analysis)
PACKAGE_SIZES = {
    "torch": 700,
    "tensorflow": 500,
    "transformers": 300,
    "opencv-python": 100,
    "scipy": 50,
    "pandas": 40,
    "numpy": 30,
    "scikit-learn": 40,
    "matplotlib": 50,
    "jupyter": 100,
    "jupyterlab": 150,
    "spacy": 50,  # Plus models can be 500MB+
    "nltk": 30,  # Plus data can be 1GB+
    "pillow": 20,
    "fastapi": 5,
    "uvicorn": 5,
    "pytest": 10,
    "black": 10,
    "mlflow": 50,
    "wandb": 30,
}

# Base image sizes in MB
BASE_IMAGE_SIZES = {
    "python:3.11": 900,
    "python:3.11-slim": 120,
    "python:3.11-alpine": 50,
    "nvidia/cuda:11.8-runtime": 3000,
    "gcr.io/distroless/python3": 50,
}


# =============================================================================
# HINTS (revealed progressively with /hint command)
# =============================================================================
"""
HINT 1 - optimize_requirements:
- Use ast module to parse Python reliably:
    import ast
    tree = ast.parse(python_content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # Not relative import
                imports.add(node.module.split('.')[0])

- Parse requirements with regex:
    import re
    pattern = r'^([a-zA-Z0-9_-]+)'
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            match = re.match(pattern, line)
            if match:
                packages.append(match.group(1).lower())

HINT 2 - analyze_bloated_container:
- Check base image:
    if "python:3.11" in dockerfile and "-slim" not in dockerfile:
        issues.append(DockerfileIssue(
            category="base_image",
            severity="critical",
            description="Using full Python image instead of slim",
            recommendation="Use python:3.11-slim",
            estimated_size_mb=780
        ))

- Check for ML framework conflicts:
    has_torch = "torch" in requirements
    has_tf = "tensorflow" in requirements
    if has_torch and has_tf:
        issues.append(DockerfileIssue(
            category="pip_packages",
            severity="critical",
            description="Both PyTorch and TensorFlow included",
            recommendation="Choose one framework based on actual usage",
            estimated_size_mb=500
        ))

HINT 3 - design_multi_stage_build:
- Basic two-stage pattern:
    stages = [
        {
            "name": "builder",
            "base": "python:3.11",
            "purpose": "Install and compile dependencies",
            "commands": ["pip install --user -r requirements.txt"]
        },
        {
            "name": "runtime",
            "base": "python:3.11-slim",
            "purpose": "Minimal runtime",
            "commands": ["COPY --from=builder /root/.local /root/.local"]
        }
    ]

- Template structure:
    dockerfile_template = '''
    # Build stage
    FROM {builder_base} AS builder
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --user --no-cache-dir -r requirements.txt

    # Runtime stage
    FROM {runtime_base}
    COPY --from=builder /root/.local /root/.local
    COPY . .
    ENV PATH=/root/.local/bin:$PATH
    CMD ["python", "app.py"]
    '''
"""

# =============================================================================
# Test your implementation (run with: python exercise_3_3.py)
# =============================================================================
if __name__ == "__main__":
    print("Exercise 3.3: Containerization Deep Dive")
    print("=" * 50)

    # Test Part B - Bloated container analysis
    print("\nPart B - Testing analyze_bloated_container:")
    try:
        result = analyze_bloated_container(
            BLOATED_DOCKERFILE,
            BLOATED_REQUIREMENTS,
            "FastAPI sentiment classifier using scikit-learn",
        )
        if result:
            print(f"  Found {len(result.issues)} issues")
            print(f"  Current estimate: {result.current_size_estimate_mb}MB")
            print(f"  Optimized estimate: {result.optimized_size_estimate_mb}MB")
            if result.optimized_size_estimate_mb < 500:
                print("  [PASS] Optimization achieves < 500MB")
            else:
                print("  [FAIL] Should optimize below 500MB")
        else:
            print("  [TODO] Function returns None")
    except Exception as e:
        print(f"  [TODO] analyze_bloated_container not implemented: {e}")

    # Test Part C - Multi-stage build design
    print("\nPart C - Testing design_multi_stage_build:")
    try:
        reqs = BuildRequirements(
            has_compiled_dependencies=True,
            has_model_artifacts=True,
            artifact_size_mb=100,
            needs_gpu=False,
            base_image_preference="debian",
            security_priority="high",
        )
        config = design_multi_stage_build(reqs)
        if config:
            print(f"  Use multi-stage: {config.use_multi_stage}")
            print(f"  Number of stages: {config.num_stages}")
            if config.use_multi_stage and config.num_stages >= 2:
                print("  [PASS] Correctly recommends multi-stage build")
            else:
                print("  [FAIL] Should use multi-stage for compiled deps")
        else:
            print("  [TODO] Function returns None")
    except Exception as e:
        print(f"  [TODO] design_multi_stage_build not implemented: {e}")

    print("\n" + "=" * 50)
    print("Run pytest test_level_3.py for full test coverage")
