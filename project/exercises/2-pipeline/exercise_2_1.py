"""
Exercise 2.1: Data Versioning
Difficulty: ★★☆
Topic: Content-addressable storage and DVC pipeline concepts

Learning Objectives:
- Understand how DVC tracks data files using content hashes
- Learn the principles behind reproducible data pipelines
- Debug common DVC pipeline configuration issues

Instructions:

PART A - Write & Verify (50%):
Implement `generate_data_hash(filepath)` that creates a content-addressable
hash for a data file, similar to how DVC tracks files.

Requirements:
1. Read the file in binary mode (works for any file type)
2. Use MD5 hashing (what DVC uses internally)
3. Return both the hash and file size
4. Handle files that don't exist gracefully

PART B - Debug (30%):
The `BUGGY_DVC_PIPELINE` below has incorrect dependency ordering.
Fix the `fix_dvc_pipeline()` function to return a corrected pipeline.

PART C - Extend (20%):
Add a function to verify data integrity by comparing stored hash vs current.

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import hashlib
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataHash:
    """Result of hashing a data file."""
    filepath: str
    md5: str
    size: int


# =============================================================================
# PART A: Write & Verify - Content Hashing
# =============================================================================

def generate_data_hash(filepath: str | Path) -> DataHash:
    """Generate a content-addressable hash for a data file.

    This mimics how DVC creates .dvc files to track data. The hash
    allows DVC to detect when file contents change and enables
    caching/deduplication.

    Args:
        filepath: Path to the file to hash.

    Returns:
        DataHash with filepath, md5 hash, and file size.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> result = generate_data_hash("data/train.csv")
        >>> print(result.md5)  # e.g., "d41d8cd98f00b204e9800998ecf8427e"
    """
    # TODO: Implement this function
    #
    # Steps:
    # 1. Convert filepath to Path object
    # 2. Check if file exists (raise FileNotFoundError if not)
    # 3. Read file in binary mode and compute MD5 hash
    # 4. Get file size
    # 5. Return DataHash object
    #
    # Tip: Use hashlib.md5() and read file in chunks for large files

    raise NotImplementedError("Implement generate_data_hash()")


def generate_data_hash_chunked(filepath: str | Path, chunk_size: int = 8192) -> DataHash:
    """Generate hash using chunked reading for large files.

    For files larger than memory, we need to read in chunks.
    This is what DVC actually does internally.

    Args:
        filepath: Path to the file to hash.
        chunk_size: Size of chunks to read (default 8KB).

    Returns:
        DataHash with filepath, md5 hash, and file size.
    """
    # TODO: Implement chunked hashing
    #
    # Steps:
    # 1. Create MD5 hasher: md5 = hashlib.md5()
    # 2. Open file in binary mode
    # 3. Read chunk by chunk: while chunk := f.read(chunk_size)
    # 4. Update hasher with each chunk: md5.update(chunk)
    # 5. Get final hash: md5.hexdigest()

    raise NotImplementedError("Implement generate_data_hash_chunked()")


# =============================================================================
# PART B: Debug - Fix DVC Pipeline
# =============================================================================

# This DVC pipeline YAML has bugs! The dependency ordering is wrong.
# In a real project, this would be in dvc.yaml
BUGGY_DVC_PIPELINE = """
stages:
  train:
    cmd: python src/train.py --data data/features.csv --model-dir models/
    deps:
      - src/train.py
      - data/raw.csv           # BUG: Should depend on features.csv, not raw.csv
    outs:
      - models/model.joblib

  featurize:
    cmd: python src/features.py --input data/processed.csv --output data/features.csv
    deps:
      - src/features.py
      - data/raw.csv           # BUG: Should depend on processed.csv
    outs:
      - data/features.csv

  preprocess:
    cmd: python src/preprocess.py --input data/raw.csv --output data/processed.csv
    deps:
      - src/preprocess.py
      - data/features.csv      # BUG: Circular dependency! Should be raw.csv
    outs:
      - data/processed.csv
"""

# The correct dependency chain should be:
# raw.csv -> preprocess -> processed.csv -> featurize -> features.csv -> train -> model.joblib


def fix_dvc_pipeline(buggy_yaml: str) -> str:
    """Fix the dependency ordering in a buggy DVC pipeline.

    The pipeline has three bugs:
    1. train stage depends on raw.csv instead of features.csv
    2. featurize stage depends on raw.csv instead of processed.csv
    3. preprocess stage has circular dep on features.csv instead of raw.csv

    Args:
        buggy_yaml: The buggy pipeline YAML string.

    Returns:
        Fixed pipeline YAML string.

    Why this matters:
    - Wrong dependencies mean DVC won't rebuild when inputs change
    - Circular dependencies cause pipeline failures
    - Correct deps ensure reproducibility
    """
    # TODO: Fix the pipeline dependencies
    #
    # You can either:
    # 1. Use string replacement (simple but fragile)
    # 2. Parse YAML, fix deps, and dump back (robust but more code)
    #
    # The fixes needed are:
    # - train.deps: data/raw.csv -> data/features.csv
    # - featurize.deps: data/raw.csv -> data/processed.csv
    # - preprocess.deps: data/features.csv -> data/raw.csv

    raise NotImplementedError("Implement fix_dvc_pipeline()")


def validate_pipeline_dag(pipeline_stages: dict) -> list[str]:
    """Validate that a pipeline forms a valid DAG (no cycles).

    Args:
        pipeline_stages: Dict mapping stage name to its dependencies.
                        e.g., {"train": ["features.csv"], "featurize": ["processed.csv"]}

    Returns:
        List of stages in valid execution order (topological sort).

    Raises:
        ValueError: If pipeline contains cycles.
    """
    # TODO: Implement topological sort to detect cycles
    # This is a bonus challenge!

    raise NotImplementedError("Implement validate_pipeline_dag()")


# =============================================================================
# PART C: Extend - Data Integrity Verification
# =============================================================================

@dataclass
class IntegrityCheckResult:
    """Result of integrity check."""
    filepath: str
    expected_hash: str
    actual_hash: str
    is_valid: bool
    size_matches: bool


def verify_data_integrity(
    filepath: str | Path,
    expected_hash: str,
    expected_size: int | None = None
) -> IntegrityCheckResult:
    """Verify a file's integrity against a stored hash.

    This is used to check if data files have been corrupted or
    modified since they were originally tracked.

    Args:
        filepath: Path to the file to verify.
        expected_hash: The MD5 hash the file should have.
        expected_size: Optional expected file size in bytes.

    Returns:
        IntegrityCheckResult with verification details.

    Example:
        >>> result = verify_data_integrity("data/train.csv", "abc123", 1024)
        >>> if not result.is_valid:
        ...     print("Data corruption detected!")
    """
    # TODO: Implement integrity verification
    #
    # Steps:
    # 1. Generate hash of current file using generate_data_hash()
    # 2. Compare with expected_hash
    # 3. If expected_size provided, also compare sizes
    # 4. Return IntegrityCheckResult

    raise NotImplementedError("Implement verify_data_integrity()")


# =============================================================================
# Example usage (uncomment to test)
# =============================================================================

if __name__ == "__main__":
    # Test Part A
    print("Testing generate_data_hash...")
    # result = generate_data_hash("some_file.txt")
    # print(f"Hash: {result.md5}, Size: {result.size}")

    # Test Part B
    print("\nTesting fix_dvc_pipeline...")
    # fixed = fix_dvc_pipeline(BUGGY_DVC_PIPELINE)
    # print(fixed)

    # Test Part C
    print("\nTesting verify_data_integrity...")
    # result = verify_data_integrity("some_file.txt", "expected_hash")
    # print(f"Valid: {result.is_valid}")


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Conceptual) ===
For generate_data_hash:
- hashlib.md5() creates a hasher object
- Call .update(bytes) to add data
- Call .hexdigest() to get the final hash string
- Path(filepath).stat().st_size gives file size

=== HINT 2 (More specific) ===
For generate_data_hash:
```python
filepath = Path(filepath)
if not filepath.exists():
    raise FileNotFoundError(f"File not found: {filepath}")

with open(filepath, 'rb') as f:
    content = f.read()

md5_hash = hashlib.md5(content).hexdigest()
size = filepath.stat().st_size
```

For fix_dvc_pipeline, the simplest approach:
```python
fixed = buggy_yaml
fixed = fixed.replace("specific_wrong_dep", "correct_dep")
# ... repeat for each fix
```

=== HINT 3 (Nearly complete solution) ===
def generate_data_hash(filepath: str | Path) -> DataHash:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        content = f.read()

    return DataHash(
        filepath=str(filepath),
        md5=hashlib.md5(content).hexdigest(),
        size=filepath.stat().st_size
    )

def fix_dvc_pipeline(buggy_yaml: str) -> str:
    # The trick is to be specific enough to not break other parts
    fixed = buggy_yaml

    # Fix train stage: depends on features, not raw
    fixed = fixed.replace(
        "train:\\n    cmd: python src/train.py",
        "train:\\n    cmd: python src/train.py"
    )
    # ... you need to fix the actual dependency lines

    return fixed
"""
