"""
Exercise 4.3: Model Drift & Retraining
Difficulty: *** (Production-level debugging)
Topic: Statistical drift detection, KS tests, retraining triggers

Instructions:
Models degrade over time as the world changes. This exercise covers detecting drift
and deciding when to retrain. You'll debug a flawed drift detection script and
implement proper statistical tests.

Part 1: Fix a drift detection script with incorrect statistical test usage (DEBUG)
Part 2: Implement proper KS test for drift detection (WRITE & VERIFY)
Part 3: Create RetrainingTrigger class (WRITE & VERIFY)

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TypedDict


# =============================================================================
# PART 1: DEBUG - Fix Incorrect Statistical Test Usage
# =============================================================================
#
# A junior data scientist implemented drift detection, but it's producing
# too many false positives. The on-call team is getting paged every night
# for "drift detected" when there's no actual problem.
#
# BUG REPORT:
# """
# Alert: Model Drift Detected
# Frequency: 3-5 alerts per day
# False positive rate: ~90%
#
# Investigation notes:
# - Checked model performance manually after each alert
# - Accuracy is stable at 87% (within normal range)
# - Prediction distribution looks normal
# - But the drift detector keeps firing
#
# Hypothesis: Something is wrong with the statistical test implementation
# """

class BrokenDriftDetector:
    """
    This drift detector has multiple bugs. DO NOT USE.

    The bugs cause it to detect "drift" when there is none.
    """

    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        """
        Initialize with reference distribution.

        Args:
            reference_data: Feature values from training data
            threshold: p-value threshold for significance
        """
        self.reference_data = reference_data
        self.threshold = threshold

    def detect_drift(self, current_data: np.ndarray) -> dict:
        """
        Detect drift using KS test. THIS IMPLEMENTATION HAS BUGS.

        Args:
            current_data: Current production feature values

        Returns:
            dict with statistic, p_value, and drift_detected
        """
        from scipy import stats

        # BUG 1: Using one-sample KS test instead of two-sample
        # ks_1samp tests if data comes from a specific distribution (like normal)
        # We need ks_2samp to compare two empirical distributions!
        statistic, p_value = stats.ks_1samp(current_data, stats.norm.cdf)

        # BUG 2: Comparing p-value incorrectly
        # A LOW p-value means distributions are DIFFERENT (reject null hypothesis)
        # This code triggers drift when p-value is HIGH (wrong!)
        drift_detected = p_value > self.threshold  # WRONG!

        # BUG 3: Not handling edge cases
        # What if current_data is empty or too small?
        # What if reference_data was empty?

        return {
            "statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift_detected,
        }


def demonstrate_broken_detector():
    """Show how the broken detector produces false positives."""
    np.random.seed(42)

    # Reference: normal distribution with mean=50, std=10
    reference = np.random.normal(50, 10, 1000)

    # Current: SAME distribution (no drift!)
    current = np.random.normal(50, 10, 500)

    detector = BrokenDriftDetector(reference, threshold=0.05)
    result = detector.detect_drift(current)

    print("Testing with identical distributions (should NOT detect drift):")
    print(f"  KS statistic: {result['statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Drift detected: {result['drift_detected']}")  # Will wrongly say True!


# YOUR TASK: Identify all three bugs and fix them


class DriftDetectionResult(TypedDict):
    """Result of drift detection."""
    statistic: float
    p_value: float
    drift_detected: bool
    sample_sizes: dict[str, int]
    error: str | None


def detect_drift_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05,
    min_samples: int = 30,
) -> DriftDetectionResult:
    """
    Detect distribution drift using the two-sample Kolmogorov-Smirnov test.

    The KS test compares two distributions by measuring the maximum distance
    between their cumulative distribution functions. A small p-value indicates
    the distributions are significantly different.

    Args:
        reference: Feature values from training/reference period
        current: Feature values from current production window
        threshold: Significance level (p-value below this = drift detected)
        min_samples: Minimum samples required for reliable test

    Returns:
        DriftDetectionResult with:
        - statistic: KS test statistic (max CDF distance)
        - p_value: Probability of observing this distance if distributions are same
        - drift_detected: True if p_value < threshold
        - sample_sizes: Dict with reference and current sample counts
        - error: Error message if test couldn't be performed

    Example:
        >>> reference = np.random.normal(50, 10, 1000)
        >>> current_no_drift = np.random.normal(50, 10, 500)
        >>> current_drift = np.random.normal(60, 10, 500)  # Mean shifted!
        >>>
        >>> result_no_drift = detect_drift_ks_test(reference, current_no_drift)
        >>> result_no_drift['drift_detected']  # Should be False
        False
        >>> result_drift = detect_drift_ks_test(reference, current_drift)
        >>> result_drift['drift_detected']  # Should be True
        True
    """
    from scipy import stats

    # TODO: Implement proper drift detection
    #
    # Steps:
    # 1. Validate inputs (check for empty arrays, minimum samples)
    # 2. Use scipy.stats.ks_2samp (TWO-sample test)
    # 3. Interpret p-value correctly (low = drift, high = no drift)
    # 4. Return complete result with all fields

    # Placeholder - implement your solution
    return DriftDetectionResult(
        statistic=0.0,
        p_value=1.0,
        drift_detected=False,
        sample_sizes={"reference": 0, "current": 0},
        error="Not implemented",
    )


# =============================================================================
# PART 2: WRITE & VERIFY - Population Stability Index (PSI)
# =============================================================================
#
# PSI is another common drift metric, especially in finance.
# It measures how much a distribution has shifted.


class PSIResult(TypedDict):
    """Result of PSI calculation."""
    psi: float
    interpretation: str
    bucket_details: list[dict]


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> PSIResult:
    """
    Calculate Population Stability Index between two distributions.

    PSI measures distribution shift by:
    1. Bucketing the reference distribution into bins
    2. Calculating the proportion in each bin for both distributions
    3. Computing: sum((current_pct - ref_pct) * ln(current_pct / ref_pct))

    Interpretation:
    - PSI < 0.1: No significant shift
    - PSI 0.1 - 0.25: Moderate shift (investigate)
    - PSI > 0.25: Significant shift (action required)

    Args:
        reference: Reference distribution values
        current: Current distribution values
        n_bins: Number of bins for bucketing

    Returns:
        PSIResult with:
        - psi: The PSI score
        - interpretation: Human-readable interpretation
        - bucket_details: Per-bucket breakdown

    Example:
        >>> reference = np.random.normal(50, 10, 1000)
        >>> current_no_drift = np.random.normal(50, 10, 500)
        >>> current_drift = np.random.normal(60, 15, 500)
        >>>
        >>> result_no_drift = calculate_psi(reference, current_no_drift)
        >>> result_no_drift['psi'] < 0.1  # Should be True
        True
        >>> result_drift = calculate_psi(reference, current_drift)
        >>> result_drift['psi'] > 0.25  # Should be True (significant shift)
        True
    """
    # TODO: Implement PSI calculation
    #
    # Steps:
    # 1. Create bin edges from reference distribution
    # 2. Calculate proportion in each bin for reference
    # 3. Calculate proportion in each bin for current
    # 4. Add small epsilon to avoid log(0) and division by zero
    # 5. Calculate PSI for each bucket and sum
    # 6. Interpret the result

    # Placeholder - implement your solution
    return PSIResult(
        psi=0.0,
        interpretation="Not implemented",
        bucket_details=[],
    )


# =============================================================================
# PART 3: WRITE & VERIFY - RetrainingTrigger
# =============================================================================
#
# When should you retrain? Not on every alert! This class encodes the logic
# for deciding when retraining is appropriate.


class RetrainingAction(Enum):
    """Actions the retraining system can take."""
    NONE = "none"  # No action needed
    MONITOR = "monitor"  # Watch more closely
    INVESTIGATE = "investigate"  # Human should look
    RETRAIN = "retrain"  # Trigger retraining pipeline
    EMERGENCY = "emergency"  # Immediate action needed


@dataclass
class RetrainingDecision:
    """Result of retraining evaluation."""
    action: RetrainingAction
    reason: str
    metrics_summary: dict
    confidence: float  # 0-1, how confident we are in this decision


@dataclass
class RetrainingTrigger:
    """
    Decides when to trigger model retraining based on drift metrics.

    This class prevents alert fatigue by requiring multiple conditions
    before recommending retraining:
    1. Drift metrics exceed thresholds
    2. Enough time since last training (cooldown)
    3. Enough new data available for retraining
    4. (Optional) Actual performance degradation confirmed

    Attributes:
        psi_warning_threshold: PSI level that triggers monitoring
        psi_retrain_threshold: PSI level that triggers retraining
        accuracy_drop_threshold: Accuracy drop that triggers retraining
        min_samples_for_retrain: Minimum new samples needed
        cooldown_hours: Minimum hours between retraining
    """
    psi_warning_threshold: float = 0.1
    psi_retrain_threshold: float = 0.25
    accuracy_drop_threshold: float = 0.05  # 5% drop
    min_samples_for_retrain: int = 10000
    cooldown_hours: int = 24

    # Internal state
    _last_retrain_time: datetime | None = field(default=None, repr=False)

    def evaluate(
        self,
        psi_score: float,
        accuracy_current: float | None,
        accuracy_baseline: float | None,
        available_samples: int,
        current_time: datetime | None = None,
    ) -> RetrainingDecision:
        """
        Evaluate whether retraining should be triggered.

        Args:
            psi_score: Current PSI score from drift detection
            accuracy_current: Current model accuracy (None if unknown)
            accuracy_baseline: Baseline accuracy from last training (None if unknown)
            available_samples: Number of new labeled samples available
            current_time: Current timestamp (defaults to now)

        Returns:
            RetrainingDecision with recommended action and reasoning

        Decision logic:
        1. If accuracy dropped significantly -> RETRAIN (if samples available)
        2. If PSI > retrain_threshold -> INVESTIGATE or RETRAIN
        3. If PSI > warning_threshold -> MONITOR
        4. If in cooldown period -> Block retraining
        5. If insufficient samples -> Block retraining

        Example:
            >>> trigger = RetrainingTrigger()
            >>> decision = trigger.evaluate(
            ...     psi_score=0.35,
            ...     accuracy_current=0.82,
            ...     accuracy_baseline=0.88,
            ...     available_samples=15000,
            ... )
            >>> decision.action
            RetrainingAction.RETRAIN
            >>> decision.reason
            'PSI exceeds threshold (0.35 > 0.25) and accuracy dropped 6.8%'
        """
        if current_time is None:
            current_time = datetime.now()

        # TODO: Implement the decision logic
        #
        # Consider:
        # 1. Check cooldown period first (if recently retrained, no action)
        # 2. Check if we have enough samples for retraining
        # 3. Check accuracy drop (most reliable signal)
        # 4. Check PSI thresholds
        # 5. Combine signals to make final decision
        # 6. Calculate confidence based on data quality

        # Placeholder - implement your solution
        return RetrainingDecision(
            action=RetrainingAction.NONE,
            reason="Not implemented",
            metrics_summary={},
            confidence=0.0,
        )

    def record_retraining(self, retrain_time: datetime | None = None):
        """Record that retraining was performed (for cooldown tracking)."""
        if retrain_time is None:
            retrain_time = datetime.now()
        self._last_retrain_time = retrain_time

    def is_in_cooldown(self, current_time: datetime | None = None) -> bool:
        """Check if still in cooldown period after last retraining."""
        if self._last_retrain_time is None:
            return False
        if current_time is None:
            current_time = datetime.now()
        elapsed = current_time - self._last_retrain_time
        return elapsed < timedelta(hours=self.cooldown_hours)


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Part 1 - Broken Drift Detector) ===

The three bugs in BrokenDriftDetector:

BUG 1: Wrong test type
  WRONG: stats.ks_1samp(current_data, stats.norm.cdf)
  RIGHT: stats.ks_2samp(reference_data, current_data)

  ks_1samp tests if data follows a theoretical distribution (like normal)
  ks_2samp compares two empirical samples (what we want!)

BUG 2: Wrong p-value interpretation
  WRONG: drift_detected = p_value > threshold
  RIGHT: drift_detected = p_value < threshold

  Low p-value means: "It's unlikely these came from the same distribution"
  High p-value means: "These could plausibly be from the same distribution"

BUG 3: No validation
  Need to check:
  - len(reference) >= min_samples
  - len(current) >= min_samples
  - Arrays are not empty
  - Handle edge cases gracefully

Fixed implementation:

```python
def detect_drift_ks_test(reference, current, threshold=0.05, min_samples=30):
    from scipy import stats

    # Validation
    if len(reference) < min_samples:
        return DriftDetectionResult(
            statistic=0.0, p_value=1.0, drift_detected=False,
            sample_sizes={"reference": len(reference), "current": len(current)},
            error=f"Reference too small: {len(reference)} < {min_samples}"
        )

    if len(current) < min_samples:
        return DriftDetectionResult(
            statistic=0.0, p_value=1.0, drift_detected=False,
            sample_sizes={"reference": len(reference), "current": len(current)},
            error=f"Current too small: {len(current)} < {min_samples}"
        )

    # Correct test: TWO-sample KS test
    statistic, p_value = stats.ks_2samp(reference, current)

    # Correct interpretation: low p-value = drift
    drift_detected = p_value < threshold

    return DriftDetectionResult(
        statistic=float(statistic),
        p_value=float(p_value),
        drift_detected=drift_detected,
        sample_sizes={"reference": len(reference), "current": len(current)},
        error=None,
    )
```


=== HINT 2 (Part 2 - PSI Calculation) ===

PSI formula: sum((current_pct - ref_pct) * ln(current_pct / ref_pct))

```python
def calculate_psi(reference, current, n_bins=10):
    # Create bins from reference distribution
    _, bin_edges = np.histogram(reference, bins=n_bins)

    # Count samples in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to percentages (add epsilon to avoid division by zero)
    epsilon = 1e-10
    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * n_bins)
    curr_pct = (curr_counts + epsilon) / (len(current) + epsilon * n_bins)

    # Calculate PSI
    psi_per_bucket = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    total_psi = float(np.sum(psi_per_bucket))

    # Interpret
    if total_psi < 0.1:
        interpretation = "No significant shift"
    elif total_psi < 0.25:
        interpretation = "Moderate shift - investigate"
    else:
        interpretation = "Significant shift - action required"

    # Build bucket details
    bucket_details = []
    for i in range(n_bins):
        bucket_details.append({
            "bin_low": float(bin_edges[i]),
            "bin_high": float(bin_edges[i+1]),
            "ref_pct": float(ref_pct[i]),
            "curr_pct": float(curr_pct[i]),
            "psi_contribution": float(psi_per_bucket[i]),
        })

    return PSIResult(
        psi=total_psi,
        interpretation=interpretation,
        bucket_details=bucket_details,
    )
```


=== HINT 3 (Part 3 - RetrainingTrigger) ===

The decision logic should be hierarchical:

```python
def evaluate(self, psi_score, accuracy_current, accuracy_baseline,
             available_samples, current_time=None):
    if current_time is None:
        current_time = datetime.now()

    metrics_summary = {
        "psi_score": psi_score,
        "accuracy_current": accuracy_current,
        "accuracy_baseline": accuracy_baseline,
        "available_samples": available_samples,
    }

    # Check cooldown first
    if self.is_in_cooldown(current_time):
        return RetrainingDecision(
            action=RetrainingAction.NONE,
            reason="In cooldown period after recent retraining",
            metrics_summary=metrics_summary,
            confidence=0.9,
        )

    # Calculate accuracy drop if possible
    accuracy_drop = None
    if accuracy_current is not None and accuracy_baseline is not None:
        accuracy_drop = accuracy_baseline - accuracy_current
        metrics_summary["accuracy_drop"] = accuracy_drop

    # Determine action based on signals
    reasons = []
    action = RetrainingAction.NONE

    # Signal 1: Accuracy drop (most reliable)
    if accuracy_drop is not None and accuracy_drop > self.accuracy_drop_threshold:
        reasons.append(f"Accuracy dropped {accuracy_drop:.1%}")
        action = RetrainingAction.RETRAIN

    # Signal 2: High PSI
    if psi_score > self.psi_retrain_threshold:
        reasons.append(f"PSI exceeds threshold ({psi_score:.2f} > {self.psi_retrain_threshold})")
        if action != RetrainingAction.RETRAIN:
            action = RetrainingAction.INVESTIGATE if accuracy_drop is None else RetrainingAction.RETRAIN
    elif psi_score > self.psi_warning_threshold:
        reasons.append(f"PSI elevated ({psi_score:.2f} > {self.psi_warning_threshold})")
        if action == RetrainingAction.NONE:
            action = RetrainingAction.MONITOR

    # Check if retraining is feasible
    if action == RetrainingAction.RETRAIN:
        if available_samples < self.min_samples_for_retrain:
            return RetrainingDecision(
                action=RetrainingAction.INVESTIGATE,
                reason=f"Retraining needed but insufficient samples ({available_samples} < {self.min_samples_for_retrain})",
                metrics_summary=metrics_summary,
                confidence=0.7,
            )

    # Calculate confidence
    confidence = 0.5
    if accuracy_drop is not None:
        confidence = 0.9  # High confidence when we have accuracy data
    elif psi_score > self.psi_retrain_threshold:
        confidence = 0.7  # Medium confidence on PSI alone

    reason = "; ".join(reasons) if reasons else "All metrics within normal range"

    return RetrainingDecision(
        action=action,
        reason=reason,
        metrics_summary=metrics_summary,
        confidence=confidence,
    )
```

Key principles:
1. Accuracy drop is the most reliable signal (but requires labels)
2. PSI alone should trigger investigation, not automatic retraining
3. Always check feasibility (cooldown, sample size) before recommending action
4. Report confidence based on data quality
"""
