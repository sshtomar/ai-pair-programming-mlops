"""
Exercise 3.4: Cloud Deployment
Difficulty: ★★★
Topic: Deploying ML Models to Cloud Platforms

This exercise teaches you to configure cloud deployments properly,
handle resource limits, and estimate costs for ML workloads.

Instructions:
1. WRITE & VERIFY: Create generate_cloud_run_config() with validation (Part A)
2. FIX THIS ISSUE: Deployment that fails under load (Part B)
3. DESIGN DECISION: Estimate costs given traffic patterns (Part C)

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# PART A: Write & Verify - Cloud Run Configuration Generator
# =============================================================================
# Create a function that generates valid Cloud Run configuration with
# proper validation of resource limits and constraints.
# =============================================================================


class CloudRunConfigError(Exception):
    """Raised when Cloud Run configuration is invalid."""

    pass


@dataclass
class CloudRunConfig:
    """Validated Cloud Run configuration."""

    cpu: str  # e.g., "1", "2", "4"
    memory: str  # e.g., "512Mi", "1Gi", "2Gi"
    min_instances: int
    max_instances: int
    concurrency: int
    timeout_seconds: int
    cpu_throttling: bool
    startup_cpu_boost: bool

    def to_yaml(self) -> str:
        """Convert config to Cloud Run YAML format."""
        return f"""
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ml-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "{self.min_instances}"
        autoscaling.knative.dev/maxScale: "{self.max_instances}"
        run.googleapis.com/cpu-throttling: "{str(self.cpu_throttling).lower()}"
        run.googleapis.com/startup-cpu-boost: "{str(self.startup_cpu_boost).lower()}"
    spec:
      containerConcurrency: {self.concurrency}
      timeoutSeconds: {self.timeout_seconds}
      containers:
      - image: gcr.io/PROJECT/IMAGE
        resources:
          limits:
            cpu: "{self.cpu}"
            memory: "{self.memory}"
"""


def generate_cloud_run_config(
    cpu: str | float | int,
    memory: str | int,
    min_instances: int = 0,
    max_instances: int = 10,
    concurrency: int = 80,
    timeout_seconds: int = 300,
    cpu_throttling: bool = True,
    startup_cpu_boost: bool = False,
) -> CloudRunConfig:
    """
    Generate a validated Cloud Run configuration.

    Args:
        cpu: CPU allocation ("1", "2", "4" or 1, 2, 4)
        memory: Memory allocation ("512Mi", "1Gi" or MB as int)
        min_instances: Minimum instances (0-100)
        max_instances: Maximum instances (1-1000)
        concurrency: Max concurrent requests per instance (1-1000)
        timeout_seconds: Request timeout (1-3600)
        cpu_throttling: Whether to throttle CPU between requests
        startup_cpu_boost: Whether to boost CPU during startup

    Returns:
        CloudRunConfig with validated settings

    Raises:
        CloudRunConfigError: If configuration is invalid

    Validation rules:
        - CPU must be 1, 2, 4, or 8 (for 2nd gen)
        - Memory must be 128Mi to 32Gi
        - Memory must be at least 256Mi per CPU
        - min_instances <= max_instances
        - If min_instances > 0, cpu_throttling should be False
        - timeout_seconds max is 3600 for HTTP, 3600 for gRPC

    Example:
        >>> config = generate_cloud_run_config(
        ...     cpu=2,
        ...     memory="4Gi",
        ...     min_instances=1,
        ...     max_instances=10
        ... )
        >>> config.cpu
        "2"
        >>> config.memory
        "4Gi"
    """
    # TODO: Implement validation and configuration generation
    # 1. Normalize cpu to string format
    # 2. Normalize memory to string format (convert int MB to "XMi" or "XGi")
    # 3. Validate CPU is in allowed values
    # 4. Validate memory is in allowed range
    # 5. Validate memory-to-CPU ratio
    # 6. Validate instance counts
    # 7. Validate concurrency
    # 8. Add warnings for suboptimal configs
    pass


def parse_memory(memory: str | int) -> int:
    """
    Parse memory string to MB.

    Args:
        memory: Memory as string ("512Mi", "1Gi") or int (MB)

    Returns:
        Memory in MB

    Example:
        >>> parse_memory("1Gi")
        1024
        >>> parse_memory("512Mi")
        512
        >>> parse_memory(256)
        256
    """
    # TODO: Implement memory parsing
    pass


def format_memory(memory_mb: int) -> str:
    """
    Format memory MB to Cloud Run format.

    Args:
        memory_mb: Memory in MB

    Returns:
        Formatted string ("512Mi" or "1Gi")

    Example:
        >>> format_memory(1024)
        "1Gi"
        >>> format_memory(512)
        "512Mi"
    """
    # TODO: Implement memory formatting
    pass


# =============================================================================
# PART B: Fix This Issue - Deployment Fails Under Load
# =============================================================================
# This deployment configuration works for testing but fails in production
# under load. Identify and fix the issues.
# =============================================================================


@dataclass
class FailingDeploymentConfig:
    """A deployment config that fails under load. Find the bugs!"""

    # BUG 1: Not enough CPU for ML inference
    cpu: str = "0.5"  # Half a CPU for ML model? Too low!

    # BUG 2: Not enough memory for model
    memory: str = "256Mi"  # ML models typically need 1-4GB

    # BUG 3: No minimum instances = cold start on every request
    min_instances: int = 0  # Cold starts take 10+ seconds

    # BUG 4: Too many concurrent requests for single-threaded model
    concurrency: int = 1000  # Model can only handle 1 at a time!

    # BUG 5: Timeout too short for model loading
    timeout_seconds: int = 10  # Model load takes 30 seconds!

    # BUG 6: CPU throttling kills performance between requests
    cpu_throttling: bool = True  # Model stays cold, slow response

    # BUG 7: No startup boost when model is loading
    startup_cpu_boost: bool = False  # Slow cold starts


@dataclass
class ProductionReadyConfig:
    """Your fixed deployment configuration."""

    cpu: str
    memory: str
    min_instances: int
    max_instances: int
    concurrency: int
    timeout_seconds: int
    cpu_throttling: bool
    startup_cpu_boost: bool


def diagnose_failing_deployment(
    config: FailingDeploymentConfig, observed_issues: list[str]
) -> dict:
    """
    Diagnose why a deployment is failing and recommend fixes.

    Args:
        config: The failing configuration
        observed_issues: List of observed symptoms, e.g.:
            - "503 errors under load"
            - "Requests timing out"
            - "Cold starts taking 30+ seconds"
            - "Memory exceeded errors"
            - "High latency p99"

    Returns:
        dict with keys:
            - root_causes: list of identified problems
            - fixes: list of specific configuration changes
            - fixed_config: ProductionReadyConfig with fixes applied
            - explanation: markdown-formatted explanation

    Example:
        >>> issues = ["503 errors under load", "Cold starts taking 30+ seconds"]
        >>> result = diagnose_failing_deployment(FailingDeploymentConfig(), issues)
        >>> "concurrency" in str(result["fixes"])
        True
        >>> result["fixed_config"].min_instances > 0
        True
    """
    # TODO: Implement diagnosis logic
    # Map symptoms to root causes:
    # - "503 errors" -> concurrency too high, not enough instances
    # - "Timeout" -> timeout_seconds too low, or need more CPU/memory
    # - "Cold starts" -> min_instances = 0, cpu_throttling = True
    # - "Memory exceeded" -> memory too low for model
    # - "High latency" -> not enough CPU, cpu_throttling = True
    pass


# =============================================================================
# PART C: Design Decision - Cost Estimation
# =============================================================================
# Given traffic patterns, estimate cloud costs and recommend configuration.
# =============================================================================


@dataclass
class TrafficPattern:
    """Expected traffic pattern for cost estimation."""

    requests_per_day: int
    peak_rps: float  # Peak requests per second
    avg_latency_ms: float  # Average request latency
    traffic_distribution: Literal["steady", "business_hours", "spiky"]
    model_memory_mb: int  # Memory required for model
    model_load_time_seconds: float  # Time to load model


@dataclass
class CostEstimate:
    """Monthly cost estimate with breakdown."""

    monthly_cost_usd: float
    cost_breakdown: dict[str, float]  # cpu, memory, requests, networking
    recommended_config: CloudRunConfig | None
    cost_optimization_tips: list[str]
    comparison_to_alternatives: dict[str, float]  # VM, kubernetes, etc.


# Cloud Run pricing (as of 2024, US regions)
CLOUD_RUN_PRICING = {
    "cpu_per_vcpu_second": 0.000024,  # $0.000024 per vCPU-second
    "memory_per_gib_second": 0.0000025,  # $0.0000025 per GiB-second
    "requests_per_million": 0.40,  # $0.40 per million requests
    "free_tier_requests": 2_000_000,  # First 2M requests free per month
    "free_tier_cpu_seconds": 180_000,  # First 180,000 vCPU-seconds free
    "free_tier_memory_gib_seconds": 360_000,  # First 360,000 GiB-seconds free
}


def estimate_cloud_costs(
    traffic: TrafficPattern,
    config: CloudRunConfig | None = None,
) -> CostEstimate:
    """
    Estimate monthly cloud costs for deployment.

    Args:
        traffic: Expected traffic pattern
        config: Optional specific config (will recommend if None)

    Returns:
        CostEstimate with monthly cost breakdown and recommendations

    Calculation methodology:
        1. Estimate instance-seconds based on traffic pattern
        2. Account for min_instances (always-on cost)
        3. Calculate CPU and memory costs
        4. Add request costs
        5. Subtract free tier
        6. Compare to alternatives (VM, GKE)

    Example:
        >>> traffic = TrafficPattern(
        ...     requests_per_day=100000,
        ...     peak_rps=10,
        ...     avg_latency_ms=200,
        ...     traffic_distribution="business_hours",
        ...     model_memory_mb=2048,
        ...     model_load_time_seconds=30
        ... )
        >>> estimate = estimate_cloud_costs(traffic)
        >>> estimate.monthly_cost_usd > 0
        True
        >>> "cpu" in estimate.cost_breakdown
        True
    """
    # TODO: Implement cost estimation
    # Key calculations:
    # 1. Instance-seconds = requests * avg_latency_seconds * overhead_factor
    # 2. For steady traffic: instances_needed = peak_rps * latency_seconds
    # 3. For business hours: ~8 hours * 5 days = 40h/week of active usage
    # 4. min_instances cost = min_instances * seconds_per_month * (cpu + memory rates)
    # 5. Don't forget to factor in concurrency!
    pass


def compare_deployment_options(traffic: TrafficPattern) -> dict:
    """
    Compare different deployment options for the given traffic.

    Args:
        traffic: Expected traffic pattern

    Returns:
        dict with options and their costs/tradeoffs:
            - cloud_run: serverless option
            - cloud_run_min_instances: with always-on instances
            - gke_autopilot: managed kubernetes
            - compute_engine: dedicated VM
    """
    # TODO: Implement comparison logic
    # Consider:
    # - Cold start impact
    # - Scaling characteristics
    # - Operational overhead
    # - Cost at different traffic levels
    pass


# =============================================================================
# HINTS (revealed progressively with /hint command)
# =============================================================================
"""
HINT 1 - generate_cloud_run_config:
- Normalize inputs:
    if isinstance(cpu, (int, float)):
        cpu = str(int(cpu))
    if isinstance(memory, int):
        if memory >= 1024:
            memory = f"{memory // 1024}Gi"
        else:
            memory = f"{memory}Mi"

- Validation:
    valid_cpus = {"1", "2", "4", "8"}
    if cpu not in valid_cpus:
        raise CloudRunConfigError(f"CPU must be one of {valid_cpus}, got {cpu}")

    memory_mb = parse_memory(memory)
    cpu_int = int(cpu)
    min_memory_mb = cpu_int * 256  # At least 256Mi per CPU
    if memory_mb < min_memory_mb:
        raise CloudRunConfigError(f"Memory must be at least {min_memory_mb}Mi for {cpu} CPU")

HINT 2 - diagnose_failing_deployment:
- Symptom to fix mapping:
    symptom_fixes = {
        "503 errors under load": [
            ("concurrency", "Reduce from 1000 to 10-50 for ML models"),
            ("max_instances", "Increase to handle peak load"),
        ],
        "Cold starts taking 30+ seconds": [
            ("min_instances", "Set to 1+ for always-warm instances"),
            ("startup_cpu_boost", "Enable for faster model loading"),
            ("cpu_throttling", "Disable to keep model warm"),
        ],
        "Memory exceeded errors": [
            ("memory", "Increase to 2Gi+ for ML models"),
        ],
        "Requests timing out": [
            ("timeout_seconds", "Increase to 60-300 for ML inference"),
            ("cpu", "Increase for faster inference"),
        ],
    }

HINT 3 - estimate_cloud_costs:
- Instance-seconds calculation:
    seconds_per_request = traffic.avg_latency_ms / 1000
    total_requests_per_month = traffic.requests_per_day * 30
    billable_seconds = total_requests_per_month * seconds_per_request

    # Account for concurrency
    instances_per_request = 1 / config.concurrency
    instance_seconds = billable_seconds * instances_per_request

    # Add min_instances cost (always-on)
    seconds_per_month = 30 * 24 * 3600
    min_instance_seconds = config.min_instances * seconds_per_month

    total_cpu_seconds = instance_seconds + min_instance_seconds
    total_memory_gib_seconds = total_cpu_seconds * (memory_mb / 1024)

- Apply pricing:
    cpu_cost = max(0, total_cpu_seconds - CLOUD_RUN_PRICING["free_tier_cpu_seconds"]) * CLOUD_RUN_PRICING["cpu_per_vcpu_second"]
    memory_cost = max(0, total_memory_gib_seconds - CLOUD_RUN_PRICING["free_tier_memory_gib_seconds"]) * CLOUD_RUN_PRICING["memory_per_gib_second"]
"""

# =============================================================================
# Test your implementation (run with: python exercise_3_4.py)
# =============================================================================
if __name__ == "__main__":
    print("Exercise 3.4: Cloud Deployment")
    print("=" * 50)

    # Test Part A - Config generation
    print("\nPart A - Testing generate_cloud_run_config:")
    try:
        config = generate_cloud_run_config(cpu=2, memory="4Gi", min_instances=1, max_instances=10)
        if config:
            print(f"  Generated config: CPU={config.cpu}, Memory={config.memory}")
            print("  [PASS] Successfully generated config")
        else:
            print("  [TODO] Function returns None")
    except CloudRunConfigError as e:
        print(f"  [FAIL] Validation error: {e}")
    except Exception as e:
        print(f"  [TODO] generate_cloud_run_config not implemented: {e}")

    # Test invalid config
    try:
        config = generate_cloud_run_config(cpu=3, memory="128Mi")  # Invalid CPU
        print("  [FAIL] Should reject CPU=3")
    except CloudRunConfigError:
        print("  [PASS] Correctly rejects invalid CPU value")
    except Exception as e:
        print(f"  [TODO] Validation not implemented: {e}")

    # Test Part B - Diagnosis
    print("\nPart B - Testing diagnose_failing_deployment:")
    try:
        issues = ["503 errors under load", "Cold starts taking 30+ seconds"]
        result = diagnose_failing_deployment(FailingDeploymentConfig(), issues)
        if result:
            print(f"  Found {len(result['root_causes'])} root causes")
            if result["fixed_config"].min_instances > 0:
                print("  [PASS] Recommends min_instances > 0")
            if result["fixed_config"].concurrency < 100:
                print("  [PASS] Recommends lower concurrency")
        else:
            print("  [TODO] Function returns None")
    except Exception as e:
        print(f"  [TODO] diagnose_failing_deployment not implemented: {e}")

    # Test Part C - Cost estimation
    print("\nPart C - Testing estimate_cloud_costs:")
    try:
        traffic = TrafficPattern(
            requests_per_day=100000,
            peak_rps=10,
            avg_latency_ms=200,
            traffic_distribution="business_hours",
            model_memory_mb=2048,
            model_load_time_seconds=30,
        )
        estimate = estimate_cloud_costs(traffic)
        if estimate:
            print(f"  Estimated monthly cost: ${estimate.monthly_cost_usd:.2f}")
            print(f"  Cost breakdown: {estimate.cost_breakdown}")
            if estimate.monthly_cost_usd > 0:
                print("  [PASS] Generated cost estimate")
        else:
            print("  [TODO] Function returns None")
    except Exception as e:
        print(f"  [TODO] estimate_cloud_costs not implemented: {e}")

    print("\n" + "=" * 50)
    print("Run pytest test_level_3.py for full test coverage")
