# Check Exercise Progress

Validate the student's exercise solutions and provide feedback.

## Behavior

1. Parse the argument to determine what to check:
   - `/check` - Check all exercises
   - `/check 1` - Check all Level 1 exercises
   - `/check 1.2` - Check specific lesson exercises
   - `/check 2.3.1` - Check specific exercise

2. Run pytest on the appropriate test file(s)

3. Provide friendly feedback:
   - Show which tests passed/failed
   - For failures, give hints without revealing answers
   - Celebrate progress!

## Implementation

When the user runs `/check`:

```bash
# Determine what to test based on $ARGUMENTS
# If empty, run all tests
# If "1", run test_level_1.py
# If "1.2", run tests for lesson 1.2 only
# If "1.2.1", run specific test

cd project/exercises

# Run appropriate pytest command
pytest <target> -v --tb=short
```

## Response Format

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        EXERCISE PROGRESS CHECK                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

Checking: [Level X / Lesson X.Y / Exercise X.Y.Z]

Results:
├── ✓ exercise_X_Y.py::test_function_1 PASSED
├── ✓ exercise_X_Y.py::test_function_2 PASSED
├── ✗ exercise_X_Y.py::test_function_3 FAILED
│   └── Hint: Check the return type - should be a dict, not a list
└── ○ exercise_X_Y.py::test_function_4 SKIPPED (depends on test_function_3)

Progress: ██████░░░░ 60% (3/5 passed)

Next step: Fix the failing test in exercise_X_Y.py
           Need a hint? Try: /hint 1.2.1
```

## After Running Tests

1. Parse pytest output
2. For each failure:
   - Read the corresponding exercise file
   - Find the relevant HINT section
   - Provide HINT 1 only (let them ask for more)
3. Update the course progress tracker if all tests pass
4. Suggest next steps

## Arguments

$ARGUMENTS - Optional: what to check (e.g., "1", "1.2", "1.2.1", or empty for all)
