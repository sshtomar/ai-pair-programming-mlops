# Get Hint for Exercise

Provide progressive hints for a specific exercise without giving away the answer.

## Usage

- `/hint 1.2` - Get hint 1 for lesson 1.2 exercises
- `/hint 1.2.1` - Get hint 1 for specific exercise
- `/hint 1.2.1 2` - Get hint 2 for specific exercise
- `/hint 1.2.1 3` - Get hint 3 (most detailed) for specific exercise

## Behavior

1. Parse $ARGUMENTS to determine exercise and hint level
2. Read the corresponding exercise file
3. Find the HINTS section at the bottom
4. Return the appropriate hint level

## Hint Philosophy

```
HINT 1: Conceptual direction
        "Think about what data type the function should return"

HINT 2: More specific guidance
        "The function should return a dictionary with keys 'passed' and 'errors'"

HINT 3: Nearly the answer
        "Use isinstance() to check the type, then build the dict like {'passed': True, 'errors': []}"
```

## Response Format

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                           HINT: Exercise X.Y.Z                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

Hint 1 of 3:

[Hint content here]

─────────────────────────────────────────────────────────────────────────────
Still stuck? Try `/hint X.Y.Z 2` for a more specific hint.
```

## Implementation

1. Parse exercise reference from $ARGUMENTS
2. Determine hint level (default 1)
3. Read exercise file: `project/exercises/{level}-{name}/exercise_{lesson}.py`
4. Extract hint from HINTS section
5. Format and display

## Arguments

$ARGUMENTS - Exercise reference and optional hint level (e.g., "1.2.1" or "1.2.1 2")
