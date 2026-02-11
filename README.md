# verifiers-interact

Observation constraints and navigation telemetry for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) `RLMEnv`.

## Install

```bash
uv sync
```

## Quick Start

```python
from verifiers_interact import NavigationEnv, LineLimit, ToolProfile

# Custom constraint
env = NavigationEnv(
    constraint=LineLimit(200),
    dataset=my_dataset,
    rubric=my_rubric,
)

# Or use a preset profile
env = NavigationEnv(**ToolProfile.standard(), dataset=my_dataset, rubric=my_rubric)
```

## Profiles

| Profile | Constraint | Max Iterations | Use Case |
|---------|-----------|----------------|----------|
| `minimal()` | `LineLimit(50)` | 100 | Forces maximum search compute |
| `standard()` | `LineLimit(200)` | 50 | Default for training |
| `power()` | `TokenBudget(16000)` | 30 | Generous context budget |
| `unconstrained()` | None | 50 | Baseline comparison |

## Test

```bash
uv run pytest
```
