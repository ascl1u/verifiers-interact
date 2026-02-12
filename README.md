# verifiers-interact

**Context management as a search problem.**

A plugin for [Prime Intellect's verifiers](https://github.com/PrimeIntellect-ai/verifiers) that gives RLMs strict observation budgets and lets RL discover how to navigate within them.

## Thesis

Today's RLMs see everything. When a tool returns 500 lines, the model gets all 500 lines. This is the opposite of the Bitter Lesson — it substitutes *context window* for *compute*, encouraging the model to scan rather than search.

`verifiers-interact` enforces **observation constraints** on RLMEnv tool output. Instead of showing the model everything and hoping it learns to skim, we show it *almost nothing* and force it to learn precise navigation.

The hypothesis: an RLM trained under `LineLimit(50)` will develop stronger search heuristics than one trained with unlimited observation — and those heuristics will transfer to harder tasks.

### The Two Decisions

Every observation constraint makes two decisions:

1. **How much to show** — the budget (`LineLimit`, `TokenBudget`)
2. **How to compress the rest** — the folding strategy (`TruncateFolder`, `HeadTailFolder`, `StructureFolder`)

These compose independently, enabling factorial ablation studies:

```python
from verifiers_interact import LineLimit, NavigationEnv
from verifiers_interact.folders import StructureFolder, HeadTailFolder

# Same budget, different folding → isolate the folding variable
env_naive = NavigationEnv(constraint=LineLimit(50))
env_struct = NavigationEnv(constraint=LineLimit(50, folder=StructureFolder()))
env_split  = NavigationEnv(constraint=LineLimit(50, folder=HeadTailFolder(0.7)))
```

### Folding Strategies

| Strategy | What the model sees | Best for |
|----------|-------------------|----------|
| `TruncateFolder` | First N lines + "[TRUNCATED]" | Baseline |
| `HeadTailFolder` | First 60% + last 40% + "[... N lines elided ...]" | Logs, sequential output |
| `StructureFolder` | `def`/`class`/`import` signatures + "[FOLDED]" | Code navigation |

The `StructureFolder` is the key research primitive. It gives the model a **table of contents** — a structural map it can use to issue precise follow-up queries — rather than a wall of truncated text. The model trades *reading* for *searching*.

## Install

```bash
uv sync
```

## Usage

```python
from verifiers_interact import NavigationEnv, LineLimit, ToolProfile

# Custom constraint with folding
env = NavigationEnv(
    constraint=LineLimit(200, folder=StructureFolder()),
    dataset=my_dataset,
    rubric=my_rubric,
)

# Or use a preset profile
env = NavigationEnv(**ToolProfile.standard(), dataset=my_dataset, rubric=my_rubric)
```

### Profiles

| Profile | Constraint | Folder | Iterations | Use case |
|---------|-----------|--------|------------|----------|
| `minimal()` | `LineLimit(50)` | `StructureFolder` | 100 | Maximum search pressure |
| `standard()` | `LineLimit(200)` | `TruncateFolder` | 50 | Balanced training |
| `power()` | `TokenBudget(16K)` | `HeadTailFolder` | 30 | Generous baseline |
| `unconstrained()` | None | — | 50 | Pure control group |

### Telemetry

`NavigationMonitorRubric` automatically exports to WandB:

- `nav_truncation_count` — how often the constraint fired
- `nav_truncation_rate` — fraction of outputs that exceeded budget
- `nav_lines_hidden` / `nav_chars_hidden` — total information withheld
- Per-step `nav_stats` in trajectory extras for post-hoc analysis

## Architecture

`NavigationEnv` subclasses `RLMEnv`. It inherits the full sandbox lifecycle and overrides exactly two methods:

- **`env_response`** — post-processes tool output through the constraint+folder pipeline
- **`add_trajectory_step`** — injects navigation telemetry into trajectory extras

This is a ~500 LOC plugin, not a fork. One import, one extra kwarg.

## Test

```bash
uv run pytest -v
```
