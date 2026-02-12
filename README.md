# verifiers-interact

**Does hand-engineered observation compression outperform forcing the model to learn navigation under naive truncation?**

A plugin for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) that enforces observation budgets on `RLMEnv` tool output, enabling controlled ablation studies on how context pressure shapes RLM search behavior.

## Thesis

Today's RLMs see everything. When a REPL dumps 500 lines, the model gets all 500 lines. This substitutes *context window* for *compute* — the model scans instead of searching.

The [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) predicts that methods leveraging computation will always beat methods leveraging human knowledge. Applied to RLM observation:

- **Budget constraints** are Bitter Lesson-compatible — they force the model to spend inference compute on targeted queries rather than passively consuming large outputs
- **Hand-engineered compression** (extracting function signatures, keeping head+tail) encodes human priors about what's "important" — Sutton's argument predicts this helps short-term but loses at scale to models that learn their own navigation strategies

`verifiers-interact` separates these two decisions so you can test them independently:

```python
from verifiers_interact import NavigationEnv, LineLimit
from verifiers_interact.folders import TruncateFolder, StructureFolder

# Bitter Lesson purist: dumb truncation, model must learn to navigate
env_bitter = NavigationEnv(constraint=LineLimit(50, folder=TruncateFolder()))

# Human prior: we extract structure for the model
env_prior = NavigationEnv(constraint=LineLimit(50, folder=StructureFolder()))

# Same budget. Which learns better search heuristics at scale?
```

## Install

```bash
uv sync
```

## Architecture

`NavigationEnv` subclasses `RLMEnv` and overrides two methods:

- **`env_response`** — post-processes tool output through the constraint+folder pipeline before the model sees it
- **`add_trajectory_step`** — injects navigation telemetry into trajectory extras for post-hoc analysis

It also overrides `setup_state` to initialize per-rollout telemetry counters and auto-attaches `NavigationMonitorRubric` for WandB metrics.

This is a ~500 LOC plugin, not a fork. One import, one extra kwarg on top of your existing `RLMEnv` setup.

## Concepts

### Constraints (the budget)

A constraint decides **how much** output the model sees. This is the Bitter Lesson mechanism — it creates search pressure.

| Constraint | What it does | Bitter Lesson? |
|---|---|---|
| `LineLimit(n)` | Cap output at `n` lines | ✅ Yes — forces compute |
| `TokenBudget(n)` | Cap output at `n` characters (~n/4 tokens) | ✅ Yes — forces compute |
| `Unconstrained()` | No-op passthrough | Control group |

### Folders (the compression strategy)

When a constraint triggers, a folder decides **how** to compress. This is where the Bitter Lesson tension lives.

| Folder | What the model sees | Bitter Lesson? |
|---|---|---|
| `TruncateFolder` | First N lines, rest discarded | ✅ Purist — zero human priors |
| `HeadTailFolder(r)` | First r% + last (1-r)% of output | ❌ Encodes "start and end matter more" |
| `StructureFolder` | `def`/`class`/`import` signatures extracted via regex | ❌ Encodes human knowledge of code structure |

Constraints and folders compose independently — enabling **N×M factorial ablation** across budgets and compression strategies.

### Profiles (preset configurations)

`ToolProfile` provides ready-made configurations for common experiment setups:

| Profile | Constraint | Folder | Max Iterations | Intent |
|---|---|---|---|---|
| `minimal()` | `LineLimit(50)` | `StructureFolder` | 100 | Maximum search pressure + human priors |
| `standard()` | `LineLimit(200)` | `TruncateFolder` | 50 | Balanced Bitter Lesson baseline |
| `power()` | `TokenBudget(16K)` | `HeadTailFolder` | 30 | Generous budget with head+tail prior |
| `unconstrained()` | `Unconstrained()` | — | 50 | Pure control group |

```python
from verifiers_interact import NavigationEnv, ToolProfile

env = NavigationEnv(**ToolProfile.standard(), dataset=ds, rubric=rubric)
```

### Telemetry

`NavigationMonitorRubric` automatically tracks and exports to WandB:

| Metric | Description |
|---|---|
| `nav_truncation_count` | How many tool outputs exceeded the budget |
| `nav_truncation_rate` | Fraction of outputs that triggered compression |
| `nav_lines_hidden` | Total lines withheld from the model |
| `nav_chars_hidden` | Total characters withheld |
| `nav_tool_output_count` | Total tool outputs processed |
| `nav_constraint_type` | Which constraint was active |

Per-step `nav_stats` are injected into trajectory extras via `add_trajectory_step`, enabling post-hoc analysis of how navigation behavior evolves during training.

## Demo

Run the ablation demo to see all four observation strategies applied to the same 80-line Python file:

```bash
uv run python examples/ablation_demo.py
```

This shows, side-by-side, what the model sees under `Unconstrained`, `TruncateFolder`, `HeadTailFolder`, and `StructureFolder` — all with the same 15-line budget.

## Test

```bash
uv run pytest -v
```
