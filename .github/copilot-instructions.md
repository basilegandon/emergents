# Emergents: AI Coding Workflow

**Project**: Genetic evolution simulation framework with configurable mutations and population dynamics.

## Workflow: Strict Staged Process

1. **Discovery**: Review `.context/` folder for active tasks, blueprints, research, and prior work
2. **Analysis** (if needed): Understand requirements before coding; produce research docs or blueprints
3. **Implementation**: Only after Discovery/Analysis complete; produce working code with tests
4. **Output**: Always deliver focused Markdown artifacts (task summaries, diagnostics, blueprints)

**Rule**: Never write code until all prior stages documented in `.context/`.

---

## Project Architecture at a Glance

### Core Components

- **Genome**: Implicit treap of Segment nodes (BASE/GAP coordinates); efficient edit operations
- **Mutations**: 7 types (Point, SmallDeletion, SmallInsertion, Deletion, Duplication, Inversion)
- **Population**: Selection, replication, statistics tracking across generations
- **SimulationService**: Orchestrator (initialize → evolve → report workflow)
- **Statistics & Plotting**: Decoupled modules for analysis and visualization

### Key File Organization

```
src/emergents/
  ├── genome/         # Genome representation (treap-based segments)
  ├── mutations/      # Mutation implementations (apply, is_neutral, serialize, describe)
  ├── config.py       # Dataclass-based configuration objects
  ├── population.py   # Population management & evolution logic
  ├── simulation_service.py  # Main orchestrator
  ├── statistics.py   # Stats collection & reporting
  └── file_plotter.py # Visualization (matplotlib, multiprocessing)
tests/
  └── test_*.py       # Pytest files; coverage target: 90%
.context/
  ├── *_task.md       # Active/done task tracking
  ├── *_blueprint.md  # Implementation plans
  └── *_research.md   # Analysis & findings
```

---

## Essential Commands

### Testing

```bash
# Run all tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_genome.py -v

# Run with coverage report
pytest --cov=src/emergents --cov-report=html

# Run in parallel (faster)
pytest -n auto
```

### Code Quality

```bash
# Lint and fix with ruff (configured: E, F, I, B, UP, N, SIM, C4, PT)
ruff check src/ tests/ --fix
ruff format src/ tests/

# Run the simulation
python main.py
```

---

## Code Patterns & Conventions

### Configuration

- All configuration is dataclass-based (see `config.py`)
- Immutable after creation; use `.replace()` for modifications
- Example: `GenomeConfig`, `MutationConfig`, `EvolutionConfig`, `SimulationConfig`

### Mutations

- Implement abstract `Mutation` from `mutations/base.py`
- Required methods: `apply(genome)`, `is_neutral()`, `serialize()`, `describe()`
- Apply mutations to genome via `MutationManager.apply_mutation()`

### Genome Operations

- Genomes are implicit treap structures (don't serialize the tree)
- Use `Segment` nodes with explicit BASE/GAP coordinate validation
- Mutations modify segments via coordinate transformations
- Always validate coordinates post-mutation

### Testing Practices

- **Behavior-driven**: Test public APIs and outcomes, not private methods
- **Parametrized tests**: Use `@pytest.mark.parametrize` for multiple cases
- **Fixtures**: Custom `make_segments()` factory for test setup
- **Avoid**: Mocking implementation details (e.g., multiprocessing queue internals)
- **Coverage target**: 90% (currently 87.35%)
- **Known limitations**: file_plotter.py has process isolation challenges; use exemptions if needed

---

## .context/ Folder Convention

Each active work item has a numbered prefix:

**Task format**: `{number}_active_task.md` or `{number}_done_task.md`

- Title, Status, Priority, Effort
- Problem statement, acceptance criteria, implementation notes

**Blueprint format**: `{number}_blueprint.md`

- Step-by-step plan with file locations, line ranges, and code samples
- Reference in task before building (e.g., "See blueprint in .context/2_blueprint.md")

**Research format**: `{number}_research.md`

- Findings, audit results, analysis that informs task design

**Always check active tasks first** before starting work.

---

## Development Practices

### Before Implementation

1. Check `.context/` for related tasks, blueprints, research
2. If unclear: produce a `*_research.md` document with findings
3. If planning complex work: produce a `*_blueprint.md` with step-by-step plan

### During Implementation

- Follow ruff lint rules (fixable: ALL, ignored: E501, PT001)
- Write behavior-driven tests alongside code
- Maintain 90% coverage target
- Run `pytest -v --cov` to verify

### Artifacts to Produce

- Code changes (src/ or tests/)
- Updated task status in `.context/`
- Research/blueprint docs if discovering or planning
- Do NOT create separate summary files unless explicitly asked

---

## Debugging & Diagnostics

For test failures or coverage gaps:

- Run `pytest -v --tb=short` to see failure details
- Check `.context/` tasks for related issues (e.g., "Close Coverage Gap to 90%")
- Use `pytest --cov --cov-report=html` to identify untested lines
- Verify by running tests: if all pass, implementation is sound

---

## Quality Standards

- **Python**: 3.14+
- **Dependencies**: matplotlib, rich, tqdm (runtime); pytest, ruff (dev)
- **Linting**: Ruff (E, F, I, B, UP, N, SIM, C4, PT)
- **Testing**: Pytest with pytest-cov, pytest-mock, pytest-xdist
- **Documentation**: Minimal; code clarity + tests = documentation

---

## Quick Reference

| Need              | Command/Action                                                                     |
| ----------------- | ---------------------------------------------------------------------------------- |
| Run tests         | `pytest -v`                                                                        |
| Check coverage    | `pytest --cov=src/emergents --cov-report=html`                                     |
| Fix linting       | `ruff check --fix && ruff format`                                                  |
| Run simulation    | `python main.py`                                                                   |
| Check active work | Review `.context/` folder                                                          |
| Understand genome | Read [genome/genome.py](../src/emergents/genome/genome.py) & its tests             |
| Add mutation type | Extend `Mutation` class in [mutations/base.py](../src/emergents/mutations/base.py) |
