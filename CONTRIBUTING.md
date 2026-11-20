# Contributing

## Setup

```bash
git clone https://github.com/snapetech-org/rigtune_amd.git && cd rigtune_amd
pip install -e ".[dev]"
pytest                                  # no GPU or backend needed
ruff check rigtune tests benchmarks
```

CI runs on Python 3.10–3.13 for every push and pull request. The package installs as `rigtune-amd`; `rigtune` remains the compatible command name, and `rigtune-amd` is an alias.

## Tests

- Every test runs without a GPU, a backend or network access. Backends are driven through fake servers and fake hardware descriptors (see the fixtures in `tests/conftest.py`).
- Tests must not depend on the order a filesystem lists a directory in, or on anything else that varies by machine. Sort, and break ties explicitly.
- A bug found on real hardware gets a test that reproduces it with the numbers that were measured (`tests/test_combinations.py` is an example).

## Changing the search

Recorded calibrations are evidence for the search as well as for the numbers. `PYTHONPATH=. python benchmarks/replay_search.py <profile.json>...` replays each through the current search with no GPU, answering every trial from the profile's own table. A profile is *exact* when the search asks for exactly the recorded trials and reaches the recorded pick, *inconclusive* when it reaches the pick past trials that were never measured (or under an option the replay cannot apply), and *different* otherwise. The script exits non-zero when a profile does worse than the class it had when the script was written (`BASELINE`); `tests/test_replay_recorded.py` runs it over every recorded profile. Only exact replays say the search is unchanged on that data. Budgeted calibrations are not replayed, since their stages were cut by wall-clock time.

## Adding a backend

Implement the `Backend` interface in [docs/usage.md](docs/usage.md#backend-interface), register it with the selector, and give it a memory model the planner can use. The core pipeline (probe, plan, calibrate, serve) does not change.

AMD GPU validation uses a local llama.cpp HIP/ROCm build; CPU-only tests and packaging run on public GitHub-hosted runners. Vulkan support is exposed when a Vulkan-enabled llama.cpp build is installed. Do not add a self-hosted GPU runner requirement to public pull-request workflows.

## Benchmark claims

A number goes into [docs/benchmarks.md](docs/benchmarks.md) only if it was measured like this:

1. rigtune's pick against stock settings on the same card, minutes apart: `rigtune compare <model> --workload W`. For any gain under about 10%, add `--repeats 3`: rows are measured interleaved, and a gain whose runs overlap the stock row's is flagged as within noise.
2. What each strategy was worth, flipped one at a time and measured back to back: `benchmarks/ablate_strategies.py`.
3. Any change of weights checked for quality: `benchmarks/task_quality.py` (GSM8K, paired against bf16).
4. The raw JSON committed under `benchmarks/strategies/`, and `SUMMARY.md` regenerated with `benchmarks/summarize_strategies.py` (the command is in its docstring).

State the GPU, engine versions and date. Report failures, losses and ties as well as wins; a strategy that lost is a result.

## Releasing

The release workflow builds and validates a distribution artifact on a GitHub-hosted runner. This fork does not publish to PyPI.

llmtrace, the GPU tracer rigtune prefers for telemetry, is not a dependency on PyPI: the name `llmtrace` there belongs to an unrelated project, and PyPI refuses git dependencies. Without it rigtune samples the GPU through `pynvml` (the `nvml` extra) or falls back to process-level `psutil` numbers.

## Style

- `ruff`, line length 110.
- Comments say why, often with the measurement that motivated the code.
- Commit messages: one short imperative line.

## License

By contributing you agree that your contributions are licensed under the MIT license.
