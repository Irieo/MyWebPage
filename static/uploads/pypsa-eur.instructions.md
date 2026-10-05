# PyPSA-Eur Coding Guidelines

PyPSA-Eur is an open-source energy system model that focuses on European electricity and sector-coupling analysis.
It is mostly used for infrastructure planning applications, including capacity expansion and pathway planning. It can also be used for operational studies such as unit commitment and economic dispatch.

## Project Architecture

- **PyPSA Framework**: The model is built on the PyPSA framework
- **Workflow Management**: Uses Snakemake for orchestrating the Python scripts, inputs and outputs linked via so-called "rules"

## Code Organization

- `scripts/*.py`: Python scripts for data processing and network creation
- `rules/*.smk`: Snakemake rule definitions connecting Python scripts
- `config/*.yaml`: Configuration files, only change `config.default.yaml`.
- `scripts/_helpers.py`: Helper functions for scripts

## Coding Quality

Run the linter:
```bash
ruff check
```

Use type hints for function signatures, especially for public functions:

```python
def determine_emission_sectors(options: dict) -> list[str]:
    """Documentation here"""
    # Implementation
```

Log significant operations with the logger but don't overdo it.

```python
logger.info(f"<notifications> {variables_to_print}")
```

New files need should have this structure:

```python
# SPDX-FileCopyrightText: : 2025 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""DOCSTRING"""

from scripts._helpers import configure_logging, set_scenario_config

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("<rulename>")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
```

Ignore that the `snakemake` object is not defined (add a linting exception). This is added automatically by Snakemake during execution.

Use `df.query()` preferably for simple enough filtering operations (more concise).

## Environments

Uses Pixi for environment management.

## Validation

Uses a Pydantic model for validating configuration files. The `config.default.yaml` and `schema.default.json` files are created with `pixi run generate-config`. Never edit automatically. Follow instructions on `validation_dev.md`.

## Documentation

- Every rule in `rules/*.smk` carries a one-line docstring directly after the `rule` line: one sentence, third person present tense, starts with a verb, at most 100 characters, no config keys or paths. It doubles as the job message printed by Snakemake, so rules have no `message:` block.
- Every script has a module-level docstring in markdown: one sentence on what it produces, then two to five sentences on the method (data in, what is computed, key assumptions). Do not list inputs, outputs or config keys; the rules pages generate them from the Snakemake rule. An optional `References` section lists literature as markdown links.
- The rules pages under `doc/rules/` list rules with the `rules()` macro from `doc/macros.py`; a new rule must be added to one of them (checked by `test/test_docs.py`).
- Any new documentation should closely follow the existing style of the documentation (language and format).

## Long-term Development Goals

- Improve modularity of the code (shorter scripts, self-contained functions without global variables)
- Clearer separation of data cleaning and model building
- Simplify the code and workflow structure
- Enhance documentation, especially in docstrings
- Improve type annotation coverage (mypy)
- Unify the workflow for electricity-only and sector-coupled models (e.g. single way to add powerplants)
- Reduce code duplication

## Best Practices

- Use consistent suffixes (e.g., " CC" for carbon capture)
- Any functional change should be documented in `release_notes.md`
- New data sources need to be added following instructions in `data_sources.md`.
- Do not use `snakemake.config`. These should be passed as parameters to rules (`snakemake.params`).
