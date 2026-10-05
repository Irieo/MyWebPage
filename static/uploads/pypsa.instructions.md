# Project Overview

PyPSA is an open-source Python framework for optimising and simulating modern
power and energy systems that include features such as conventional generators
with unit commitment, variable wind and solar generation, hydro-electricity,
inter-temporal storage, coupling to other energy sectors, elastic demands, and
linearised power flow with loss approximations in DC and AC networks. PyPSA is
designed to scale well with large networks and long time series. It can do:

- Economic Dispatch
- Linear Optimal Power Flow
- Security-Constrained LOPF
- Capacity Expansion Planning
- Pathway Planning
- Stochastic Optimization
- Modelling-to-Generate-Alternatives
- Sector-Coupling

## Architecture

PyPSA represents different components (e.g. generators, lines, buses) as objects in a network model.
The data for these components is stored in Pandas DataFrames.
The optimization problems are automatically formulated based on the data stored in the DataFrames.
That means standard components are associated with standard equations.
The optimization problems are built with linopy framework, which leans more heavily on xarray.

## Installation

```bash
uv venv
uv sync --all-extras
source .venv/bin/activate
```

## Testing

Run all tests:
```bash
uv run pytest
```

Run a specific test file:
```bash
uv run pytest test/test_stochastic.py
```

Run a specific test:
```bash
uv run pytest test/test_stochastic.py::test_stoch_example
```

## Code Quality

Run the linter:
```bash
ruff check pypsa
```

Run static type checking:
```bash
mypy pypsa
```

## Usage

```python
import pypsa

# create a new network
n = pypsa.Network()
n.add("Bus", "mybus")
n.add("Load", "myload", bus="mybus", p_set=100)
n.add("Generator", "mygen", bus="mybus", p_nom=100, marginal_cost=20)

# load an example network
n = pypsa.examples.ac_dc_meshed()

# run the optimisation
n.optimize()

# run a power flow
n.pf()

# access and plot results
n.generators_t.p.plot()
n.plot()

# get statistics
n.statistics()
n.statistics.energy_balance()
```