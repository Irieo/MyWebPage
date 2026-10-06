"""
Workaround for ler-scenario KVL for stochastic PyPSA networks with uncertain grid build-out.

PyPSA builds a single Kirchhoff-Voltage-Law (KVL) cycle basis, weighted with
the reactances of the first scenario, and applies it to all scenarios. A line
with s_nom = 0 then stays in the cycle basis: its flow is fixed to zero, so KVL
locks the voltage angles of its two endpoints together.

Here we use ``ac_dc_meshed`` for a toy model. Brownfield grid plus four candidate AC
lines (say, NEP/TYNDP projects), with three branches of grid realisation. First-stage
decisions are investments in backup gas plants and batteries.

IR 06.10.2026
"""

# %% Setup: workaround, toy model and helpers
import logging
import warnings
import numpy as np
import pandas as pd
from xarray import DataArray
import pypsa

# quiet logs and warnings
logging.disable(logging.WARNING)
warnings.simplefilter("ignore", FutureWarning)

# =============================================================================
# >>> SHAREABLE WORKAROUND: per-scenario KVL. Copy from here ...
# Usage: n.optimize(extra_functionality=add_scenario_kvl)
# =============================================================================


def add_scenario_kvl(n: pypsa.Network, sns: pd.Index) -> None:
    """Replace PyPSA's KVL by one KVL constraint per scenario.

    Builds each scenario's cycle matrix with PyPSA on a deep copy of that
    scenario, without its lines fixed at zero capacity. The copy includes the
    linopy model, so peak memory roughly doubles. Supports lines in
    single-period models.

    Requires PyPSA newer than v1.3.0: PyPSA/PyPSA#1951 shortcuts some code.
    """
    if not n.c.transformers.static.empty or n.has_investment_periods:
        msg = "add_scenario_kvl supports lines in single-period models only."
        raise NotImplementedError(msg)
    m = n.model
    if "Kirchhoff-Voltage-Law" in m.constraints:
        m.remove_constraints("Kirchhoff-Voltage-Law")
    for s in n.scenarios:
        n_s = n.get_scenario(s)
        lines = n_s.c.lines.static
        n_s.remove("Line", lines.index[(lines.s_nom == 0) & ~lines.s_nom_extendable])
        C = n_s.cycle_matrix(apply_weights=True)
        if C.empty:
            continue
        flow = m["Line-s"].sel(scenario=s, snapshot=sns, name=C.loc["Line"].index)
        m.add_constraints(
            flow @ DataArray(C.loc["Line"]) * 1e5 == 0,
            name=f"Kirchhoff-Voltage-Law-{s}",
        )


# =============================================================================
# <<< ... to here.
# =============================================================================


CANDIDATES = pd.DataFrame(
    {
        "bus0": ["London", "Bremen", "Bremen", "Frankfurt"],
        "bus1": ["Norwich", "Frankfurt", "Norway", "Norway"],
        "x": [0.2388, 0.4, 0.8, 1.2],
        "s_nom": [400.0, 600.0, 800.0, 800.0],
    },
    index=[
        "NEP London-Norwich",
        "NEP Bremen-Frankfurt",
        "TYNDP Bremen-Norway",
        "TYNDP Frankfurt-Norway",
    ],
)
BRANCHES = {
    "existing-lines": [],
    "2/4 realised": ["NEP London-Norwich", "TYNDP Bremen-Norway"],
    "4/4 realised": CANDIDATES.index.tolist(),
}
SOLVER = {"solver_name": "highs", "io_api": "lp", "output_flag": False}


def brownfield() -> pypsa.Network:
    """Build the brownfield grid from ``ac_dc_meshed``.

    Existing assets get fixed capacities. Candidate lines are added at zero capacity,
    and backup gas plants and batteries are first-stage investment options.
    """
    # We use ac_dc_meshed example, also drop CO2 cap
    n = pypsa.examples.ac_dc_meshed()
    n.remove("GlobalConstraint", n.c.global_constraints.static.index)
    # readable names for the existing AC lines (DC lines keep 2, 3, 4)
    names = {"0": "London-Manchester", "1": "Manchester-Norwich",
             "5": "London-Norwich", "6": "Bremen-Frankfurt"}  # fmt: skip
    n.c.lines.static = n.c.lines.static.rename(index=names)

    # Trim down existing capacity. With default example numbers, the new candidates are worthless,
    lines, links, gens = n.c.lines.static, n.c.links.static, n.c.generators.static
    s_nom = pd.Series([500, 500, 400, 600], index=list(names.values()))
    lines["s_nom"] = s_nom.reindex(lines.index, fill_value=300)
    links["p_nom"] = np.where(links.index == "DC link", 200, 300)
    gens["p_nom"] = np.where(gens.carrier == "wind", 1000, 1500)

    # brownfield: existing assets cannot be expanded
    for df, attr in ((lines, "s_nom"), (links, "p_nom"), (gens, "p_nom")):
        df[f"{attr}_extendable"] = False

    # NEP/TYNDP candidates at zero capacity; stochastic() realises them per branch
    c = CANDIDATES
    n.add("Line", c.index, bus0=c.bus0, bus1=c.bus1, x=c.x, r=0.01, s_nom=0)

    # first-stage investment options: backup gas and batteries at the load hubs
    # without own generation
    hubs = pd.Index(["London", "Norwich", "Bremen"])
    gas = {"carrier": "gas", "capital_cost": 60, "marginal_cost": 80}
    battery = {"carrier": "battery", "capital_cost": 40, "max_hours": 2}
    n.add("Generator", hubs + " backup", bus=hubs, p_nom_extendable=True, **gas)
    n.add("StorageUnit", hubs + " battery", bus=hubs, p_nom_extendable=True,
          efficiency_store=0.9, efficiency_dispatch=0.9, **battery)  # fmt: skip
    return n


def stochastic(per_scenario_kvl: bool) -> pypsa.Network:
    """Two-stage problem: candidates at s_nom = 0, realised per branch."""
    n = brownfield()
    n.set_scenarios(dict.fromkeys(BRANCHES, 1 / len(BRANCHES)))
    for s, realised in BRANCHES.items():
        rows = [(s, line) for line in realised]
        n.c.lines.static.loc[rows, "s_nom"] = CANDIDATES.s_nom[realised].values
    extra = add_scenario_kvl if per_scenario_kvl else None
    n.optimize(extra_functionality=extra, **SOLVER)
    return n


def deterministic(n: pypsa.Network, scenario: str) -> pypsa.Network:
    """Branch ``scenario`` solved alone on its true grid, with n's investments."""
    n.model.solver_model = None
    d = n.get_scenario(scenario)
    d.optimize.fix_optimal_capacities()
    d.remove("Line", d.c.lines.static.query("s_nom == 0").index)
    d.optimize(**SOLVER)
    return d


# --- post-processing ---------------------------------------------------------


def kvl_weights(n: pypsa.Network) -> pd.Series:
    """Weight of each line in each scenario's KVL, read from the linopy model."""
    m = n.model
    names = [c for c in m.constraints if c.startswith("Kirchhoff-Voltage-Law")]
    terms = pd.concat([m.constraints[c].to_polars().to_pandas() for c in names])
    coords = [pos for _, pos in m.variables.get_label_position(terms.vars.values)]
    coords = pd.DataFrame(coords)
    weights = pd.Series(terms.coeffs.abs().values / 1e5)
    return weights.groupby([coords.scenario, coords.name]).max()


# %% Solve the stochastic problem with default KVL and with the workaround
models = {"default": stochastic(False), "workaround": stochastic(True)}
lines = models["workaround"].c.lines.static
bus_carrier = models["workaround"].c.buses.static.carrier.droplevel("scenario")
ac = lines.index[lines.bus0.map(bus_carrier.groupby(level=0).first()) == "AC"]
ac = ac.set_names("line", level="name")

# %% 1) KVL constraints in the linopy model: one per scenario with the workaround
cycles = {
    (k, name): n.model.constraints[name].coeffs.sizes["cycle"]
    for k, n in models.items()
    for name in n.model.constraints
    if name.startswith("Kirchhoff")
}
cycles = pd.Series(cycles).rename_axis(["model", "constraint"]).to_frame("cycles")
cycles

# %% 1) AC lines: x_pu_eff in the data vs KVL weight in the model (NaN: in no cycle)
weights = {f"KVL {k}": kvl_weights(n) for k, n in models.items()}
weights = pd.DataFrame({"x_pu_eff": lines.x_pu_eff, **weights}) * 1e6
s_nom = lines.s_nom.astype(int).rename("s_nom [MW]")
table = pd.concat([s_nom, weights.add_suffix(" [1e-6 p.u.]")], axis=1).reindex(ac)
table.round(2)

# %% 2) Dispatch vs each branch solved alone on its true grid
rows = {}
for k, n in models.items():
    opex = n.statistics.opex().groupby(level="scenario").sum()
    for s in n.scenarios:
        d = deterministic(n, s)
        rows[(k, s)] = 100 * (opex[s] / d.statistics.opex().sum() - 1)
dispatch = pd.Series(rows).rename_axis(["model", "scenario"])
dispatch.to_frame("opex error [%]").round(2)

# %% 3) AC lines: mean |flow| [MW] (default: 0 MW where unbuilt lines angle-lock)
flows = {k: n.c.lines.dynamic.p0.abs().mean() for k, n in models.items()}
flows = pd.concat(flows, axis=1, names=["model"]).reindex(ac).unstack("scenario")
flows.round().astype(int)
