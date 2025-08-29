# -*- coding: utf-8 -*-
"""
Three model implementations:
    1. Deterministic optimization over gas-price scenarios
    2. Stochastic optimization via PyPSA API
    3. Two-stage SDDP with multi-cut Benders decomposition
"""

# Step 1: Imports and Configuration
import matplotlib.pyplot as plt
import pandas as pd
from xarray import DataArray
import pypsa
import numpy as np
from linopy import Model
from pypsa.components.common import as_components
from linopy.expressions import merge


# Plot helpers
def plot_capacity(
    df,
    title="Capacity Mix",
    xlabel="Scenario",
    ylabel="Capacity (MW)",
    figsize=(6, 3),
    color_map=None,
    rotation=0,
):
    """
    Plot capacity mix as stacked bar chart
    """
    if color_map is None:
        color_map = COLOR_MAP

    colors = [color_map.get(c, "gray") for c in df.columns]
    ax = df.plot(kind="bar", stacked=True, figsize=figsize, color=colors)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation > 0 else "center")
    plt.tight_layout()
    return ax


def plot_cost(
    series,
    title="Total Cost",
    xlabel="Scenario",
    ylabel="EUR",
    figsize=(4, 3),
    rotation=0,
):
    """
    Plot costs as bar chart
    """
    ax = series.plot(kind="bar", figsize=figsize)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation > 0 else "center")
    plt.tight_layout()
    return ax


# Scenario definitions
SCENARIOS = ["low", "med", "high"]
GAS_PRICES = {"low": 40, "med": 70, "high": 100}
PROB = {"low": 0.4, "med": 0.3, "high": 0.3}
BASE = "low"
FREQ = "3h"
LOAD_MW = 1
SOLVER = "highs"
TS_URL = (
    "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
)


# Technology specs
def annuity(life, rate):
    return rate / (1 - (1 + rate) ** -life) if rate else 1 / life


TECH = {
    "solar": {"profile": "solar", "inv": 1e6, "m_cost": 0.01},
    "wind": {"profile": "onwind", "inv": 2e6, "m_cost": 0.02},
    "gas": {"inv": 7e5, "eff": 0.6},
    "lignite": {"inv": 1.3e6, "eff": 0.4, "m_cost": 130},
}
FOM, DR, LIFE = 3.0, 0.03, 25
for cfg in TECH.values():
    cfg["fixed_cost"] = (annuity(LIFE, DR) + FOM / 100) * cfg["inv"]

COLOR_MAP = {
    "solar": "yellow",
    "wind": "skyblue",
    "gas": "brown",
    "lignite": "black",
}

# Step 2: Load Time Series
ts = pd.read_csv(TS_URL, index_col=0, parse_dates=True)
ts = ts.resample(FREQ).asfreq()


###########################################
# Step 3: Build toy network & run deterministic scenarios
def build_network(gas_price):
    """
    Create PyPSA network:
      - DE bus and constant load
      - extendable generators for solar, wind, gas, lignite
    """
    n = pypsa.Network()
    n.set_snapshots(ts.index)
    n.snapshot_weightings = pd.Series(int(FREQ[:-1]), index=ts.index)
    n.add("Bus", "DE")
    n.add("Load", "DE_load", bus="DE", p_set=LOAD_MW)

    for tech in ["solar", "wind"]:
        cfg = TECH[tech]
        n.add(
            "Generator",
            tech,
            bus="DE",
            p_nom_extendable=True,
            p_max_pu=ts[cfg["profile"]],
            capital_cost=cfg["fixed_cost"],
            marginal_cost=cfg["m_cost"],
        )
    for tech in ["gas", "lignite"]:
        cfg = TECH[tech]
        mc = (gas_price / cfg.get("eff")) if tech == "gas" else cfg["m_cost"]
        n.add(
            "Generator",
            tech,
            bus="DE",
            p_nom_extendable=True,
            efficiency=cfg.get("eff"),
            capital_cost=cfg["fixed_cost"],
            marginal_cost=mc,
        )
    return n


caps_det = pd.DataFrame(index=SCENARIOS, columns=TECH.keys())
objs_det = pd.Series(index=SCENARIOS)

for sc in SCENARIOS:
    n = build_network(GAS_PRICES[sc])
    n.optimize(solver_name=SOLVER)
    caps_det.loc[sc] = n.generators.p_nom_opt
    objs_det.loc[sc] = n.objective


"""
Linopy LP model
===============

Variables:
----------
 * Generator-p_nom (Generator-ext)
 * Generator-p (snapshot, Generator)

Constraints:
------------
 * Generator-ext-p_nom-lower (Generator-ext)
 * Generator-ext-p_nom-upper (Generator-ext)
 * Generator-ext-p-lower (snapshot, Generator-ext)
 * Generator-ext-p-upper (snapshot, Generator-ext)
 * Bus-nodal_balance (Bus, snapshot)

"""

# Step 4: Plot Deterministic Results
plot_capacity(caps_det, title="Deterministic Capacity Mix")
plt.show()

plot_cost(objs_det, title="Deterministic Total Cost")
plt.show()


###########################################
# Step 5: Build stochastic network with PyPSA API

n_stoch = build_network(GAS_PRICES[BASE])

# Adding scenarios & probabilities to the network are super easy
n_stoch.set_scenarios(PROB)

"""
Unnamed Stochastic PyPSA Network
--------------------------------
Components:
 - Bus: 3
 - Generator: 12
 - Load: 3
Snapshots: 2920
Scenarios: 3
"""

# n_stoch
# n_stoch.scenarios
# n_stoch.generators
# n_stoch.generators_t.p_max_pu
# n_stoch.generators.loc[:, "marginal_cost"]
# n_stoch.optimize.create_model()
# n_stoch.model.constraints['Generator-ext-p-lower']

# Set data that is varying by scenario
for sc in SCENARIOS:
    if sc != BASE:
        idx = (sc, "gas")
        n_stoch.generators.loc[idx, "marginal_cost"] = (
            GAS_PRICES[sc] / n_stoch.generators.loc[idx, "efficiency"]
        )

n_stoch.optimize(solver_name=SOLVER)
caps_api = n_stoch.generators.p_nom_opt.xs("med", level="scenario")
obj_api = n_stoch.objective


"""
Linopy LP model
===============

Variables:
----------
 * Generator-p_nom (component)
 * Generator-p (scenario, component, snapshot)

Constraints:
------------
 * Generator-ext-p_nom-lower (component, scenario)
 * Generator-ext-p-lower (scenario, component, snapshot)
 * Generator-ext-p-upper (scenario, component, snapshot)
 * Bus-nodal_balance (component, scenario, snapshot)

"""

# Step 6: Plot API Stochastic Results
caps_all = caps_det.copy()
caps_all.loc["Stochastic API"] = caps_api
plot_capacity(caps_all, title="Capacity Mix: Deterministic + Stochastic")
plt.show()

objs_all = objs_det.copy()
objs_all.loc["Stochastic API"] = obj_api
plot_cost(objs_all, title="Total Cost: Deterministic + Stochastic")
plt.show()


##########################################
# Step 7: Two-stage SDDP (Benders with multiple cuts per iteration)


def run_sddp_benders(max_iters=40, tol=1e-4, solver_name=SOLVER):
    """Two-stage risk-neutral SDDP with multi-cut Benders."""
    gens = list(TECH.keys())
    sns = ts.index

    # Capital costs and availability
    cap_cost = pd.Series({g: TECH[g]["fixed_cost"] for g in gens})
    cap_cost_da = DataArray(cap_cost.values, coords={"gen": gens}, dims=["gen"])
    M_cols = {
        g: (
            ts[TECH[g]["profile"]].values.astype(float)
            if "profile" in TECH[g]
            else np.ones(len(sns))
        )
        for g in gens
    }
    M = DataArray(
        np.column_stack([M_cols[g] for g in gens]),
        coords={"snapshot": sns, "gen": gens},
        dims=["snapshot", "gen"],
    )

    load = DataArray(
        np.full(len(sns), LOAD_MW, dtype=float),
        coords={"snapshot": sns},
        dims=["snapshot"],
    )
    w = DataArray(
        np.full(len(sns), int(FREQ[:-1]), dtype=float),
        coords={"snapshot": sns},
        dims=["snapshot"],
    )

    # Scenario marginal costs
    costs = {}
    for sc in SCENARIOS:
        cvals = [
            GAS_PRICES[sc] / TECH["gas"]["eff"]
            if g == "gas"
            else TECH[g].get("m_cost", 0.0)
            for g in gens
        ]
        costs[sc] = DataArray(
            np.array(cvals, dtype=float), coords={"gen": gens}, dims=["gen"]
        )

    # Master (cap, theta[scenario])
    master = Model()
    cap_upper = pd.Series(index=gens, dtype=float)

    # somewhat random upper caps not to overthink it now
    for g in gens:
        cap_upper[g] = LOAD_MW * (5.0 if "profile" in TECH[g] else 2.0)
    cap_upper_da = DataArray(cap_upper.values, coords={"gen": gens}, dims=["gen"])
    cap = master.add_variables(
        coords=cap_cost_da.coords,
        dims=cap_cost_da.dims,
        name="cap",
        lower=0.0,
        upper=cap_upper_da,
    )
    theta = master.add_variables(
        coords={"scenario": SCENARIOS}, dims=["scenario"], name="theta", lower=0.0
    )
    prob_da = DataArray(
        [PROB[s] for s in SCENARIOS], coords={"scenario": SCENARIOS}, dims=["scenario"]
    )

    # master is “minimize expected cost by choosing capacities + scenario cost proxies, but constrained by all the supporting hyperplanes we’ve learned so far.”
    master.objective = (cap * cap_cost_da).sum() + (theta * prob_da).sum()

    best_ub = np.inf
    cap_best = None
    qbar_last = None  # expected second stage cost
    shed_cost = 1e6  # need this for the subproblem

    for it in range(1, max_iters + 1):
        master.solve(solver_name=solver_name)
        cap_now = pd.Series(master["cap"].solution.to_pandas(), index=gens)

        expected_Q = 0.0
        for sc in SCENARIOS:
            # Subproblem: given that capacity from Master problem, my real cost is Q_s
            sub = Model()
            p = sub.add_variables(coords=M.coords, dims=M.dims, name="p", lower=0.0)
            s = sub.add_variables(
                coords=load.coords, dims=load.dims, name="shed", lower=0.0
            )
            # dispatch constraint
            sub.add_constraints(
                p <= M * DataArray(cap_now.values, coords={"gen": gens}, dims=["gen"]),
                name="ub",
            )
            # nodal balance constraint
            sub.add_constraints(p.sum(dim="gen") + s == load, name="balance")

            # objective
            c_sc = costs[sc]
            sub.objective = (p * c_sc * w).sum() + (s * shed_cost * w).sum()
            sub.solve(solver_name=solver_name)

            # Get second stage cost
            Q_s = float(sub.objective.value)

            # Time for a true magic

            # <g> is a slope of how much the recourse cost decreases when we increase capacity
            # alpha_s is base cost if we had no capacity (shed everything)
            u = sub.constraints["ub"].dual
            pi = sub.constraints["balance"].dual
            g_s = (u * M).sum(dim="snapshot").to_pandas()
            alpha_s = float((pi * load).sum().item())

            expected_Q += PROB[sc] * Q_s
            g_da = DataArray(g_s.values, coords={"gen": gens}, dims=["gen"])

            # This cut is essentially a supporting hyperplane to the true cost function Q_s(x)
            # Like Benders
            lhs_sc = theta.loc[{"scenario": sc}] - (g_da * cap).sum()
            master.add_constraints(lhs_sc >= alpha_s, name=f"cut_{sc}_{it}")

        lb = float(master.objective.value)
        ub = float((cap_now * cap_cost).sum() + expected_Q)
        qbar_last = expected_Q
        if ub < best_ub:
            best_ub, cap_best = ub, cap_now.copy()

        gap = (best_ub - lb) / max(1.0, abs(best_ub))
        if gap <= tol:
            print(f"SDDP converged after {it} iterations (gap: {gap:.2e})")
            break
    else:
        print(
            f"SDDP reached max iterations ({max_iters}) without convergence (gap: {gap:.2e})"
        )

    cap_opt = cap_best if cap_best is not None else cap_now
    obj_opt = float(
        (cap_opt * cap_cost).sum() + (qbar_last if qbar_last is not None else 0.0)
    )
    return cap_opt, obj_opt


caps_sddp, obj_sddp = run_sddp_benders()

# Compare results numerically
print("\n=== COMPARISON of PyPSA stochastic problem vs SDDP toy implementation ===")
print("API Stochastic Capacities (MW):")
print(caps_api)
print("API Stochastic Objective:", obj_api)

print("\nSDDP Capacities (MW):")
print(caps_sddp)
print("SDDP Objective:", obj_sddp)

print(f"\nObjective Difference: {abs(obj_sddp - obj_api):.2f} EUR")
print(f"Objective Relative Error: {abs(obj_sddp - obj_api) / obj_api * 100:.2f}%")

print("\nCapacity Differences (MW):")
for gen in caps_api.index:
    diff = caps_sddp[gen] - caps_api[gen]
    print(f"{gen}: {diff:.1f} MW ({diff / max(caps_api[gen], 0.1) * 100:.1f}%)")

caps_sddp_compare = caps_det.copy()
caps_sddp_compare.loc["Stochastic API"] = caps_api
caps_sddp_compare.loc["Stochastic SDDP"] = caps_sddp

plot_capacity(
    caps_sddp_compare,
    title="Capacity Mix: Deterministic + Stochastic (API/SDDP)",
    xlabel="Scenario / Optimization Type",
    figsize=(10, 5),
    rotation=45,
)
plt.show()

objs_sddp_compare = objs_det.copy()
objs_sddp_compare.loc["Stochastic API"] = obj_api
objs_sddp_compare.loc["Stochastic SDDP"] = obj_sddp

plot_cost(
    objs_sddp_compare,
    title="Objective Value: Deterministic + Stochastic (API/SDDP)",
    xlabel="Scenario / Optimization Type",
    ylabel="Total Cost (EUR)",
    figsize=(8, 5),
    rotation=45,
)
plt.show()
