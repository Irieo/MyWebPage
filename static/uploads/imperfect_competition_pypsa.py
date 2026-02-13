"""
Imperfect Competition with PyPSA
========================================================
IR, Feb 2026

This example demonstrates how oligopolistic behavior in a gas supply market
increases the cost of electricity. We embed a Cournot game (from prototype.py)
between two gas producers into a PyPSA power system model.

Network topology:
-----------------
    [gas_producer_1] ──┐                    ┌── [solar]
                       ├─ (gas bus) ── Link ─┤
    [gas_producer_2] ──┘   "gas_plant"       ├── [elastic_demand]
                                             └── (electricity bus)

- "electricity" bus: power market with elastic demand and solar generation
- "gas" bus: fuel supply with two strategic gas producers
- Link "gas_plant": gas-fired backup power plant (fixed capacity, 1:1 efficiency)

Parametrisation (from prototype.py):
------------------------------------
- Gas producer cost:  C(q) = q + 0.5·q²   →  MC(q) = 1 + q
- Inverse demand:     P(d) = 30 − 4·d     →  slope b = 4

Scenarios:
----------
A) Perfect competition (cv=0): standard PyPSA optimization = social welfare max.
   Gas producers bid at marginal cost. First Welfare Theorem applies.
B) Cournot-Nash (cv=1): gas producers exercise market power. Each accounts for
   its impact on the gas price. Implemented by adding a Cournot markup term
   (b·cv/2)·q² to each producer's objective in the linopy model.

Key result: electricity costs increase under Cournot. The increase decomposes
into (i) oligopolist revenue gain (transfer) and (ii) deadweight welfare loss.
"""

import pypsa
import pandas as pd
from linopy.expressions import LinearExpression, QuadraticExpression

# ============================================================================
# 1. Network parametrisation
# ============================================================================

# Game parameters (same as prototype.py)
DEMAND_INTERCEPT = 30  # a in P(d) = a − b·d  [EUR/MWh]
DEMAND_SLOPE = 4  # b in P(d) = a − b·d  [EUR/MWh per MW]
LINEAR_COST = 1  # linear cost coefficient  [EUR/MWh]
QUADRATIC_COST = 0.5  # quadratic cost coefficient  [EUR/MWh per MW]

# Power system parameters
SOLAR_CAPACITY = 3.0  # [MW] installed solar capacity
SOLAR_CF = [0.8, 0.5, 0.1]  # capacity factors per snapshot (decreasing sun)
GAS_PLANT_CAPACITY = 8.0  # [MW] gas backup plant capacity
GAS_PRODUCER_CAPACITY = 5.0  # [MW] each gas producer's capacity
DEMAND_MAX = 10.0  # [MW] maximum demand quantity

# %%


def create_network():
    """Create the two-bus electricity + gas network.

    The network has 3 snapshots with decreasing solar output,
    creating increasing residual demand for gas backup.
    """
    n = pypsa.Network()
    n.set_snapshots(range(3))

    # --- Buses ---
    n.add("Bus", "electricity")
    n.add("Bus", "gas")

    # --- Electricity market ---

    # Solar: zero-cost renewable with time-varying capacity factor
    n.add(
        "Generator",
        "solar",
        bus="electricity",
        p_nom=SOLAR_CAPACITY,
        marginal_cost=0,
        p_max_pu=SOLAR_CF,
    )

    # Elastic demand: inverse demand curve P(d) = 30 − 4·d
    #
    # Modelled as a sign=−1 generator ("demand bid"):
    #   Objective contribution: (−30)·d + 2·d²
    #   Minimising → FOC: −30 + 4·d = −λ  (λ = shadow price on bus)
    #   Hence: λ = 30 − 4·d = P(d)  ✓
    n.add(
        "Generator",
        "elastic_demand",
        bus="electricity",
        sign=-1,
        p_nom=DEMAND_MAX,
        marginal_cost=-DEMAND_INTERCEPT,
        marginal_cost_quadratic=DEMAND_SLOPE / 2,
    )

    # --- Gas supply ---

    # Two gas producers with identical convex cost C(q) = q + 0.5·q²
    for i in [1, 2]:
        n.add(
            "Generator",
            f"gas_producer_{i}",
            bus="gas",
            p_nom=GAS_PRODUCER_CAPACITY,
            marginal_cost=LINEAR_COST,
            marginal_cost_quadratic=QUADRATIC_COST,
        )

    # --- Gas-to-electricity link (gas backup plant) ---
    # Converts gas to electricity 1:1, at fixed capacity, no extra cost.
    # The gas bus shadow price becomes the fuel cost of gas-fired power.
    n.add(
        "Link",
        "gas_plant",
        bus0="gas",
        bus1="electricity",
        p_nom=GAS_PLANT_CAPACITY,
        marginal_cost=0,
        efficiency=1.0,
    )

    return n


# %%
# Inspect the network
n = create_network()

print("=== Generators ===")
print(
    n.generators[["bus", "sign", "p_nom", "marginal_cost", "marginal_cost_quadratic"]]
)
print("\n=== Links ===")
print(n.links[["bus0", "bus1", "p_nom", "efficiency"]])
print("\n=== Solar capacity factors ===")
print(n.generators_t.p_max_pu["solar"])


# ============================================================================
# 2. Problem A — Perfect competition (cv = 0)
# ============================================================================
#
# Standard PyPSA optimization = social welfare maximisation.
# Under convexity + price-taking, cost minimisation yields the competitive
# equilibrium (First Welfare Theorem).

# %%
n_a = create_network()
m_a = n_a.optimize.create_model()

# Problem A is the unmodified base model
status_a, cond_a = n_a.optimize.solve_model(solver_name="highs")

print(f"\n{'=' * 60}")
print("  Problem A: Perfect Competition (cv = 0)")
print(f"{'=' * 60}")
print(f"  Status: {status_a} | {cond_a}")
print(f"  Objective (= −welfare): {n_a.objective:.4f}")


# ============================================================================
# 3. Problem B — Cournot-Nash competition (cv = 1)
# ============================================================================
#
# Each gas producer i internalises its impact on the market price.
#
# Recall the Cournot FOC from prototype.py:
#   −P + b·cv·q_i + MC(q_i) = 0
#   → effective MC = MC(q_i) + b·cv·q_i = (1 + q_i) + 4·q_i = 1 + 5·q_i
#
# To obtain this FOC from a minimisation problem, we add a markup term
# to each producer's objective:
#
#   markup_i(q) = (b · cv / 2) · q_i²
#
# so that d/dq_i [C(q_i) + markup_i(q_i)] = MC(q_i) + b·cv·q_i   ✓
#
# Derived gas demand slope:
# -------------------------
# At each snapshot, gas demand g = d − solar.
# Inverse demand for gas:  P_gas(g) = [a − b·solar_t] − b·g
# The slope is still −b = −4, same as the electricity demand curve,
# because the Link has 1:1 efficiency and zero marginal cost.

# %%
n_b = create_network()
m_b = n_b.optimize.create_model()

# Access gas producer dispatch variables from the linopy model
# Dimensions: (snapshot,) after selecting by name
q1 = m_b["Generator-p"].sel(name="gas_producer_1")
q2 = m_b["Generator-p"].sel(name="gas_producer_2")

# Cournot game parameters
cv = 1  # conjectural variation: 1 = Cournot-Nash
b = DEMAND_SLOPE  # inverse demand slope = 4

# Cournot markup: (b·cv/2) · q_i²  for each producer, each snapshot
# This effectively raises each producer's marginal cost from (1+q) to (1+5q)
cournot_markup = (b * cv / 2) * (q1 * q1 + q2 * q2)

# Append markup to the PyPSA model's objective
current_obj = m_b.objective
if not isinstance(current_obj, (LinearExpression, QuadraticExpression)):
    current_obj = current_obj.expression
m_b.objective = current_obj + cournot_markup.sum()

# Solve
status_b, cond_b = n_b.optimize.solve_model(solver_name="highs")

print(f"\n{'=' * 60}")
print("  Problem B: Cournot-Nash Competition (cv = 1)")
print(f"{'=' * 60}")
print(f"  Status: {status_b} | {cond_b}")


# ============================================================================
# 4. Results comparison
# ============================================================================

# %%


def extract_results(n, label):
    """Extract key results from a solved network."""
    q1 = n.generators_t.p["gas_producer_1"].values
    q2 = n.generators_t.p["gas_producer_2"].values
    solar = n.generators_t.p["solar"].values
    demand = n.generators_t.p["elastic_demand"].values
    p_elec = n.buses_t.marginal_price["electricity"].values
    p_gas = n.buses_t.marginal_price["gas"].values
    gas_flow = n.links_t.p0["gas_plant"].values

    return pd.DataFrame(
        {
            "solar": solar,
            f"gas1": q1,
            f"gas2": q2,
            f"gas_total": q1 + q2,
            f"gas_plant": gas_flow,
            f"demand": demand,
            f"P_elec": p_elec,
            f"P_gas": p_gas,
        },
        index=[f"t={t}" for t in range(len(q1))],
    )


res_a = extract_results(n_a, "A")
res_b = extract_results(n_b, "B")

print(f"\n{'=' * 70}")
print("  DISPATCH & PRICES: Problem A (Competitive)")
print(f"{'=' * 70}")
print(res_a.to_string(float_format="{:.4f}".format))

print(f"\n{'=' * 70}")
print("  DISPATCH & PRICES: Problem B (Cournot)")
print(f"{'=' * 70}")
print(res_b.to_string(float_format="{:.4f}".format))


# ============================================================================
# 5. Economic analysis: oligopolist revenue gain = electricity cost increase
# ============================================================================

# %%


def production_cost(q):
    """True production cost: C(q) = q + 0.5·q² (same for both producers)."""
    return LINEAR_COST * q + QUADRATIC_COST * q**2


def analyse_economics(n, label):
    """Compute economic metrics for a solved network."""
    q1 = n.generators_t.p["gas_producer_1"].values
    q2 = n.generators_t.p["gas_producer_2"].values
    p_gas = n.buses_t.marginal_price["gas"].values
    p_elec = n.buses_t.marginal_price["electricity"].values
    demand = n.generators_t.p["elastic_demand"].values

    # Production costs
    cost1 = production_cost(q1)
    cost2 = production_cost(q2)

    # Revenue = price × quantity
    rev1 = p_gas * q1
    rev2 = p_gas * q2

    # Profit = revenue − cost
    profit1 = rev1 - cost1
    profit2 = rev2 - cost2

    # Consumer expenditure (what electricity consumers pay)
    consumer_exp = p_elec * demand

    # Consumer surplus: integral under demand curve minus expenditure
    # ∫₀ᵈ P(x)dx = a·d − (b/2)·d²
    consumer_surplus = (
        DEMAND_INTERCEPT * demand - (DEMAND_SLOPE / 2) * demand**2 - consumer_exp
    )

    return {
        "label": label,
        "q1": q1,
        "q2": q2,
        "demand": demand,
        "p_gas": p_gas,
        "p_elec": p_elec,
        "cost1": cost1,
        "cost2": cost2,
        "profit1": profit1,
        "profit2": profit2,
        "total_producer_profit": (profit1 + profit2).sum(),
        "total_production_cost": (cost1 + cost2).sum(),
        "total_consumer_expenditure": consumer_exp.sum(),
        "total_consumer_surplus": consumer_surplus.sum(),
    }


econ_a = analyse_economics(n_a, "A: Competitive")
econ_b = analyse_economics(n_b, "B: Cournot")

print(f"\n{'=' * 70}")
print("  ECONOMIC ANALYSIS")
print(f"{'=' * 70}")

for econ in [econ_a, econ_b]:
    print(f"\n  --- {econ['label']} ---")
    for t in range(3):
        print(
            f"  t={t}: P_gas={econ['p_gas'][t]:.2f}, "
            f"q1={econ['q1'][t]:.2f}, q2={econ['q2'][t]:.2f}, "
            f"π1={econ['profit1'][t]:.2f}, π2={econ['profit2'][t]:.2f}, "
            f"demand={econ['demand'][t]:.2f}, P_elec={econ['p_elec'][t]:.2f}"
        )
    print(f"  TOTALS:")
    print(f"    Total gas production cost:    {econ['total_production_cost']:.4f}")
    print(f"    Total producer profit:        {econ['total_producer_profit']:.4f}")
    print(f"    Total consumer expenditure:   {econ['total_consumer_expenditure']:.4f}")
    print(f"    Total consumer surplus:       {econ['total_consumer_surplus']:.4f}")

# %%

# --- Key result: decomposition of welfare loss ---

print(f"\n\n{'=' * 70}")
print("  KEY RESULT: Market Power Impact Decomposition")
print(f"{'=' * 70}")

# Welfare = Consumer surplus + Producer surplus (profit)
welfare_a = econ_a["total_consumer_surplus"] + econ_a["total_producer_profit"]
welfare_b = econ_b["total_consumer_surplus"] + econ_b["total_producer_profit"]

welfare_loss = welfare_a - welfare_b  # positive = welfare decreased
revenue_increase = econ_b["total_producer_profit"] - econ_a["total_producer_profit"]
consumer_loss = econ_a["total_consumer_surplus"] - econ_b["total_consumer_surplus"]
deadweight_loss = welfare_loss

# Cost increase to electricity system = consumer expenditure increase
cost_increase = (
    econ_b["total_consumer_expenditure"] - econ_a["total_consumer_expenditure"]
)

print(f"\n  Competitive total welfare:      {welfare_a:.4f}")
print(f"  Cournot total welfare:          {welfare_b:.4f}")
print(f"  Welfare loss (DWL):             {deadweight_loss:.4f}")

print(f"\n  Producer profit (competitive):  {econ_a['total_producer_profit']:.4f}")
print(f"  Producer profit (Cournot):      {econ_b['total_producer_profit']:.4f}")
print(f"  Oligopolist revenue increase:   {revenue_increase:.4f}")

print(f"\n  Consumer surplus (competitive): {econ_a['total_consumer_surplus']:.4f}")
print(f"  Consumer surplus (Cournot):     {econ_b['total_consumer_surplus']:.4f}")
print(f"  Consumer surplus loss:          {consumer_loss:.4f}")

print(f"\n  Consumer expenditure increase:  {cost_increase:.4f}")

print(f"\n  Decomposition of consumer surplus loss:")
print(f"    = Revenue transfer to producers: {revenue_increase:.4f}")
print(f"    + Deadweight loss:               {deadweight_loss:.4f}")
print(f"    = Total consumer loss:           {revenue_increase + deadweight_loss:.4f}")
print(f"    (actual consumer loss:           {consumer_loss:.4f})")

# %%

# --- Side-by-side summary table ---

print(f"\n\n{'=' * 70}")
print("  SUMMARY TABLE")
print(f"{'=' * 70}")

summary = pd.DataFrame(
    {
        "A: Competitive": {
            "Gas price (avg)": econ_a["p_gas"].mean(),
            "Elec price (avg)": econ_a["p_elec"].mean(),
            "Gas output (total)": (econ_a["q1"] + econ_a["q2"]).sum(),
            "Demand (total)": econ_a["demand"].sum(),
            "Production cost": econ_a["total_production_cost"],
            "Consumer expenditure": econ_a["total_consumer_expenditure"],
            "Producer profit": econ_a["total_producer_profit"],
            "Consumer surplus": econ_a["total_consumer_surplus"],
            "Total welfare": welfare_a,
        },
        "B: Cournot": {
            "Gas price (avg)": econ_b["p_gas"].mean(),
            "Elec price (avg)": econ_b["p_elec"].mean(),
            "Gas output (total)": (econ_b["q1"] + econ_b["q2"]).sum(),
            "Demand (total)": econ_b["demand"].sum(),
            "Production cost": econ_b["total_production_cost"],
            "Consumer expenditure": econ_b["total_consumer_expenditure"],
            "Producer profit": econ_b["total_producer_profit"],
            "Consumer surplus": econ_b["total_consumer_surplus"],
            "Total welfare": welfare_b,
        },
    }
)
summary["Δ (B−A)"] = summary["B: Cournot"] - summary["A: Competitive"]
print(summary.to_string(float_format="{:.4f}".format))

print(
    "\n  → Under Cournot competition, gas producers restrict output to raise"
    "\n    the gas price. This increases electricity costs and transfers"
    "\n    surplus from consumers to producers, with an additional deadweight"
    "\n    welfare loss from reduced consumption."
)
