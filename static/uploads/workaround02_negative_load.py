import pypsa
import numpy as np
import matplotlib.pyplot as plt

"""
Two-bus UA with demand-side cap at 100 €/MWh.

Structure:
- UA_supply: holds UA generators and connects to PL via free interconnector.
- UA_demand: holds the UA load and a DR "load shedding" generator at 100 €/MWh.
- One-way internal link UA_supply -> UA_demand (no reverse flow), so DR cannot export.

Behavior:
- When PL price > 100, UA_demand prefers DR at 100 over importing via UA_supply.
- UA_supply can still export to PL if profitable (e.g., PL price very high).
"""

PRICE_CAP = 100
VOLL = 300

n = pypsa.Network()
n.set_snapshots(range(24))

# Buses
n.add("Bus", "UA_supply")
n.add("Bus", "UA_demand")
n.add("Bus", "PL")

# Loads
ua_load_profile = np.tile([20, 40, 60, 80, 100, 120], 4)
pl_load_profile = np.tile([20, 40, 60, 80, 100, 120], 4)
ua_load_profile[10] = 200

n.add("Load", "UA_load", bus="UA_demand", p_set=ua_load_profile)
n.add("Load", "PL_load", bus="PL", p_set=pl_load_profile)

# Generators
n.add("Generator", "PL_base", bus="PL", p_nom=40, marginal_cost=30)
n.add("Generator", "PL_peak", bus="PL", p_nom=40, marginal_cost=110)

n.add("Generator", "UA_base", bus="UA_supply", p_nom=80, marginal_cost=35)
n.add("Generator", "UA_peak", bus="UA_supply", p_nom=40, marginal_cost=150)
n.add(
    "Link",
    "PL_UA",
    bus0="PL",
    bus1="UA_supply",
    p_nom=60,
    efficiency=1.0,
    marginal_cost=0.0,
    p_min_pu=-1.0,  # allow both import (>0 p0) and export (<0 p0)
)

# Internal UA link: one-way UA_supply -> UA_demand
n.add(
    "Link",
    "UA_supply_to_UA_demand",
    bus0="UA_supply",
    bus1="UA_demand",
    p_nom=1e3,
    efficiency=1.0,
    marginal_cost=0.0,
    p_min_pu=0.0,  # only forward flow, prevents UA_demand from exporting
)

# Demand response (cap) and last-resort shedding at UA_demand
n.add("Generator", "UA_DR_cap", bus="UA_demand", p_nom=1e3, marginal_cost=PRICE_CAP)
n.add("Generator", "UA_shedding", bus="UA_demand", p_nom=1e3, marginal_cost=VOLL)

# Optimize
n.optimize(solver_name="highs")

# Results & checks
pl_prices = n.buses_t.marginal_price["PL"]
ua_supply_prices = n.buses_t.marginal_price["UA_supply"]
ua_demand_prices = n.buses_t.marginal_price["UA_demand"]

imports_from_pl = n.links_t.p0[
    "PL_UA"
]  # >0 means PL -> UA_supply; <0 means UA_supply -> PL
ua_internal_flow = n.links_t.p0["UA_supply_to_UA_demand"]

ua_base_gen = n.generators_t.p["UA_base"]
ua_peak_gen = n.generators_t.p["UA_peak"]
ua_dr_cap = n.generators_t.p["UA_DR_cap"]
ua_shed = n.generators_t.p["UA_shedding"]
ua_load = n.loads_t.p["UA_load"]

# Verify condition: when PL price > 100, PL->UA imports should be ~0
over_cap_hours = pl_prices > PRICE_CAP + 1e-6
# Count only PL→UA imports (p0>0) during over-cap hours
imports_when_pl_over_cap = imports_from_pl.clip(lower=0)[over_cap_hours].sum()

print("\n" + "=" * 60)
print("TWO-BUS UA WITH DR CAP RESULTS")
print("=" * 60)
print(
    f"PL price min/avg/max: {pl_prices.min():.1f} / {pl_prices.mean():.1f} / {pl_prices.max():.1f} €/MWh"
)
print(
    f"UA_supply price min/avg/max: {ua_supply_prices.min():.1f} / {ua_supply_prices.mean():.1f} / {ua_supply_prices.max():.1f} €/MWh"
)
print(
    f"UA_demand price min/avg/max: {ua_demand_prices.min():.1f} / {ua_demand_prices.mean():.1f} / {ua_demand_prices.max():.1f} €/MWh"
)
print(f"Total PL->UA imports (p0>0): {imports_from_pl.clip(lower=0).sum():.1f} MWh")
print(f"Total UA->PL exports (p0<0): {-imports_from_pl.clip(upper=0).sum():.1f} MWh")
print(f"Flow UA_supply->UA_demand: {ua_internal_flow.sum():.1f} MWh")
print(
    f"UA DR at 100 €/MWh: {ua_dr_cap.sum():.1f} MWh, Shedding at 300 €/MWh: {ua_shed.sum():.1f} MWh"
)
print(
    f"Imports when PL price > {PRICE_CAP}: {imports_when_pl_over_cap:.1f} MWh (should be ~0)"
)

# Plots
hours = range(24)
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# 1) UA_demand stack
ax1 = axes[0]
stack1 = ua_internal_flow
stack2 = stack1 + ua_dr_cap
stack3 = stack2 + ua_shed
ax1.fill_between(hours, 0, stack1, label="Supply from UA_supply", alpha=0.7)
ax1.fill_between(
    hours, stack1, stack2, label="DR at 100 €/MWh", alpha=0.7, color="#ffaa00"
)
ax1.fill_between(
    hours, stack2, stack3, label="Shedding at 300 €/MWh", alpha=0.7, color="red"
)
ax1.plot(hours, ua_load, "k--", lw=2, label="UA Load (UA_demand)")
ax1.set_title("UA_demand Supply Stack (from UA_supply + DR)")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(alpha=0.3)

# 2) Cross-border link direction
ax2 = axes[1]
ax2.plot(hours, imports_from_pl, label="PL_UA (p0>0 import, <0 export)")
ax2.axhline(0, color="k", lw=1)
ax2.set_ylabel("MW")
ax2.set_title("PL ↔ UA_supply Interconnector Flow")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

# 3) Prices
ax3 = axes[2]
ax3.plot(hours, pl_prices, label="PL price", marker="s")
ax3.plot(hours, ua_supply_prices, label="UA_supply price", marker="o")
ax3.plot(hours, ua_demand_prices, label="UA_demand price", marker="^")
ax3.axhline(PRICE_CAP, color="r", ls="--", label=f"DR cap = {PRICE_CAP}")
ax3.set_ylabel("€/MWh")
ax3.set_title("Nodal Prices")
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(alpha=0.3)

# 4) PL supply stack (generation and exports)
ax4 = axes[3]
pl_base_gen = n.generators_t.p["PL_base"]
pl_peak_gen = n.generators_t.p["PL_peak"]
pl_load = n.loads_t.p["PL_load"]
pl_exports = imports_from_pl  # >0 means PL is exporting to UA

ax4.fill_between(hours, 0, pl_base_gen, label="PL Base Gen", alpha=0.7)
ax4.fill_between(
    hours,
    pl_base_gen,
    pl_base_gen + pl_peak_gen,
    label="PL Peak Gen",
    alpha=0.7,
)
ax4.plot(hours, pl_load, "k--", lw=2, label="PL Load")
ax4.plot(hours, pl_exports, "r:", lw=2, label="Exports to UA (+)")
ax4.set_ylabel("MW")
ax4.set_xlabel("Hour")
ax4.set_title("PL Supply Stack and Exports")
ax4.legend(loc="upper left", fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("price_cap_negative_load.png", dpi=150, bbox_inches="tight")
print("Plot saved as price_cap_negative_load.png")
plt.show()
