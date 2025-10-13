import pypsa
import numpy as np
import matplotlib.pyplot as plt

"""
Piecewise import acceptance via tranche-specific links

Goal:
- Allow UA to import cheap Polish generation freely (e.g., 30 €/MWh)
- Block UA from importing expensive Polish generation (>100 €/MWh),
  while still allowing UA to dispatch its own peaker at 150 €/MWh and
  shed at 300 €/MWh when necessary.

Approach:
- Split the exporting side into two buses: PL_base_bus (cheap) and PL_peak_bus (expensive)
- Connect generators accordingly
- Provide two parallel import links to UA:
  - From PL_base_bus with marginal_cost = 0 (no tariff)
  - From PL_peak_bus with marginal_cost = 50 €/MWh (= 150 - 100 EUR/MWh import cap)
    This ensures delivered cost for the expensive tranche is 110 + 50 = 160 €/MWh,
    making it more expensive than UA's peaker (150), so UA prefers domestic gen over
    importing above the cap.
"""

# Problem parameters
PRICE_CAP = 100  # €/MWh (policy cap for imports; used indirectly via link cost choice)
VOLL = 300  # €/MWh (value of lost load)

# Create network with 24 hourly snapshots
n = pypsa.Network()
n.set_snapshots(range(24))

# Buses
n.add("Bus", "UA")
n.add("Bus", "PL")  # Main PL bus for domestic load accounting
n.add("Bus", "PL_base_bus")  # Cheap Polish generation bus
n.add("Bus", "PL_peak_bus")  # Expensive Polish generation bus

# Loads (MW): step pattern and one UA spike to trigger scarcity
ua_load_profile = np.tile([20, 40, 60, 80, 100, 120], 4)
pl_load_profile = np.tile([20, 40, 60, 80, 100, 120], 4)
ua_load_profile[10] = 200  # Scarcity hour to show peaker/shedding

n.add("Load", "UA_load", bus="UA", p_set=ua_load_profile)
n.add("Load", "PL_load", bus="PL", p_set=pl_load_profile)

# Generators in Poland
n.add(
    "Generator",
    "PL_base",
    bus="PL_base_bus",
    p_nom=130,
    marginal_cost=30,
)

n.add(
    "Generator",
    "PL_peak",
    bus="PL_peak_bus",
    p_nom=40,
    marginal_cost=110,
)

# Connect tranche buses to main PL bus for domestic supply (free internal transfer)
for src in ["PL_base_bus", "PL_peak_bus"]:
    n.add(
        "Link",
        f"{src}_to_PL",
        bus0=src,
        bus1="PL",
        p_nom=1e3,
        efficiency=1.0,
        marginal_cost=0.0,
        p_min_pu=-1.0,  # allow both directions just in case
    )

# Ukraine domestic generators
n.add("Generator", "UA_base", bus="UA", p_nom=80, marginal_cost=35)
n.add("Generator", "UA_peak", bus="UA", p_nom=40, marginal_cost=150)

# Import links to UA
# - Cheap tranche: no tariff so delivered cost = 30 €/MWh when marginal
n.add(
    "Link",
    "PL_base_to_UA",
    bus0="PL_base_bus",
    bus1="UA",
    p_nom=60,
    efficiency=1.0,
    marginal_cost=0.0,
    p_min_pu=0.0,  # import only
)

# - Expensive tranche: add tariff so delivered cost = 110 + 50 = 160 €/MWh
#   making it more expensive than UA peaker (150); thus prevents importing above cap
n.add(
    "Link",
    "PL_peak_to_UA",
    bus0="PL_peak_bus",
    bus1="UA",
    p_nom=60,
    efficiency=1.0,
    marginal_cost=50.0,  # 150 - 100 cap = 50; enforces cap per expensive tranche
    p_min_pu=0.0,  # import only
)

# Load curtailment at UA (last resort)
n.add("Generator", "UA_load_curtailment", bus="UA", p_nom=200, marginal_cost=VOLL)

# Optimize
n.optimize(solver_name="highs")

# Results
ua_prices = n.buses_t.marginal_price["UA"]
pl_prices = n.buses_t.marginal_price["PL"]
imports_base = n.links_t.p0["PL_base_to_UA"]
imports_peak = n.links_t.p0["PL_peak_to_UA"]
ua_imports = imports_base + imports_peak

ua_base_gen = n.generators_t.p["UA_base"]
ua_peak_gen = n.generators_t.p["UA_peak"]
ua_curtail = n.generators_t.p["UA_load_curtailment"]
ua_load = n.loads_t.p["UA_load"]

print("\n" + "=" * 60)
print("PIECEWISE IMPORT RESULTS")
print("=" * 60)
print(
    f"UA price min/avg/max: {ua_prices.min():.1f} / {ua_prices.mean():.1f} / {ua_prices.max():.1f} €/MWh"
)
print(
    f"PL price min/avg/max: {pl_prices.min():.1f} / {pl_prices.mean():.1f} / {pl_prices.max():.1f} €/MWh"
)
print(f"Total imports (cheap tranche): {imports_base.sum():.1f} MWh")
print(f"Total imports (expensive tranche): {imports_peak.sum():.1f} MWh")
print(f"UA peaker generation: {ua_peak_gen.sum():.1f} MWh")
print(f"UA curtailment: {ua_curtail.sum():.1f} MWh")

# Expectation:
# - imports_peak should be ~0 if domestic peaker is cheaper than delivered expensive tranche

# Plot
hours = range(24)
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

ax1 = axes[0]
ax1.fill_between(hours, 0, ua_base_gen, label="UA Base Gen", alpha=0.7)
ax1.fill_between(
    hours, ua_base_gen, ua_base_gen + ua_peak_gen, label="UA Peak Gen", alpha=0.7
)
ax1.fill_between(
    hours,
    ua_base_gen + ua_peak_gen,
    ua_base_gen + ua_peak_gen + ua_imports,
    label="Imports (total)",
    alpha=0.7,
)
ax1.fill_between(
    hours,
    ua_base_gen + ua_peak_gen + ua_imports,
    ua_base_gen + ua_peak_gen + ua_imports + ua_curtail,
    label="Curtailment (VOLL)",
    alpha=0.7,
    color="red",
)
ax1.plot(hours, ua_load, "k--", lw=2, label="UA Load")
ax1.set_title("UA Dispatch with Piecewise Import Cap")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(hours, imports_base, label="Imports (cheap tranche)", marker="o")
ax2.plot(hours, imports_peak, label="Imports (expensive tranche)", marker="x")
ax2.set_ylabel("MW")
ax2.set_title("Import Flows by Tranche")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

ax3 = axes[2]
ax3.plot(hours, ua_prices, label="UA Price", marker="o")
ax3.plot(hours, pl_prices, label="PL Price", marker="s")
ax3.axhline(PRICE_CAP, color="r", ls="--", label=f"Import cap = {PRICE_CAP}")
ax3.set_ylabel("€/MWh")
ax3.set_title("Prices")
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(alpha=0.3)

ax4 = axes[3]
ax4.plot(hours, n.links_t.p0["PL_base_to_UA"], label="PL_base_to_UA")
ax4.plot(hours, n.links_t.p0["PL_peak_to_UA"], label="PL_peak_to_UA")
ax4.set_ylabel("MW")
ax4.set_xlabel("Hour")
ax4.set_title("Cross-Border Flows (PL → UA)")
ax4.legend(loc="upper left", fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("price_cap_piecewise.png", dpi=150, bbox_inches="tight")
print("Plot saved as price_cap_piecewise.png")
plt.show()
