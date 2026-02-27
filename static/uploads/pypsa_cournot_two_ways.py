"""
Compare two approaches to implementing Cournot-Nash competition in PyPSA.

Approach 1 (notebook approach)
-------------------------------
Build the network normally, then append the Cournot markup term
    (b * cv / 2) * q_i^2
to the linopy objective after create_model().

Approach 2
---------------------------------
Bake the markup directly into marginal_cost_quadratic before create_model():
    marginal_cost_quadratic = QUADRATIC_COST + b * cv / 2

Result
------
Both approaches are fully equivalent — identical dispatch quantities AND
identical bus shadow prices (market-clearing prices).

Why: the bus shadow price equals the effective marginal cost at optimality.
The KKT condition for q_i yields the same expression in both cases:

  Approach 1:  MC_quad = 0.5,  explicit markup +4*q_i
               → lambda_gas = 1 + q_i + 4*q_i = 1 + 5*q_i

  Approach 2:  MC_quad = 0.5 + 2.0 = 2.5,  no explicit markup
               → lambda_gas = 1 + 2*2.5*q_i = 1 + 5*q_i  (same)
"""

import warnings
import pandas as pd
import pypsa

warnings.filterwarnings("error", category=DeprecationWarning)
pypsa.options.params.optimize.log_to_console = False

# ---------------------------------------------------------------------------
# Parameters

DEMAND_INTERCEPT = 30  # a  [EUR/MWh]
DEMAND_SLOPE = 4  # b  [EUR/MWh per MW]
LINEAR_COST = 1  # [EUR/MWh]
QUADRATIC_COST = 0.5  # [EUR/MWh per MW]

SOLAR_CAPACITY = 3.0
SOLAR_CF = [0.8, 0.5, 0.1]
GAS_PLANT_CAPACITY = 8.0
GAS_PRODUCER_CAPACITY = 5.0
DEMAND_MAX = 10.0

CV = 1  # conjectural variation = 1 → Cournot-Nash
COURNOT_MARKUP = DEMAND_SLOPE * CV / 2  # b*cv/2 = 2.0

# ---------------------------------------------------------------------------
# Network factory


def create_network(marginal_cost_quadratic_gas=QUADRATIC_COST):
    """Create the two-bus electricity + gas network.

    marginal_cost_quadratic_gas lets us optionally bake in the Cournot markup.
    """
    n = pypsa.Network()
    n.set_snapshots(range(3))

    n.add("Bus", "electricity")
    n.add("Bus", "gas")

    n.add(
        "Generator",
        "solar",
        bus="electricity",
        p_nom=SOLAR_CAPACITY,
        marginal_cost=0,
        p_max_pu=SOLAR_CF,
    )

    # Elastic demand: inverse demand curve P(d) = 30 - 4*d
    n.add(
        "Generator",
        "elastic_demand",
        bus="electricity",
        sign=-1,
        p_nom=DEMAND_MAX,
        marginal_cost=-DEMAND_INTERCEPT,
        marginal_cost_quadratic=DEMAND_SLOPE / 2,
    )

    for i in [1, 2]:
        n.add(
            "Generator",
            f"gas_producer_{i}",
            bus="gas",
            p_nom=GAS_PRODUCER_CAPACITY,
            marginal_cost=LINEAR_COST,
            marginal_cost_quadratic=marginal_cost_quadratic_gas,
        )

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


# ---------------------------------------------------------------------------
# Approach 1 — post create_model() objective modification (notebook example approach)

n1 = create_network()
m1 = n1.optimize.create_model()

q1 = m1["Generator-p"].sel(name="gas_producer_1")
q2 = m1["Generator-p"].sel(name="gas_producer_2")

cournot_markup = COURNOT_MARKUP * (q1 * q1 + q2 * q2)
m1.objective = m1.objective.expression + cournot_markup.sum()

n1.optimize.solve_model(solver_name="highs")

# ---------------------------------------------------------------------------
# Approach 2 — markup baked into marginal_cost_quadratic

n2 = create_network(marginal_cost_quadratic_gas=QUADRATIC_COST + COURNOT_MARKUP)
n2.optimize(solver_name="highs")

# ---------------------------------------------------------------------------
# Compare results
dispatch1 = n1.generators_t.p[["gas_producer_1", "gas_producer_2"]]
dispatch2 = n2.generators_t.p[["gas_producer_1", "gas_producer_2"]]

gas_price1 = n1.buses_t.marginal_price["gas"]
gas_price2 = n2.buses_t.marginal_price["gas"]

elec_price1 = n1.buses_t.marginal_price["electricity"]
elec_price2 = n2.buses_t.marginal_price["electricity"]

print("\nDispatch quantities (should be identical)")
cmp_dispatch = pd.DataFrame(
    {
        "Approach 1 q1": dispatch1["gas_producer_1"],
        "Approach 2 q1": dispatch2["gas_producer_1"],
        "Approach 1 q2": dispatch1["gas_producer_2"],
        "Approach 2 q2": dispatch2["gas_producer_2"],
    }
)
print(cmp_dispatch.round(6).to_string())
print(
    f"\nMax absolute difference in dispatch: "
    f"{(dispatch1.values - dispatch2.values).__abs__().max():.2e}"
)

print("\nGas bus shadow price")
cmp_gas = pd.DataFrame({"Approach 1": gas_price1, "Approach 2": gas_price2})
print(cmp_gas.round(6).to_string())

print("\nElectricity bus shadow price")
cmp_elec = pd.DataFrame({"Approach 1": elec_price1, "Approach 2": elec_price2})
print(cmp_elec.round(6).to_string())
