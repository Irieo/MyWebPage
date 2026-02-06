"""
Imperfect competition games using linopy | IR, 6-7 Feb 2026

Adapted from GAMS code (IR ESM Tutorial 12):
https://github.com/Irieo/EnergySystemsModelling-course/blob/master/tutorial%2011%2612%20Lagrange%20multipliers.%20MCPs.%20Cournot%20competition.%20(08.01.20)/Code/Tutorial%2012.gms

Problem setup:
    - Two producers: p1, p2
    - Cost function:  C(q) = q + 0.5 * q^2
    - Inverse demand: P = 30 - 4 * (q1 + q2)

The GAMS model solves MCP (Mixed Complementarity Problems) with conjectural
variations (cv). Since linopy does not have a PATH solver for complementarity,
we reformulate the MCP using Big-M constraints:

    FOC(p): -P + 4*cv(p)*q(p) + 1 + q(p) >= 0,  q(p) >= 0  (complementary)
    Demand: P = 30 - 4*(q1 + q2)

The complementarity condition  f(q) >= 0 ⊥ q >= 0  means:
    f(q) >= 0,  q >= 0,  and  q * f(q) = 0
This is linearized via Big-M with binary variables z(p):
    q(p) <= M * z(p)              (z=0 forces q=0)
    f(q(p)) <= M * (1 - z(p))    (z=1 forces f=0)

Scenarios:
    a)  Perfect competition:         cv1=0,   cv2=0
    b)  Cournot-Nash (symmetric):    cv1=1,   cv2=1
    c1) Asymmetric Cournot (myopic): cv1=1,   cv2=0
    c2) Profit maximization of p1 against p2's reaction function (NLP)
    c3) Conjectured variation:       cv1=0.2, cv2=0
    d1) Iterate cv1 from 0 to 1
    d2) Iterate both cv1 and cv2 from 0 to 1
"""

import numpy as np
import pandas as pd
import linopy


# ============================================================================
# Let's prepare math
# ============================================================================


def solve_cournot(cv1: float, cv2: float, M: float = 1000) -> dict:
    """
    Solve the Cournot game as an MCP using Big-M reformulation.

    The GAMS MCP conditions are:
        FOC(p):  -price + 4*cv(p)*q(p) + (1 + q(p)) >= 0  ⊥  q(p) >= 0
        Demand:  price = 30 - 4*(q1 + q2)

    Big-M linearization of complementarity  f >= 0 ⊥ q >= 0:
        q  <= M * z          (z=0 => q=0)
        f  <= M * (1 - z)    (z=1 => f=0)
        f  >= 0
        q  >= 0
        z  ∈ {0, 1}

    Since we solve system of equations (MCP has no objective), here I use a trick I picked from the N. Andrei's book "Nonlinear Optimization Applications": minimize a dummy objective 0
    """
    m = linopy.Model()

    # --- Variables ---
    q1 = m.add_variables(lower=0, name="q1")
    q2 = m.add_variables(lower=0, name="q2")
    price = m.add_variables(name="price")  # free variable

    # Binary variables for complementarity
    z1 = m.add_variables(binary=True, name="z1")
    z2 = m.add_variables(binary=True, name="z2")

    # Slack variables for FOC values (foc >= 0)
    foc1 = m.add_variables(lower=0, name="foc1")
    foc2 = m.add_variables(lower=0, name="foc2")

    # --- Constraints ---

    # Demand function (equality): price = 30 - 4*(q1 + q2)
    m.add_constraints(price - 30 + 4 * q1 + 4 * q2 == 0, name="demand")

    # FOC definitions: foc(p) = -price + 4*cv(p)*q(p) + 1 + q(p)
    m.add_constraints(foc1 == -price + (1 + 4 * cv1) * q1 + 1, name="foc1_def")
    m.add_constraints(foc2 == -price + (1 + 4 * cv2) * q2 + 1, name="foc2_def")

    # Big-M complementarity: either q=0 or foc=0
    m.add_constraints(q1 <= M * z1, name="bigM_q1")
    m.add_constraints(foc1 <= M * (1 - z1), name="bigM_foc1")
    m.add_constraints(q2 <= M * z2, name="bigM_q2")
    m.add_constraints(foc2 <= M * (1 - z2), name="bigM_foc2")

    # Dummy objective
    m.add_objective(0 * q1, sense="min")

    m.solve(solver_name="highs")

    q1_val = m.solution["q1"].item()
    q2_val = m.solution["q2"].item()
    price_val = m.solution["price"].item()
    cost1 = q1_val + 0.5 * q1_val**2
    cost2 = q2_val + 0.5 * q2_val**2
    profit1 = price_val * q1_val - cost1
    profit2 = price_val * q2_val - cost2

    return {
        "q1": q1_val,
        "q2": q2_val,
        "price": price_val,
        "profit1": profit1,
        "profit2": profit2,
    }


def solve_profit_max_player1() -> dict:
    """
    Scenario c2: Player 1 maximizes profit knowing player 2's reaction function.

    Player 2 plays perfect competition (cv2=0), so FOC gives:
        q2 = (29 - 4*q1) / 5

    Player 1 substitutes this into their profit function and maximizes:
        max pi1 = (30 - 4*(q1 + q2(q1))) * q1 - q1 - 0.5*q1^2
                = (29/5)*q1 - (13/10)*q1^2
    """
    m = linopy.Model()

    q1 = m.add_variables(lower=0, name="q1")

    # Objective: (29/5)*q1 - (13/10)*q1^2
    # See in GAMS formulation: prof_max = (30-4*(q1+29/5-4/5*q1))*q1-q1-0.5*q1**2
    obj = (29.0 / 5) * q1 - (13.0 / 10) * q1**2
    m.add_objective(obj, sense="max")

    m.solve(solver_name="highs")

    q1_val = m.solution["q1"].item()
    q2_val = 29.0 / 5 - (4.0 / 5) * q1_val  # from reaction function
    price = 30 - 4 * (q1_val + q2_val)
    cost1 = q1_val + 0.5 * q1_val**2
    cost2 = q2_val + 0.5 * q2_val**2
    profit1 = price * q1_val - cost1
    profit2 = price * q2_val - cost2

    return {
        "q1": q1_val,
        "q2": q2_val,
        "price": price,
        "profit1": profit1,
        "profit2": profit2,
    }


def print_results(label: str, res: dict) -> None:
    """Print scenario results in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  q1      = {res['q1']:.4f}")
    print(f"  q2      = {res['q2']:.4f}")
    print(f"  price   = {res['price']:.4f}")
    print(f"  profit1 = {res['profit1']:.4f}")
    print(f"  profit2 = {res['profit2']:.4f}")


# ============================================================================
# We're ready to solve
# ============================================================================

results = {}

# ------------------------------------------------------------------
# a) Perfect competition: cv1=0, cv2=0
#     Both players are price-takers (cv=0): they ignore their impact
#     on the market price. Yields the highest total output and lowest price.
# ------------------------------------------------------------------
results["Perfect competition"] = solve_cournot(cv1=0, cv2=0)
print_results("a) Perfect competition (cv1=0, cv2=0)", results["Perfect competition"])

# ------------------------------------------------------------------
# b) Cournot-Nash: cv1=1, cv2=1
#     Both players internalize their effect on the market price (cv=1).
#     Symmetric Nash equilibrium: lower output, higher price and profits.
# ------------------------------------------------------------------
results["Cournot competition"] = solve_cournot(cv1=1, cv2=1)
print_results("b) Cournot-Nash (cv1=1, cv2=1)", results["Cournot competition"])

# ------------------------------------------------------------------
# c1) One player exerts market power, the other plays perfect
#     competition (myopic): cv1=1, cv2=0
#     Player 1 acts strategically while player 2 remains a price-taker.
#     "Myopic" because player 1 doesn't anticipate player 2's reaction.
# ------------------------------------------------------------------
results["Cournot 1st (myopic)"] = solve_cournot(cv1=1, cv2=0)
print_results(
    "c1) Asymmetric Cournot - myopic (cv1=1, cv2=0)",
    results["Cournot 1st (myopic)"],
)

# ------------------------------------------------------------------
# c2) Maximization of player 1's profit (Stackelberg-like)
#     Player 1 anticipates player 2's reaction function and optimizes
#     against it. This is an NLP, not an MCP.
# ------------------------------------------------------------------
results["profit_max 1st"] = solve_profit_max_player1()
print_results("c2) Profit maximization of player 1", results["profit_max 1st"])

# ------------------------------------------------------------------
# c3) Conjectured variation: cv1=0.2, cv2=0
#     From c1 (myopic), player 2's reaction function is q2 = (29-4*q1)/5,
#     so dq2/dq1 = -4/5 = -0.8. The conjectured variation is
#     cv = 1 - |dq2/dq1| = 1 - 0.8 = 0.2
# ------------------------------------------------------------------
results["Conjecture 1st"] = solve_cournot(cv1=0.2, cv2=0)
print_results("c3) Conjectured variation (cv1=0.2, cv2=0)", results["Conjecture 1st"])

# ------------------------------------------------------------------
# Summary table for scenarios a-c
# ------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print("  Summary Report (scenarios a - c3)")
print(f"{'=' * 70}")
df_report = pd.DataFrame(results).T
print(df_report.to_string(float_format="{:.4f}".format))

# ------------------------------------------------------------------
# d1) Iterate cv1 from 0 to 1 (step 0.05), cv2=0
#     Sweep cv1 while player 2 stays competitive (cv2=0).
#     Empirically confirms that cv1=0.2 maximizes player 1's profit,
#     matching the analytical result from c2/c3.
# ------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print("  d1) Iterating cv1 from 0 to 1 (cv2=0)")
print(f"{'=' * 70}")
cv_values = np.arange(0, 1.05, 0.05)
records_d1 = []
for cv1 in cv_values:
    res = solve_cournot(cv1=cv1, cv2=0)
    res["cv1"] = cv1
    res["total_profit"] = res["profit1"] + res["profit2"]
    records_d1.append(res)

df_d1 = pd.DataFrame(records_d1)
df_d1 = df_d1[["cv1", "q1", "q2", "price", "profit1", "profit2", "total_profit"]]
print(df_d1.to_string(index=False, float_format="{:.4f}".format))

# ------------------------------------------------------------------
# d2) Iterate both cv1 and cv2 from 0 to 1 (step 0.05)
#     Sweep both cv values symmetrically. Empirically confirms that
#     cv1=cv2=1 is the Nash-Cournot equilibrium (scenario b).
# ------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print("  d2) Iterating cv1 and cv2 from 0 to 1")
print(f"{'=' * 70}")
records_d2 = []
for cv in cv_values:
    res = solve_cournot(cv1=cv, cv2=cv)
    res["cv1/cv2"] = cv
    res["total_profit"] = res["profit1"] + res["profit2"]
    records_d2.append(res)

df_d2 = pd.DataFrame(records_d2)
df_d2 = df_d2[["cv1/cv2", "q1", "q2", "price", "profit1", "profit2", "total_profit"]]
print(df_d2.to_string(index=False, float_format="{:.4f}".format))
