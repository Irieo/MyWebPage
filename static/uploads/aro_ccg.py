"""
Adaptive Robust Optimization (ARO) of electricity system expansion
with Column-and-Constraint Generation (CCG) decomposition.

A 6-node power system and ARO problem used here is taken from a great book: Conejo et al. (2016). Investment in Electricity Generation and Transmission: Decision Making under Uncertainty. DOI: 10.1007/978-3-319-29501-5

The implementation of a column-and-constraint generation (or Berders-primal) algorithm is translated from GAMS implementation: https://github.com/Irieo/public-code-vault/tree/master/20191030_Robust_optimization_of_electricity_system_expansion/

IR, Feb 2026
"""

import warnings
import numpy as np
import pandas as pd
from linopy import Model

# Suppress linopy warning
warnings.filterwarnings("ignore", message="Coordinates across variables not equal")

# ------------------------------------------
# SETS
# ------------------------------------------
nodes = ["n1", "n2", "n3", "n4", "n5", "n6"]
gens = ["g1", "g2", "g3", "g4", "g5"]
dems = ["d1", "d2", "d3", "d4"]
lines = ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9"]
pros = ["l4", "l5", "l6", "l7", "l8", "l9"]  # prospective lines
ex = ["l1", "l2", "l3"]  # existing lines
ref = "n1"  # slack / reference node

n_idx = pd.Index(nodes, name="node")
g_idx = pd.Index(gens, name="gen")
d_idx = pd.Index(dems, name="dem")
l_idx = pd.Index(lines, name="line")

# ------------------------------------------
# MAPPINGS  (generator/demand node) and (line sending/receiving node)
# ------------------------------------------
mapG = {"g1": "n1", "g2": "n2", "g3": "n3", "g4": "n5", "g5": "n6"}
mapD = {"d1": "n1", "d2": "n4", "d3": "n5", "d4": "n6"}
mapSL = {
    "l1": "n1",
    "l2": "n1",
    "l3": "n4",
    "l4": "n2",
    "l5": "n2",
    "l6": "n3",
    "l7": "n3",
    "l8": "n4",
    "l9": "n5",
}
mapRL = {
    "l1": "n2",
    "l2": "n3",
    "l3": "n5",
    "l4": "n3",
    "l5": "n4",
    "l6": "n4",
    "l7": "n6",
    "l8": "n6",
    "l9": "n6",
}

# Reverse lookups
gens_at = {n: [g for g in gens if mapG[g] == n] for n in nodes}
dems_at = {n: [d for d in dems if mapD[d] == n] for n in nodes}
send_at = {n: [l for l in lines if mapSL[l] == n] for n in nodes}  # lines leaving n
recv_at = {n: [l for l in lines if mapRL[l] == n] for n in nodes}  # lines entering n

# ------------------------------------------
# DATA
# ------------------------------------------
LDATA = pd.DataFrame(
    {
        "B": [500, 500, 500, 500, 500, 500, 500, 500, 500],
        "FLmax": [150, 150, 150, 150, 150, 200, 200, 150, 150],
        "IC": [0, 0, 0, 700_000, 1_400_000, 1_800_000, 1_600_000, 800_000, 700_000],
    },
    index=lines,
)

DDATA = pd.DataFrame(
    {
        "PDmin": [180, 135, 90, 180],
        "PDmax": [220, 165, 110, 220],
        "LScost": [140, 152, 155, 165],
    },
    index=dems,
)

DDATA["LScost"] = (
    DDATA["LScost"] * 2
)  # bit more preference for building transmission rather than to shed load.

GDATA = pd.DataFrame(
    {
        "PEmin": [0, 0, 0, 0, 0],
        "PEmax": [300, 250, 400, 300, 150],
        "Gcost": [18, 25, 16, 32, 35],
    },
    index=gens,
)

SIGMA = 8760  # hours/year
M_BIG = 5_000_000  # big-M  (same as GAMS)
ILmax = 4_500_000  # investment budget (larger than in original example to explore more topologies)
TOL = 1e-6  # CCG convergence tolerance
MAX_IT = 20  # maximum CCG iterations


# ------------------------------------------
# Build master problem
# ------------------------------------------
def build_master(scenarios):
    """
    Build the master MILP
    """
    m = Model()

    # --- Investment variables ---
    x_m = m.add_variables(
        binary=True,
        coords=[pd.Index(pros, name="line")],
        name="x_m",
    )
    ETA = m.add_variables(lower=0, name="ETA")

    # --- Objective ---
    inv_cost = (LDATA.loc[pros, "IC"] * x_m).sum()
    m.add_objective(inv_cost + ETA, sense="min")

    # --- Investment budget ---
    m.add_constraints(
        (LDATA.loc[pros, "IC"] * x_m).sum() <= ILmax,
        name="inv_budget",
    )

    # --- Operational variables + constraints for each scenario v ---
    for v, scen in enumerate(scenarios):
        Pdem_v = scen["Pdem"]  # pd.Series indexed by dems
        PE_v = scen["PE"]  # pd.Series indexed by gens

        PG = m.add_variables(
            lower=0,
            upper=pd.Series(PE_v.values, index=g_idx, dtype=float),
            name=f"PG_{v}",
        )
        PLS = m.add_variables(
            lower=0,
            upper=pd.Series(Pdem_v.values, index=d_idx, dtype=float),
            name=f"PLS_{v}",
        )
        PL = m.add_variables(
            lower=-np.inf,
            upper=np.inf,
            coords=[l_idx],
            name=f"PL_{v}",
        )
        THETA = m.add_variables(
            lower=-3.14,
            upper=3.14,
            coords=[n_idx],
            name=f"THETA_{v}",
        )

        # ETA epigraph cut
        op_cost = SIGMA * ((GDATA["Gcost"] * PG).sum() + (DDATA["LScost"] * PLS).sum())
        m.add_constraints(ETA >= op_cost, name=f"eta_cut_{v}")

        # Reference node angle
        m.add_constraints(THETA.sel(node=ref) == 0, name=f"ref_angle_{v}")

        # Market clearing at each node
        for n in nodes:
            gl = gens_at[n]
            dl = dems_at[n]
            sl = send_at[n]
            rl = recv_at[n]
            gen_sum = PG.sel(gen=gl).sum() if gl else 0
            pls_sum = PLS.sel(dem=dl).sum() if dl else 0
            pd_sum = float(Pdem_v[dl].sum()) if dl else 0
            sf = PL.sel(line=sl).sum() if sl else 0
            rf = PL.sel(line=rl).sum() if rl else 0
            m.add_constraints(
                gen_sum - sf + rf == pd_sum - pls_sum,
                name=f"mc_{n}_{v}",
            )

        # Flow definitions — existing lines
        for l in ex:
            ns, nr = mapSL[l], mapRL[l]
            B = LDATA.loc[l, "B"]
            m.add_constraints(
                PL.sel(line=l) == B * (THETA.sel(node=ns) - THETA.sel(node=nr)),
                name=f"flowdef_ex_{l}_{v}",
            )
            m.add_constraints(
                PL.sel(line=l) <= LDATA.loc[l, "FLmax"],
                name=f"flow_ub_{l}_{v}",
            )
            m.add_constraints(
                PL.sel(line=l) >= -LDATA.loc[l, "FLmax"],
                name=f"flow_lb_{l}_{v}",
            )

        # Flow definitions — prospective lines (big-M linearisation of x_m * B * Δθ)
        for l in pros:
            ns, nr = mapSL[l], mapRL[l]
            B = LDATA.loc[l, "B"]
            Fmax = LDATA.loc[l, "FLmax"]
            xl = x_m.sel(line=l)
            dth = THETA.sel(node=ns) - THETA.sel(node=nr)

            # if x_m=0: PL=0; if x_m=1: PL = B*Δθ
            # big-M on flow def
            m.add_constraints(
                PL.sel(line=l) - B * dth >= -(1 - xl) * M_BIG,
                name=f"flowdefMILP_a_{l}_{v}",
            )
            m.add_constraints(
                PL.sel(line=l) - B * dth <= (1 - xl) * M_BIG,
                name=f"flowdefMILP_b_{l}_{v}",
            )
            # flow capacity (only active when x_m=1)
            m.add_constraints(PL.sel(line=l) <= xl * Fmax, name=f"flow_ub_{l}_{v}")
            m.add_constraints(PL.sel(line=l) >= -xl * Fmax, name=f"flow_lb_{l}_{v}")

    return m


# ------------------------------------------
# Build subproblem (KKT-linearised worst-case finder)
# ------------------------------------------
def build_subproblem(x_fixed, GAMMA_D, GAMMA_G):
    """
    Build the single-level MIP subproblem obtained by replacing the inner
    minimisation (dispatch) with its KKT conditions.

    The subproblem *maximises* operational cost over uncertain demand (Pdem)
    and generation availability (PE), subject to:
      - uncertainty budget constraints (GAMMA_D, GAMMA_G)
      - KKT stationarity + complementary slackness of the inner dispatch problem
    """
    m = Model()

    x = {l: x_fixed[l] for l in lines}  # x[l]=0 for existing lines (not invested)

    # ===========================================================
    # Uncertainty variables
    # ===========================================================
    Pdem = m.add_variables(
        lower=pd.Series(DDATA["PDmin"].values, index=d_idx, dtype=float),
        upper=pd.Series(DDATA["PDmax"].values, index=d_idx, dtype=float),
        name="Pdem",
    )
    PE = m.add_variables(
        lower=pd.Series(GDATA["PEmin"].values, index=g_idx, dtype=float),
        upper=pd.Series(GDATA["PEmax"].values, index=g_idx, dtype=float),
        name="PE",
    )

    # ===========================================================
    # Primal variables (inner dispatch)
    # ===========================================================
    PG = m.add_variables(
        lower=0,
        upper=pd.Series(GDATA["PEmax"].values, index=g_idx, dtype=float),
        name="PG",
    )
    PLS = m.add_variables(
        lower=0,
        upper=pd.Series(DDATA["PDmax"].values, index=d_idx, dtype=float),
        name="PLS",
    )
    PL = m.add_variables(lower=-np.inf, upper=np.inf, coords=[l_idx], name="PL")
    THETA = m.add_variables(lower=-3.14, upper=3.14, coords=[n_idx], name="THETA")

    # ===========================================================
    # Dual variables (KKT multipliers of inner problem)
    # ===========================================================
    lam = m.add_variables(lower=-np.inf, upper=np.inf, coords=[n_idx], name="lam")
    phi_l = m.add_variables(
        lower=-np.inf, upper=np.inf, coords=[pd.Index(ex, name="line")], name="phi_l"
    )
    phi_lp = m.add_variables(
        lower=-np.inf, upper=np.inf, coords=[pd.Index(pros, name="line")], name="phi_lp"
    )
    phi_lmax = m.add_variables(lower=0, coords=[l_idx], name="phi_lmax")
    phi_lmin = m.add_variables(lower=0, coords=[l_idx], name="phi_lmin")
    phi_Emax = m.add_variables(lower=0, coords=[g_idx], name="phi_Emax")
    phi_Emin = m.add_variables(lower=0, coords=[g_idx], name="phi_Emin")
    phi_Dmax = m.add_variables(lower=0, coords=[d_idx], name="phi_Dmax")
    phi_Dmin = m.add_variables(lower=0, coords=[d_idx], name="phi_Dmin")
    phi_Nmax = m.add_variables(lower=0, coords=[n_idx], name="phi_Nmax")
    phi_Nmin = m.add_variables(lower=0, coords=[n_idx], name="phi_Nmin")
    xi_ref = m.add_variables(lower=-np.inf, upper=np.inf, name="xi_ref")

    # ===========================================================
    # Binary variables for complementary slackness linearisation
    # ===========================================================
    u1 = m.add_variables(binary=True, coords=[l_idx], name="u1")  # phi_lmax
    u2 = m.add_variables(binary=True, coords=[l_idx], name="u2")  # phi_lmin
    u3 = m.add_variables(binary=True, coords=[g_idx], name="u3")  # phi_Emax
    u4 = m.add_variables(binary=True, coords=[g_idx], name="u4")  # phi_Emin
    u5 = m.add_variables(binary=True, coords=[d_idx], name="u5")  # phi_Dmax
    u6 = m.add_variables(binary=True, coords=[d_idx], name="u6")  # phi_Dmin
    u7 = m.add_variables(binary=True, coords=[n_idx], name="u7")  # phi_Nmax
    u8 = m.add_variables(binary=True, coords=[n_idx], name="u8")  # phi_Nmin

    # ===========================================================
    # Objective: maximise operational cost (adversary's goal)
    # ===========================================================
    obj = SIGMA * ((GDATA["Gcost"] * PG).sum() + (DDATA["LScost"] * PLS).sum())
    m.add_objective(obj, sense="max")

    # ===========================================================
    # UNCERTAINTY BUDGET CONSTRAINTS
    # ===========================================================
    denom_D = float((DDATA["PDmax"] - DDATA["PDmin"]).sum())
    denom_G = float(GDATA["PEmax"].sum())

    m.add_constraints(
        (Pdem - DDATA["PDmin"].values).sum() <= GAMMA_D * denom_D,
        name="budget_D",
    )
    m.add_constraints(
        (GDATA["PEmax"].values - PE).sum() <= GAMMA_G * denom_G,
        name="budget_G",
    )

    # ===========================================================
    # PRIMAL FEASIBILITY (inner dispatch problem constraints)
    # ===========================================================

    # Reference angle
    m.add_constraints(THETA.sel(node=ref) == 0, name="ref_angle")

    # Market clearing at each node
    for n in nodes:
        gl = gens_at[n]
        dl = dems_at[n]
        sl = send_at[n]
        rl = recv_at[n]
        gen_sum = PG.sel(gen=gl).sum() if gl else 0
        pls_sum = PLS.sel(dem=dl).sum() if dl else 0
        sf = PL.sel(line=sl).sum() if sl else 0
        rf = PL.sel(line=rl).sum() if rl else 0
        m.add_constraints(
            gen_sum - sf + rf == Pdem.sel(dem=dl).sum() - pls_sum
            if dl
            else gen_sum - sf + rf == 0,
            name=f"mc_{n}",
        )

    # Flow definitions — existing lines
    for l in ex:
        ns, nr = mapSL[l], mapRL[l]
        B = float(LDATA.loc[l, "B"])
        m.add_constraints(
            PL.sel(line=l) == B * (THETA.sel(node=ns) - THETA.sel(node=nr)),
            name=f"flowdef_{l}",
        )

    # Flow definitions — prospective lines (x is a fixed parameter here)
    for l in pros:
        ns, nr = mapSL[l], mapRL[l]
        B = float(LDATA.loc[l, "B"])
        xl = float(x[l])
        m.add_constraints(
            PL.sel(line=l) == xl * B * (THETA.sel(node=ns) - THETA.sel(node=nr)),
            name=f"flowdef_{l}",
        )

    # Generation capacity: 0 <= PG <= PE  (PG upper set dynamically below)
    m.add_constraints(PG <= PE, name="gen_cap")

    # Load shedding capacity: 0 <= PLS <= Pdem
    m.add_constraints(PLS <= Pdem, name="ls_cap")

    # Flow limits
    m.add_constraints(PL <= LDATA["FLmax"].values, name="flow_ub")
    m.add_constraints(PL >= -LDATA["FLmax"].values, name="flow_lb")

    # ===========================================================
    # KKT STATIONARITY CONDITIONS (gradient of Lagrangian = 0)
    # ===========================================================

    # FOC w.r.t. PG(g):  SIGMA*Gcost - lam(n(g)) + phi_Emax - phi_Emin = 0
    for g in gens:
        n = mapG[g]
        m.add_constraints(
            SIGMA * float(GDATA.loc[g, "Gcost"])
            - lam.sel(node=n)
            + phi_Emax.sel(gen=g)
            - phi_Emin.sel(gen=g)
            == 0,
            name=f"foc_PG_{g}",
        )

    # FOC w.r.t. PLS(d):  SIGMA*LScost - lam(n(d)) + phi_Dmax - phi_Dmin = 0
    for d in dems:
        n = mapD[d]
        m.add_constraints(
            SIGMA * float(DDATA.loc[d, "LScost"])
            - lam.sel(node=n)
            + phi_Dmax.sel(dem=d)
            - phi_Dmin.sel(dem=d)
            == 0,
            name=f"foc_PLS_{d}",
        )

    # FOC w.r.t. PL(l) — existing lines:
    #   lam(sl) - lam(rl) - phi_l + phi_lmax - phi_lmin = 0
    for l in ex:
        ns, nr = mapSL[l], mapRL[l]
        m.add_constraints(
            lam.sel(node=ns)
            - lam.sel(node=nr)
            - phi_l.sel(line=l)
            + phi_lmax.sel(line=l)
            - phi_lmin.sel(line=l)
            == 0,
            name=f"foc_PL_ex_{l}",
        )

    # FOC w.r.t. PL(l) — prospective lines:
    #   lam(sl) - lam(rl) - phi_lplus + phi_lmax - phi_lmin = 0
    for l in pros:
        ns, nr = mapSL[l], mapRL[l]
        m.add_constraints(
            lam.sel(node=ns)
            - lam.sel(node=nr)
            - phi_lp.sel(line=l)
            + phi_lmax.sel(line=l)
            - phi_lmin.sel(line=l)
            == 0,
            name=f"foc_PL_pros_{l}",
        )

    # FOC w.r.t. THETA(n) — non-reference nodes:
    #   sum_{l in ex: sl=n} B*phi_l  -  sum_{l in ex: rl=n} B*phi_l
    # + sum_{l in pros: sl=n} x*B*phi_lp  - sum_{l in pros: rl=n} x*B*phi_lp
    # + phi_Nmax - phi_Nmin = 0
    for n in nodes:
        sl_ex = [l for l in ex if mapSL[l] == n]
        rl_ex = [l for l in ex if mapRL[l] == n]
        sl_pros = [l for l in pros if mapSL[l] == n]
        rl_pros = [l for l in pros if mapRL[l] == n]

        expr_foc = phi_Nmax.sel(node=n) - phi_Nmin.sel(node=n)
        for l in sl_ex:
            expr_foc = expr_foc + float(LDATA.loc[l, "B"]) * phi_l.sel(line=l)
        for l in rl_ex:
            expr_foc = expr_foc - float(LDATA.loc[l, "B"]) * phi_l.sel(line=l)
        for l in sl_pros:
            expr_foc = expr_foc + float(x[l]) * float(LDATA.loc[l, "B"]) * phi_lp.sel(
                line=l
            )
        for l in rl_pros:
            expr_foc = expr_foc - float(x[l]) * float(LDATA.loc[l, "B"]) * phi_lp.sel(
                line=l
            )

        if n == ref:
            # Reference node: includes xi_ref (multiplier for theta_ref = 0)
            m.add_constraints(expr_foc - xi_ref == 0, name=f"foc_THETA_{n}")
        else:
            m.add_constraints(expr_foc == 0, name=f"foc_THETA_{n}")

    # ===========================================================
    # COMPLEMENTARY SLACKNESS (linearised via big-M)
    # kkt1: phi_lmax * (FLmax - PL) = 0
    # kkt2: phi_lmin * (PL + FLmax) = 0
    # kkt3: phi_Emax * (PE - PG)    = 0
    # kkt4: phi_Emin * PG           = 0
    # kkt5: phi_Dmax * (Pdem - PLS) = 0
    # kkt6: phi_Dmin * PLS          = 0
    # kkt7: phi_Nmax * (pi - THETA) = 0
    # kkt8: phi_Nmin * (THETA + pi) = 0
    # ===========================================================

    # kkt1: phi_lmax ∈ [0, M*u1],  FLmax - PL ∈ [0, M*(1-u1)]
    m.add_constraints(phi_lmax <= M_BIG * u1, name="kkt1c")
    m.add_constraints(LDATA["FLmax"].values - PL <= M_BIG * (1 - u1), name="kkt1d")

    # kkt2: phi_lmin ∈ [0, M*u2],  PL + FLmax ∈ [0, M*(1-u2)]
    m.add_constraints(phi_lmin <= M_BIG * u2, name="kkt2c")
    m.add_constraints(PL + LDATA["FLmax"].values <= M_BIG * (1 - u2), name="kkt2d")

    # kkt3: phi_Emax ∈ [0, M*u3],  PE - PG ∈ [0, M*(1-u3)]
    m.add_constraints(phi_Emax <= M_BIG * u3, name="kkt3c")
    m.add_constraints(PE - PG <= M_BIG * (1 - u3), name="kkt3d")

    # kkt4: phi_Emin ∈ [0, M*u4],  PG ∈ [0, M*(1-u4)]
    m.add_constraints(phi_Emin <= M_BIG * u4, name="kkt4c")
    m.add_constraints(PG <= M_BIG * (1 - u4), name="kkt4d")

    # kkt5: phi_Dmax ∈ [0, M*u5],  Pdem - PLS ∈ [0, M*(1-u5)]
    m.add_constraints(phi_Dmax <= M_BIG * u5, name="kkt5c")
    m.add_constraints(Pdem - PLS <= M_BIG * (1 - u5), name="kkt5d")

    # kkt6: phi_Dmin ∈ [0, M*u6],  PLS ∈ [0, M*(1-u6)]
    m.add_constraints(phi_Dmin <= M_BIG * u6, name="kkt6c")
    m.add_constraints(PLS <= M_BIG * (1 - u6), name="kkt6d")

    # kkt7: phi_Nmax ∈ [0, M*u7],  pi - THETA ∈ [0, M*(1-u7)]
    m.add_constraints(phi_Nmax <= M_BIG * u7, name="kkt7c")
    m.add_constraints(3.14 - THETA <= M_BIG * (1 - u7), name="kkt7d")

    # kkt8: phi_Nmin ∈ [0, M*u8],  THETA + pi ∈ [0, M*(1-u8)]
    m.add_constraints(phi_Nmin <= M_BIG * u8, name="kkt8c")
    m.add_constraints(THETA + 3.14 <= M_BIG * (1 - u8), name="kkt8d")

    return m


# ------------------------------------------
# CCG ALGORITHM
# ------------------------------------------
def run_ccg(GAMMA_D=0.0, GAMMA_G=0.0, solver="highs", verbose=True):
    """Run the CCG decomposition for given uncertainty budgets."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  CCG  |  GAMMA_D={GAMMA_D:.1f}  GAMMA_G={GAMMA_G:.1f}")
        print(f"{'=' * 60}")

    Z_Lower = -np.inf
    Z_Upper = np.inf
    scenarios = []  # list of {Pdem: Series, PE: Series}
    x_sol = None

    # Initialise with nominal scenario (PDmin for demand, PEmax for generation)
    nominal = {
        "Pdem": DDATA["PDmin"].copy(),
        "PE": GDATA["PEmax"].copy(),
    }
    scenarios.append(nominal)

    for it in range(1, MAX_IT + 1):
        # ---- MASTER ------------------------
        master = build_master(scenarios)
        status, cond = master.solve(
            solver_name=solver,
            io_api="lp",
            log_fn=None,
            keep_files=False,
            output_flag=False,
        )
        if cond not in ("optimal", "Optimal"):
            print(f"Master returned {status}/{cond} — stopping.")
            break

        inv_cost = float(
            sum(
                LDATA.loc[l, "IC"] * master.solution[f"x_m"].sel(line=l).item()
                for l in pros
            )
        )
        ETA_val = float(master.solution["ETA"])
        Z_Lower = inv_cost + ETA_val

        x_sol = {l: float(master.solution["x_m"].sel(line=l).item()) for l in pros}
        x_all = {l: 0.0 for l in ex}
        x_all.update(x_sol)

        if verbose:
            built = [l for l in pros if x_sol[l] > 0.5]  # TODO why this threshold?
            print(
                f"  It {it:2d} | LB={Z_Lower:>14.0f} | UB={Z_Upper:>14.0f}"
                f" | built={built or '—'}"
            )

        # ---- CONVERGENCE CHECK ------------------------
        # Avoid division by zero on first iteration (UB still inf)
        if Z_Upper < np.inf and abs(1 - Z_Lower / Z_Upper) < TOL:
            if verbose:
                print(f"Converged at iteration {it}!")
            break

        # ---- SUBPROBLEM ------------------------
        sub = build_subproblem(x_all, GAMMA_D, GAMMA_G)
        s_status, s_cond = sub.solve(
            solver_name=solver,
            io_api="lp",
            log_fn=None,
            keep_files=False,
            output_flag=False,
        )
        if s_cond not in ("optimal", "Optimal"):
            print(f"Subproblem returned {s_status}/{s_cond} — stopping.")
            break

        Z_sub = float(sub.objective.value)
        Z_Upper = min(Z_Upper, inv_cost + Z_sub)

        # Worst-case scenario to add to master
        new_scen = {
            "Pdem": pd.Series(sub.solution["Pdem"].values, index=dems),
            "PE": pd.Series(sub.solution["PE"].values, index=gens),
        }
        scenarios.append(new_scen)

        if verbose:
            print(f" UB updated → {Z_Upper:>14.0f}  (Z_sub={Z_sub:.0f})")

        if abs(1 - Z_Lower / Z_Upper) < TOL:
            if verbose:
                print(f"Converged at iteration {it}!")
            break
    else:
        if verbose:
            print(f"Reached max iterations ({MAX_IT}).")

    built_lines = [l for l in pros if x_sol and x_sol[l] > 0.5]
    return {
        "x": x_sol,
        "built": built_lines,
        "Z_Lower": Z_Lower,
        "Z_Upper": Z_Upper,
        "inv_cost": inv_cost,
        "iterations": it,
        "scenarios": scenarios,
    }


# ------------------------------------
# MAIN
# ------------------------------------
if __name__ == "__main__":
    # GAMMA_G & GAMMA_D steps
    gamma_G = [0.0, 0.2, 0.4, 0.6, 0.8]
    gamma_D = [0.2]

    report = {}
    for g_G in gamma_G:
        for g_D in gamma_D:
            res = run_ccg(GAMMA_D=g_D, GAMMA_G=g_G, solver="highs", verbose=True)
            report[(g_G, g_D)] = res

    # ---- Pretty-print summary ----
    print(f"\n{'=' * 70}")
    print("  DETAIL: Built lines per scenario")
    print(f"{'=' * 70}")
    for (g_G, g_D), r in sorted(report.items()):
        built_str = str(r["built"]) if r["built"] else "none"
        print(
            f"  GAMMA_G={g_G:.1f}  GAMMA_D={g_D:.1f} | "
            f"built={built_str:<22s} | "
            f"total={r['Z_Upper'] / 1e6:.3f}M  "
            f"iters={r['iterations']}"
        )
