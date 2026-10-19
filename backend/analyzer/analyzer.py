"""Structural analysis of fgPROMS models.

Produces:
  * degrees-of-freedom counting,
  * variable/parameter tables with evaluated defaults,
  * equation list with incidence information,
  * detection of dynamic (differential) variables and schedule validation,
  * diagnostic messages (all message codes are resolved client-side).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..parser.gproms_parser import (
    Model, Num, Var, Deriv, UnOp, BinOp, Func, IfExpr,
    EqEquation, IfEquation, expr_str,
)
from ..properties import PROP_FUNCS, COOLPROP_AVAILABLE
from ..solver.solver import make_eval, SUPPORTED_FUNCS


def _collect(node, vars_out: Set[str], derivs_out: Set[str]) -> None:
    if isinstance(node, Var):
        vars_out.add(node.name.lower())
    elif isinstance(node, Deriv):
        derivs_out.add(node.name.lower())
    elif isinstance(node, UnOp):
        _collect(node.x, vars_out, derivs_out)
    elif isinstance(node, BinOp):
        _collect(node.l, vars_out, derivs_out)
        _collect(node.r, vars_out, derivs_out)
    elif isinstance(node, Func):
        for a in node.args:
            _collect(a, vars_out, derivs_out)
    elif isinstance(node, IfExpr):
        _collect(node.cond, vars_out, derivs_out)
        _collect(node.then, vars_out, derivs_out)
        _collect(node.els, vars_out, derivs_out)


def _collect_funcs(node, out: Set[str]) -> None:
    if isinstance(node, Func):
        out.add(node.name)
        for a in node.args:
            _collect_funcs(a, out)
    elif isinstance(node, UnOp):
        _collect_funcs(node.x, out)
    elif isinstance(node, BinOp):
        _collect_funcs(node.l, out)
        _collect_funcs(node.r, out)
    elif isinstance(node, IfExpr):
        _collect_funcs(node.cond, out)
        _collect_funcs(node.then, out)
        _collect_funcs(node.els, out)


def _eval_const(node, env: Dict[str, float]) -> Optional[float]:
    if node is None:
        return None
    if isinstance(node, Num):
        return float(node.v)
    try:
        return float(make_eval(node)(env))
    except Exception:
        return None


def _display_eq(eq: EqEquation) -> str:
    if eq.op == "=":
        return f"{expr_str(eq.lhs)} = {expr_str(eq.rhs)}"
    return f"{expr_str(eq.lhs)} {eq.op} {expr_str(eq.rhs)}"


def analyze(model: Model) -> dict:
    messages: List[dict] = []

    # ---- declarations ------------------------------------------------------
    params: List[dict] = []
    variables: List[dict] = []
    seen: Set[str] = set()

    for d in model.parameters:
        for nm in d.names:
            key = nm.lower()
            if key in seen:
                messages.append({"level": "warn", "code": "warn_duplicate",
                                 "args": {"name": nm}})
                continue
            seen.add(key)
            params.append({"name": nm, "lower": key, "type": d.type_text,
                           "default_node": d.default, "line": d.line})
    for d in model.variables:
        for nm in d.names:
            key = nm.lower()
            if key in seen:
                messages.append({"level": "warn", "code": "warn_duplicate",
                                 "args": {"name": nm}})
                continue
            seen.add(key)
            variables.append({"name": nm, "lower": key, "type": d.type_text,
                              "default_node": d.default, "line": d.line})

    # ---- parameter values (evaluated in declaration order) ------------------
    env: Dict[str, float] = {}
    for p in params:
        val = _eval_const(p["default_node"], env)
        if val is None:
            val = 0.0
            messages.append({"level": "warn", "code": "warn_param_no_default",
                             "args": {"name": p["name"]}})
        env[p["lower"]] = val

    # ---- SET section (fixes values / variables) ------------------------------
    fixed: Set[str] = set()
    for a in model.sets:
        val = _eval_const(a.value, env)
        key = a.name.lower()
        if val is None:
            messages.append({"level": "warn", "code": "warn_set_unresolved",
                             "args": {"name": a.name}})
            val = 0.0
        env[key] = val
        fixed.add(key)

    # ---- schedule ------------------------------------------------------------
    schedule = None
    if model.schedule:
        st = model.schedule[0]
        t0 = _eval_const(st.t0, env)
        t1 = _eval_const(st.t1, env)
        h = _eval_const(st.step, env) if st.step is not None else 1.0
        if t0 is None or t1 is None or h is None or h <= 0:
            messages.append({"level": "error", "code": "err_bad_schedule", "args": {}})
            schedule = {"var": st.var, "t0": 0.0, "t1": 1.0, "step": 1.0}
        else:
            schedule = {"var": st.var, "t0": float(t0), "t1": float(t1),
                        "step": float(h)}

    # ---- equations: display rows, incidence, leaf ids -------------------------
    rows: List[dict] = []
    leaf_counter = [0]
    derivs_used: Set[str] = set()
    funcs_used: Set[str] = set()
    eq_count = 0

    def count_leaves(eq) -> int:
        if isinstance(eq, EqEquation):
            return 1 if eq.op == "=" else 0
        n_t = sum(count_leaves(e) for e in eq.then_eqs)
        n_e = sum(count_leaves(e) for e in eq.else_eqs)
        return max(n_t, n_e)

    def scan_funcs(eq) -> None:
        if isinstance(eq, EqEquation):
            _collect_funcs(eq.lhs, funcs_used)
            _collect_funcs(eq.rhs, funcs_used)
        else:
            _collect_funcs(eq.cond, funcs_used)
            for e in eq.then_eqs + eq.else_eqs:
                scan_funcs(e)

    def union_vars(eq) -> (Set[str], Set[str]):
        if isinstance(eq, EqEquation):
            v, d = set(), set()
            _collect(eq.lhs, v, d)
            _collect(eq.rhs, v, d)
            return v, d
        v, d = set(), set()
        _collect(eq.cond, v, d)
        for e in eq.then_eqs + eq.else_eqs:
            sv, sd = union_vars(e)
            v |= sv
            d |= sd
        return v, d

    def walk_rows(eq, indent: int) -> None:
        nonlocal derivs_used
        if isinstance(eq, EqEquation):
            v, d = set(), set()
            _collect(eq.lhs, v, d)
            _collect(eq.rhs, v, d)
            derivs_used |= d
            eq.leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            rows.append({"id": eq.leaf_id,
                         "kind": "eq" if eq.op == "=" else "ineq",
                         "text": _display_eq(eq),
                         "variables": sorted(v), "derivatives": sorted(d),
                         "indent": indent})
            if eq.op != "=":
                messages.append({"level": "warn", "code": "warn_inequality_skipped",
                                 "args": {"eq": _display_eq(eq)}})
        else:
            v, d = union_vars(eq)
            rows.append({"id": None, "kind": "if",
                         "text": f"IF {expr_str(eq.cond)} THEN … ELSE … END",
                         "variables": sorted(v), "derivatives": sorted(d),
                         "indent": indent})
            for e in eq.then_eqs:
                walk_rows(e, indent + 1)
            for e in eq.else_eqs:
                walk_rows(e, indent + 1)

    for eq in model.equations:
        scan_funcs(eq)
        eq_count += count_leaves(eq)
        walk_rows(eq, 0)

    # ---- unknowns, states, undeclared names -----------------------------------
    unknowns = [v["lower"] for v in variables if v["lower"] not in fixed]
    declared = {p["lower"] for p in params} | {v["lower"] for v in variables}
    if schedule:
        declared.add(schedule["var"].lower())

    all_used: Set[str] = set()
    for row in rows:
        all_used |= set(row["variables"])
    for n in sorted(all_used - declared):
        messages.append({"level": "warn", "code": "warn_undeclared",
                         "args": {"name": n}})

    for dname in sorted(derivs_used - declared):
        messages.append({"level": "error", "code": "err_undeclared_var",
                         "args": {"name": dname}})

    states = [u for u in unknowns if u in derivs_used]

    # ---- INITIAL / guesses ------------------------------------------------------
    guesses: Dict[str, float] = {}
    has_initial: Set[str] = set()
    for a in model.initial:
        v = _eval_const(a.value, env)
        guesses[a.name.lower()] = v if v is not None else 0.0
        has_initial.add(a.name.lower())
    initial_guess: Dict[str, float] = {}
    for v in variables:
        key = v["lower"]
        if key in guesses:
            initial_guess[key] = guesses[key]
        else:
            gv = _eval_const(v["default_node"], env)
            initial_guess[key] = gv if gv is not None else 0.0

    # ---- optimization: objective & bounds -----------------------------------------
    objective = None
    if model.objective:
        objective = {"kind": model.objective.kind,
                     "text": expr_str(model.objective.expr)}
    bounds_info: Dict[str, tuple] = {}
    for b in model.bounds:
        key = b.name.lower()
        lo = _eval_const(b.lo, env)
        hi = _eval_const(b.hi, env)
        if lo is None or hi is None or lo > hi:
            messages.append({"level": "warn", "code": "warn_bounds_invalid",
                             "args": {"name": b.name}})
            continue
        bounds_info[key] = (float(lo), float(hi))

    # ---- degrees of freedom & validation -----------------------------------------
    dof = len(unknowns) - eq_count
    is_dynamic = bool(derivs_used)
    is_optimization = objective is not None
    blocked = False

    if not model.equations:
        messages.append({"level": "error", "code": "err_no_equations", "args": {}})
        blocked = True

    unsupported = sorted(f for f in funcs_used
                         if f not in SUPPORTED_FUNCS and f not in PROP_FUNCS)
    for f in unsupported:
        messages.append({"level": "error", "code": "err_unsupported_func",
                         "args": {"name": f}})
    if unsupported:
        blocked = True

    prop_used = sorted(f for f in funcs_used if f in PROP_FUNCS)
    if prop_used and not COOLPROP_AVAILABLE:
        messages.append({"level": "error", "code": "err_coolprop_missing",
                         "args": {"name": prop_used[0]}})
        blocked = True

    if is_optimization and is_dynamic:
        messages.append({"level": "error", "code": "err_opt_dynamic", "args": {}})
        blocked = True

    if dof < 0:
        messages.append({"level": "error", "code": "err_overdetermined",
                         "args": {"u": len(unknowns), "e": eq_count}})
        blocked = True
    elif dof > 0:
        if is_optimization:
            messages.append({"level": "info", "code": "dof_for_opt",
                             "args": {"u": len(unknowns), "e": eq_count,
                                      "d": dof}})
        else:
            messages.append({"level": "error", "code": "dof_nonzero",
                             "args": {"u": len(unknowns), "e": eq_count,
                                      "d": dof}})
            blocked = True
    else:
        messages.append({"level": "info", "code": "dof_zero",
                         "args": {"u": len(unknowns), "e": eq_count}})
        if is_optimization:
            messages.append({"level": "info", "code": "info_trivial_opt",
                             "args": {}})

    if is_dynamic and not schedule:
        messages.append({"level": "error", "code": "err_dynamic_no_schedule",
                         "args": {}})
        blocked = True
    if is_dynamic and schedule:
        for s in states:
            if s not in has_initial:
                messages.append({"level": "error", "code": "err_state_no_initial",
                                 "args": {"name": s}})
                blocked = True
        steps = int(round((schedule["t1"] - schedule["t0"]) / schedule["step"]))
        if steps > 200000:
            messages.append({"level": "error", "code": "err_too_many_steps",
                             "args": {"n": steps}})
            blocked = True
    if schedule and not is_dynamic:
        messages.append({"level": "info", "code": "info_schedule_ignored",
                         "args": {}})

    messages.append({"level": "info", "code": "info_model_summary",
                     "args": {"name": model.name, "nv": len(variables),
                              "ne": eq_count}})

    # ---- frontend model JSON ------------------------------------------------------
    model_json = {
        "name": model.name,
        "line": model.line,
        "parameters": [{"name": p["name"], "type": p["type"],
                        "default": (expr_str(p["default_node"])
                                    if p["default_node"] is not None else None),
                        "value": env[p["lower"]]} for p in params],
        "variables": [{"name": v["name"], "type": v["type"],
                       "guess": initial_guess[v["lower"]],
                       "fixed": v["lower"] in fixed,
                       "isState": v["lower"] in states} for v in variables],
        "equations": rows,
        "dof": {"unknowns": len(unknowns), "equations": eq_count, "dof": dof},
        "schedule": schedule,
        "isDynamic": is_dynamic,
        "objective": objective,
        "bounds": [{"name": b.name, "lo": bounds_info.get(b.name.lower(), (None, None))[0],
                    "hi": bounds_info.get(b.name.lower(), (None, None))[1]}
                   for b in model.bounds],
    }

    return {
        "messages": messages,
        "model": model_json,
        "blocked": blocked,
        # internal
        "_unknowns": unknowns,
        "_states": states,
        "_fixed_values": env,
        "_initial_guess": initial_guess,
        "_schedule": schedule,
        "_has_initial": has_initial,
        "_dof": dof,
        "_is_optimization": is_optimization,
        "_bounds": bounds_info,
        "_objective": objective,
    }
