"""Flowsheet support: PROCESS definitions composed of MODEL units.

Design (simplified gPROMS semantics):
  * A PROCESS instantiates UNITs (`R1 AS CSTRUnit`) of previously defined
    MODELs and connects them with equations using dotted names (`R2.CAin =
    R1.CA`).
  * Unit parameters become namespaced fixed values (`r1.v`); unit variables
    become namespaced unknowns (`r1.ca`).
  * The combined system (rewritten unit equations + connection equations)
    is solved with the standard steady-state machinery.
  * Dynamic units (derivatives / SCHEDULE) are rejected for now.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .parser.gproms_parser import (
    Model, ProcessDef, Num, Str, Var, Deriv, UnOp, BinOp, Func, IfExpr, Indexed,
    EqEquation, IfEquation, expr_str,
)
from .properties import COOLPROP_AVAILABLE
from .solver.solver import make_eval

import re  # noqa: E402


def _expand_names(name: str, type_text: str):
    m = re.match(r"ARRAY\s*\(\s*(\d+)\s*\)", type_text or "", re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return [f"{name}({k})" for k in range(1, n + 1)]
    return [name]

# --------------------------------------------------------------------------
# AST rewriting
# --------------------------------------------------------------------------
def _eval(node, env: Dict[str, float]) -> Optional[float]:
    if node is None:
        return None
    if isinstance(node, Num):
        return float(node.v)
    try:
        return float(make_eval(node)(env))
    except Exception:
        return None


def _rewrite(node, prefix: str):
    if isinstance(node, (Num, Str)):
        return node
    if isinstance(node, Var):
        return Var(prefix + node.name)
    if isinstance(node, Deriv):
        return Deriv(prefix + node.name)
    if isinstance(node, UnOp):
        return UnOp(node.op, _rewrite(node.x, prefix))
    if isinstance(node, BinOp):
        return BinOp(node.op, _rewrite(node.l, prefix), _rewrite(node.r, prefix))
    if isinstance(node, Func):
        return Func(node.name, [_rewrite(a, prefix) for a in node.args])
    if isinstance(node, Indexed):
        return Indexed(prefix + node.name,
                       [_rewrite(i, prefix) for i in node.indices])
    if isinstance(node, IfExpr):
        return IfExpr(_rewrite(node.cond, prefix), _rewrite(node.then, prefix),
                      _rewrite(node.els, prefix))
    raise ValueError(f"cannot rewrite {type(node).__name__}")


def _rewrite_eq(eq, prefix: str):
    if isinstance(eq, EqEquation):
        return EqEquation(_rewrite(eq.lhs, prefix), _rewrite(eq.rhs, prefix),
                          eq.op)
    return IfEquation(_rewrite(eq.cond, prefix),
                      [_rewrite_eq(e, prefix) for e in eq.then_eqs],
                      [_rewrite_eq(e, prefix) for e in eq.else_eqs])


def _has_derivatives(model: Model) -> bool:
    def scan_eq(eq) -> bool:
        if isinstance(eq, EqEquation):
            return _scan(eq.lhs) or _scan(eq.rhs)
        return _scan(eq.cond) or any(scan_eq(e) for e in eq.then_eqs + eq.else_eqs)

    def _scan(node) -> bool:
        if isinstance(node, Deriv):
            return True
        if isinstance(node, (UnOp,)):
            return _scan(node.x)
        if isinstance(node, BinOp):
            return _scan(node.l) or _scan(node.r)
        if isinstance(node, Func):
            return any(_scan(a) for a in node.args)
        if isinstance(node, IfExpr):
            return _scan(node.cond) or _scan(node.then) or _scan(node.els)
        return False

    return any(scan_eq(e) for e in model.equations)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyze_process(process: ProcessDef, models: Dict[str, Model]) -> dict:
    messages: List[dict] = []
    blocked = False

    # ---- validate unit types -----------------------------------------------
    units_ok: List[Tuple[str, str, Model, object]] = []   # (lower, display, model, UnitDecl)
    seen_units: Set[str] = set()
    any_dynamic = False
    for u in process.units:
        key = u.name.lower()
        if key in seen_units:
            messages.append({"level": "error", "code": "err_duplicate_unit",
                             "args": {"name": u.name}})
            blocked = True
            continue
        seen_units.add(key)
        m = models.get(u.type_name.lower())
        if m is None:
            messages.append({"level": "error", "code": "err_unknown_unit",
                             "args": {"name": u.name, "type": u.type_name}})
            blocked = True
            continue
        if m.schedule or _has_derivatives(m):
            any_dynamic = True
        units_ok.append((key, u.name, m, u))

    # ---- schedule ----------------------------------------------------------
    schedule = None
    if process.schedule:
        st = process.schedule[0]
        t0 = _eval(st.t0, {})
        t1 = _eval(st.t1, {})
        h = _eval(st.step, {}) if st.step is not None else 1.0
        if t0 is None or t1 is None or h is None or h <= 0:
            messages.append({"level": "error", "code": "err_bad_schedule",
                             "args": {}})
            blocked = True
        else:
            schedule = {"var": st.var, "t0": float(t0), "t1": float(t1),
                        "step": float(h)}
    is_dynamic = any_dynamic
    if is_dynamic and schedule is None:
        messages.append({"level": "error", "code": "err_flowsheet_no_schedule",
                         "args": {}})
        blocked = True

    if blocked:
        return {"messages": messages, "model": None, "blocked": True}

    # ---- per-unit parameter environments (unit-local) -------------------------
    unit_envs: Dict[str, Dict[str, float]] = {}
    unit_params: Dict[str, Set[str]] = {}
    unit_var_names: Dict[str, Set[str]] = {}
    unit_var_display: Dict[str, Dict[str, str]] = {}

    for key, disp, m, u in units_ok:
        env: Dict[str, float] = {}
        param_names: Set[str] = set()
        for d in m.parameters:
            for nm in d.names:
                k = nm.lower()
                param_names.add(k)
                val = _eval(d.default, env)
                env[k] = val if val is not None else 0.0
        for a in m.sets:
            v = _eval(a.value, env)
            env[a.name.lower()] = v if v is not None else 0.0
        for a in u.sets:  # instance-level SET overrides
            v = _eval(a.value, env)
            env[a.name.lower()] = v if v is not None else 0.0
        unit_envs[key] = env
        unit_params[key] = param_names
        var_names = set()
        var_disp: Dict[str, str] = {}
        for d in m.variables:
            for nm in d.names:
                for exp in _expand_names(nm, d.type_text):
                    var_names.add(exp.lower())
                    var_disp[exp.lower()] = exp
        unit_var_names[key] = var_names
        unit_var_display[key] = var_disp

    # ---- namespaced fixed values / unknowns -------------------------------------
    fixed_values: Dict[str, float] = {}
    display: Dict[str, str] = {}
    for key, disp, m, u in units_ok:
        for k, v in unit_envs[key].items():
            full = f"{key}.{k}"
            fixed_values[full] = v
            display[full] = f"{disp}.{unit_var_display[key].get(k, k)}"

    fixed: Set[str] = set()
    for a in process.sets:
        name = a.name.lower()
        if "." not in name:
            messages.append({"level": "warn", "code": "warn_process_set_ignored",
                             "args": {"name": a.name}})
            continue
        ukey, rest = name.split(".", 1)
        full = f"{ukey}.{rest}"
        if ukey in unit_envs and rest in unit_envs[ukey]:
            v = _eval(a.value, fixed_values)
            fixed_values[full] = v if v is not None else 0.0
        elif ukey in unit_var_names and rest in unit_var_names[ukey]:
            v = _eval(a.value, fixed_values)
            fixed_values[full] = v if v is not None else 0.0
            fixed.add(full)
        else:
            messages.append({"level": "warn", "code": "warn_process_set_ignored",
                             "args": {"name": a.name}})

    # ---- unknowns, guesses, rewritten equations ----------------------------------
    unknowns: List[str] = []
    guesses: Dict[str, float] = {}
    combined: List = []
    has_initial: Set[str] = set()

    for key, disp, m, u in units_ok:
        prefix = disp + "."
        for d in m.variables:
            for nm in d.names:
                for exp in _expand_names(nm, d.type_text):
                    k = exp.lower()
                    full = f"{key}.{k}"
                    display[full] = prefix + exp
                    if full not in fixed:
                        unknowns.append(full)
                    g = None
                    for a2 in m.initial:
                        if a2.name.lower() == k:
                            g = _eval(a2.value, unit_envs[key])
                            has_initial.add(full)
                    if g is None:
                        g = _eval(d.default, unit_envs[key])
                    guesses[full] = g if g is not None else 0.0
        for eq in m.equations:
            combined.append(_rewrite_eq(eq, prefix))

    # ---- material streams: variables + implicit property equations ---------
    stream_info = []
    if process.streams and not COOLPROP_AVAILABLE:
        messages.append({"level": "error", "code": "err_coolprop_missing",
                         "args": {"name": "H"}})
        blocked = True
    for st in process.streams:
        key = st.name.lower()
        stream_info.append({"name": st.name, "fluid": st.fluid or "Water"})
        for prop, g in (("F", 1.0), ("T", 300.0), ("P", 1.0E5)):
            full = f"{key}.{prop.lower()}"
            display[full] = f"{st.name}.{prop}"
            unknowns.append(full)
            guesses[full] = g
        fluid = st.fluid or "Water"
        for prop, fn in (("H", "H"), ("CP", "CP"), ("RHO", "RHO")):
            full = f"{key}.{prop.lower()}"
            display[full] = f"{st.name}.{prop}"
            unknowns.append(full)
            guesses[full] = 0.0
            lhs = Var(f"{st.name}.{prop}")
            rhs = Func(fn, [Str(fluid), Var(f"{st.name}.T"),
                            Var(f"{st.name}.P")])
            combined.append(EqEquation(lhs, rhs, "="))

    for eq in process.equations:
        combined.append(eq)

    # ---- process-level INITIAL (dotted names, e.g. T1.h = 2.0) -----------------
    for a in process.initial:
        key = a.name.lower()
        if "." in key:
            v = _eval(a.value, fixed_values)
            guesses[key] = v if v is not None else 0.0
            has_initial.add(key)
        else:
            messages.append({"level": "warn", "code": "warn_process_set_ignored",
                             "args": {"name": a.name}})

    # ---- incidence rows & undeclared names ----------------------------------------
    rows: List[dict] = []
    leaf_counter = [0]
    derivs_used: Set[str] = set()

    def collect(node, vars_out, derivs_out):
        if isinstance(node, Var):
            vars_out.add(node.name.lower())
        elif isinstance(node, Deriv):
            derivs_out.add(node.name.lower())
        elif isinstance(node, Indexed):
            if all(isinstance(i, Num) for i in node.indices):
                vars_out.add(f"{node.name.lower()}"
                             f"({','.join(str(int(i.v)) for i in node.indices)})")
        elif isinstance(node, UnOp):
            collect(node.x, vars_out, derivs_out)
        elif isinstance(node, BinOp):
            collect(node.l, vars_out, derivs_out)
            collect(node.r, vars_out, derivs_out)
        elif isinstance(node, Func):
            if node.name == "SIGMA" and len(node.args) == 4:
                collect(node.args[0], vars_out, derivs_out)
                collect(node.args[2], vars_out, derivs_out)
                collect(node.args[3], vars_out, derivs_out)
                return
            for a in node.args:
                collect(a, vars_out, derivs_out)
        elif isinstance(node, IfExpr):
            collect(node.cond, vars_out, derivs_out)
            collect(node.then, vars_out, derivs_out)
            collect(node.els, vars_out, derivs_out)

    def walk_rows(eq, indent):
        nonlocal derivs_used
        if isinstance(eq, EqEquation):
            v, d = set(), set()
            collect(eq.lhs, v, d)
            collect(eq.rhs, v, d)
            derivs_used |= d
            eq.leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            text = f"{expr_str(eq.lhs)} = {expr_str(eq.rhs)}" if eq.op == "=" \
                else f"{expr_str(eq.lhs)} {eq.op} {expr_str(eq.rhs)}"
            rows.append({"id": eq.leaf_id, "kind": "eq" if eq.op == "=" else "ineq",
                         "text": text, "variables": sorted(v),
                         "derivatives": sorted(d), "indent": indent})
        else:
            v, d = set(), set()
            collect(eq.cond, v, d)
            for e in eq.then_eqs + eq.else_eqs:
                sv, sd = set(), set()
                def _u(x, o1, o2):
                    if isinstance(x, EqEquation):
                        collect(x.lhs, o1, o2)
                        collect(x.rhs, o1, o2)
                    else:
                        collect(x.cond, o1, o2)
                        for s in x.then_eqs + x.else_eqs:
                            _u(s, o1, o2)
                _u(e, sv, sd)
                v |= sv
                d |= sd
            rows.append({"id": None, "kind": "if",
                         "text": f"IF {expr_str(eq.cond)} THEN … ELSE … END",
                         "variables": sorted(v), "derivatives": sorted(d),
                         "indent": indent})
            for e in eq.then_eqs:
                walk_rows(e, indent + 1)
            for e in eq.else_eqs:
                walk_rows(e, indent + 1)

    def count_leaves(eq) -> int:
        if isinstance(eq, EqEquation):
            return 1 if eq.op == "=" else 0
        n_t = sum(count_leaves(e) for e in eq.then_eqs)
        n_e = sum(count_leaves(e) for e in eq.else_eqs)
        return max(n_t, n_e)

    for eq in combined:
        walk_rows(eq, 0)
    eq_count = sum(count_leaves(e) for e in combined)

    # ---- dynamic states ---------------------------------------------------------
    states = [u for u in unknowns if u in derivs_used]
    if is_dynamic:
        for s in states:
            if s not in has_initial:
                messages.append({"level": "error", "code": "err_state_no_initial",
                                 "args": {"name": display.get(s, s)}})
                blocked = True

    declared = set(unknowns) | set(fixed_values) | set(fixed)
    all_used: Set[str] = set()
    for row in rows:
        all_used |= set(row["variables"])
    for n in sorted(all_used - declared):
        messages.append({"level": "warn", "code": "warn_undeclared",
                         "args": {"name": n}})

    # ---- degrees of freedom ---------------------------------------------------------
    dof = len(unknowns) - eq_count
    if dof != 0:
        messages.append({"level": "error", "code": "dof_nonzero",
                         "args": {"u": len(unknowns), "e": eq_count, "d": dof}})
        blocked = True
    else:
        messages.append({"level": "info", "code": "dof_zero",
                         "args": {"u": len(unknowns), "e": eq_count}})

    messages.append({"level": "info", "code": "info_flowsheet_summary",
                     "args": {"name": process.name, "nu": len(units_ok),
                              "nv": len(unknowns), "ne": eq_count}})

    # ---- model JSON & solver payload ---------------------------------------------
    params_json = []
    for full, v in sorted(fixed_values.items()):
        params_json.append({"name": display.get(full, full), "type": "",
                            "default": None, "value": v})
    vars_json = []
    for full in unknowns:
        vars_json.append({"name": display.get(full, full), "type": "",
                          "guess": guesses[full], "fixed": False,
                          "isState": full in states})
    for full in sorted(fixed):
        vars_json.append({"name": display.get(full, full), "type": "",
                          "guess": fixed_values[full], "fixed": True,
                          "isState": False})

    model_json = {
        "name": process.name,
        "line": process.line,
        "isProcess": True,
        "units": [{"name": disp, "type": m.name} for _, disp, m, _ in units_ok],
        "streams": stream_info,
        "parameters": params_json,
        "variables": vars_json,
        "equations": rows,
        "dof": {"unknowns": len(unknowns), "equations": eq_count, "dof": dof},
        "schedule": schedule,
        "isDynamic": is_dynamic,
        "objective": None,
        "bounds": [],
    }

    synth = Model(name=process.name, line=process.line, equations=combined)
    pseudo_info = {
        "_unknowns": unknowns,
        "_fixed_values": fixed_values,
        "_initial_guess": guesses,
        "_states": states,
        "_schedule": schedule,
    }
    return {"messages": messages, "model": model_json, "blocked": blocked,
            "_synth": synth, "_info": pseudo_info}
