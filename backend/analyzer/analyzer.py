"""Structural analysis of fgPROMS models.

Produces:
  * degrees-of-freedom counting,
  * variable/parameter tables with evaluated defaults,
  * equation list with incidence information,
  * detection of dynamic (differential) variables and schedule validation,
  * diagnostic messages (all message codes are resolved client-side).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..parser.gproms_parser import (
    Model, Num, Var, Deriv, UnOp, BinOp, Func, IfExpr, Indexed,
    EqEquation, IfEquation, expr_str,
)
from ..properties import PROP_FUNCS, COOLPROP_AVAILABLE
from ..solver.solver import make_eval, SUPPORTED_FUNCS
from ..distributed import DomainInfo, expand_equations


def _collect(node, vars_out: Set[str], derivs_out: Set[str]) -> None:
    if isinstance(node, Var):
        vars_out.add(node.name.lower())
    elif isinstance(node, Deriv):
        derivs_out.add(node.name.lower())
    elif isinstance(node, Indexed):
        if all(isinstance(i, Num) for i in node.indices):
            vars_out.add(f"{node.name.lower()}"
                         f"({','.join(str(int(i.v)) for i in node.indices)})")
        # symbolic indices are SIGMA loop variables — not model variables
    elif isinstance(node, UnOp):
        _collect(node.x, vars_out, derivs_out)
    elif isinstance(node, BinOp):
        _collect(node.l, vars_out, derivs_out)
        _collect(node.r, vars_out, derivs_out)
    elif isinstance(node, Func):
        if node.name == "SIGMA" and len(node.args) == 4:
            # the index variable is not a model variable — skip it
            _collect(node.args[0], vars_out, derivs_out)
            _collect(node.args[2], vars_out, derivs_out)
            _collect(node.args[3], vars_out, derivs_out)
            return
        for a in node.args:
            _collect(a, vars_out, derivs_out)
    elif isinstance(node, IfExpr):
        _collect(node.cond, vars_out, derivs_out)
        _collect(node.then, vars_out, derivs_out)
        _collect(node.els, vars_out, derivs_out)


def _collect_funcs(node, out: Set[str]) -> None:
    if isinstance(node, Func):
        if node.name == "SIGMA" and len(node.args) == 4:
            _collect_funcs(node.args[0], out)
            _collect_funcs(node.args[2], out)
            _collect_funcs(node.args[3], out)
            return
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
    elif isinstance(node, Indexed):
        for i in node.indices:
            _collect_funcs(i, out)


def _expand_decl_names(name: str, type_text: str) -> List[str]:
    """Expand ARRAY(d1[, d2, ...]) declarations into indexed scalar names:
    z(1,1), z(1,2), ... — row-major, indices from 1."""
    m = re.match(r"ARRAY\s*\(\s*([\d,\s]+)\s*\)", type_text or "",
                 re.IGNORECASE)
    if m:
        dims = [int(x) for x in re.findall(r"\d+", m.group(1))]
        out: List[str] = []
        total = 1
        for d in dims:
            total *= d
        for flat in range(total):
            idxs = []
            rem = flat
            for d in reversed(dims):
                idxs.insert(0, (rem % d) + 1)
                rem //= d
            out.append(f"{name}({','.join(str(i) for i in idxs)})")
        return out
    return [name]


def _array_dims(type_text: str) -> Optional[tuple]:
    m = re.match(r"ARRAY\s*\(\s*([\d,\s]+)\s*\)", type_text or "",
                 re.IGNORECASE)
    if m:
        return tuple(int(x) for x in re.findall(r"\d+", m.group(1)))
    return None


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


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _load_csv_data(fname: str) -> Optional[List[tuple]]:
    """Load (t, value) pairs from data/<fname>. Returns None on failure."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.csv", fname):
        return None
    path = DATA_DIR / fname
    if not path.is_file():
        return None
    pts = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
    except Exception:
        return None
    return pts or None


def analyze(model: Model) -> dict:
    messages: List[dict] = []
    blocked = False

    # ---- declarations ------------------------------------------------------
    params: List[dict] = []
    variables: List[dict] = []
    dist_decls: List[dict] = []
    seen: Set[str] = set()
    array_sizes: Dict[str, int] = {}
    base_map: Dict[str, str] = {}

    for d in model.parameters:
        for nm in d.names:
            dims = _array_dims(d.type_text)
            if dims:
                array_sizes[nm.lower()] = dims
            for exp in _expand_decl_names(nm, d.type_text):
                key = exp.lower()
                if key in seen:
                    messages.append({"level": "warn", "code": "warn_duplicate",
                                     "args": {"name": exp}})
                    continue
                seen.add(key)
                params.append({"name": exp, "lower": key, "base": nm.lower(),
                               "type": d.type_text,
                               "default_node": d.default, "line": d.line})
    for d in model.variables:
        m_dist = re.match(r"DISTRIBUTION\s*\(\s*(\w+)\s*\)", d.type_text or "",
                          re.IGNORECASE)
        for nm in d.names:
            if m_dist:
                dist_decls.append({"name": nm, "lower": nm.lower(),
                                   "type": d.type_text,
                                   "default_node": d.default, "line": d.line,
                                   "domain": m_dist.group(1).lower()})
                continue
            m_arr = _array_dims(d.type_text)
            if m_arr:
                array_sizes[nm.lower()] = m_arr
            for exp in _expand_decl_names(nm, d.type_text):
                key = exp.lower()
                if key in seen:
                    messages.append({"level": "warn", "code": "warn_duplicate",
                                     "args": {"name": exp}})
                    continue
                seen.add(key)
                variables.append({"name": exp, "lower": key,
                                  "base": nm.lower(), "type": d.type_text,
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

    # ---- distribution domains ---------------------------------------------
    domains: Dict[str, DomainInfo] = {}
    if model.distributions:
        for d in model.distributions:
            ext = _eval_const(d.extent, env)
            n = _eval_const(d.points, env)
            key = d.name.lower()
            if (ext is None or n is None or n < 3 or int(n) != n
                    or ext <= 0):
                messages.append({"level": "error",
                                 "code": "err_bad_distribution",
                                 "args": {"name": d.name}})
                blocked = True
                continue
            domains[key] = DomainInfo(d.name, float(ext), int(n),
                                      float(ext) / (int(n) - 1.0))
    if any(re.match(r"DISTRIBUTION", d.type_text or "", re.IGNORECASE)
           for d in model.parameters):
        messages.append({"level": "error", "code": "err_dist_param",
                         "args": {}})
        blocked = True

    # ---- expand distributed variables into point-wise unknowns -------------
    dist_vars: Dict[str, str] = {dv["lower"]: dv["domain"] for dv in dist_decls}
    if dist_decls:
        new_vars: List[dict] = []
        for dv in dist_decls:
            dom = domains.get(dv["domain"])
            if dom is None:
                messages.append({"level": "error", "code": "err_dist_domain",
                                 "args": {"name": dv["name"],
                                          "domain": dv["domain"]}})
                blocked = True
                continue
            array_sizes[dv["lower"]] = (dom.points,)
            for k in range(1, dom.points + 1):
                key = f"{dv['lower']}({k})"
                if key in seen:
                    messages.append({"level": "warn", "code": "warn_duplicate",
                                     "args": {"name": key}})
                    continue
                seen.add(key)
                new_vars.append({"name": f"{dv['name']}({k})", "lower": key,
                                 "base": dv["lower"], "type": dv["type"],
                                 "default_node": dv["default_node"],
                                 "line": dv["line"]})
        variables = variables + new_vars
    for v in variables:
        base_map[v["lower"]] = v["base"]

    # ---- schedule ------------------------------------------------------------
    schedule = None
    phase_intervals: Dict[str, tuple] = {}
    if model.schedule:
        st = model.schedule[0]
        t0 = _eval_const(st.t0, env)
        t1 = _eval_const(st.t1, env)
        h = _eval_const(st.step, env) if st.step is not None else 1.0
        t_last = t1
        st_last = model.schedule[-1]
        tl = _eval_const(st_last.t1, env)
        if tl is not None:
            t_last = tl
        if t0 is None or t1 is None or h is None or h <= 0 or t_last is None:
            messages.append({"level": "error", "code": "err_bad_schedule", "args": {}})
            schedule = {"var": st.var, "t0": 0.0, "t1": 1.0, "step": 1.0}
        else:
            schedule = {"var": st.var, "t0": float(t0), "t1": float(t_last),
                        "step": float(h)}
        for s2 in model.schedule:
            if s2.phase:
                p0 = _eval_const(s2.t0, env)
                p1 = _eval_const(s2.t1, env)
                h2 = _eval_const(s2.step, env) if s2.step is not None else None
                hs = float(h2) if h2 is not None else float(h or 1.0)
                if p0 is not None and p1 is not None:
                    phase_intervals[s2.phase.lower()] = (float(p0), float(p1),
                                                         hs)

    # ---- task phases validation -------------------------------------------
    any_event_phase = False
    if model.task_phases:
        any_event_phase = any(ph.when is not None
                              for ph in model.task_phases.values())
        for p in model.task_phases:
            if p not in phase_intervals:
                messages.append({"level": "warn", "code": "warn_phase_unused",
                                 "args": {"phase": p}})
        for p in phase_intervals:
            if p not in model.task_phases:
                messages.append({"level": "error", "code": "err_phase_unknown",
                                 "args": {"phase": p}})
                blocked = True
        # time-driven phases must cover the horizon without gaps; event
        # phases chain dynamically and only need a correct starting point
        if phase_intervals and schedule is not None:
            ivs = sorted(phase_intervals.values())
            if any_event_phase:
                if ivs and abs(ivs[0][0] - schedule["t0"]) > 1e-9:
                    messages.append({"level": "error", "code": "err_phase_gap",
                                     "args": {}})
                    blocked = True
            else:
                if ivs and abs(ivs[0][0] - schedule["t0"]) > 1e-9:
                    messages.append({"level": "error", "code": "err_phase_gap",
                                     "args": {}})
                    blocked = True
                for a, b in zip(ivs, ivs[1:]):
                    if abs(a[1] - b[0]) > 1e-9:
                        messages.append({"level": "error", "code": "err_phase_gap",
                                         "args": {}})
                        blocked = True
                if ivs and abs(ivs[-1][1] - schedule["t1"]) > 1e-9:
                    messages.append({"level": "error", "code": "err_phase_gap",
                                     "args": {}})
                    blocked = True
        if model.task_phases and (model.distributions or model.boundaries):
            messages.append({"level": "error", "code": "err_phase_dist",
                             "args": {}})
            blocked = True

    # ---- equations: display rows, incidence, leaf ids -------------------------
    # distributed equations expand per interior point (method of lines)
    if model.distributions or model.boundaries or dist_decls:
        try:
            equations = expand_equations(model, domains, dist_vars, env)
        except ValueError as e:
            messages.append({"level": "error", "code": "err_dist_expand",
                             "args": {"detail": str(e)}})
            blocked = True
            equations = model.equations
    else:
        equations = model.equations
    # the solver and everything downstream must see the EXPANDED system
    model.equations = equations
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

    for eq in equations:
        scan_funcs(eq)
        eq_count += count_leaves(eq)
        walk_rows(eq, 0)
    common_count = eq_count
    # phase equations: walked for display/incidence, counted per phase
    for pname, ph in model.task_phases.items():
        for eq in ph.equations:
            scan_funcs(eq)
            walk_rows(eq, 1)
    if model.task_phases:
        first_p = next(iter(model.task_phases))
        eq_count = common_count + sum(count_leaves(e)
                                      for e in model.task_phases[first_p].equations)

    # indexed bases used anywhere in the equations
    indexed_bases: Set[str] = set()

    def scan_indexed(node):
        if isinstance(node, Indexed):
            indexed_bases.add(node.name.lower())
            for i in node.indices:
                scan_indexed(i)
        elif isinstance(node, UnOp):
            scan_indexed(node.x)
        elif isinstance(node, BinOp):
            scan_indexed(node.l)
            scan_indexed(node.r)
        elif isinstance(node, Func):
            for a in node.args:
                scan_indexed(a)
        elif isinstance(node, IfExpr):
            scan_indexed(node.cond)
            scan_indexed(node.then)
            scan_indexed(node.els)

    def scan_eq_indexed(eq):
        if isinstance(eq, EqEquation):
            scan_indexed(eq.lhs)
            scan_indexed(eq.rhs)
        else:
            scan_indexed(eq.cond)
            for e in eq.then_eqs + eq.else_eqs:
                scan_eq_indexed(e)

    for eq in equations:
        scan_eq_indexed(eq)
    for pname, ph in model.task_phases.items():
        for eq in ph.equations:
            scan_eq_indexed(eq)

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
        elif v["base"] in guesses:
            initial_guess[key] = guesses[v["base"]]
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

    # ---- parameter estimation (DATA / ESTIMATE) -----------------------------
    is_estimation = bool(model.estimate)
    estimate_names: List[str] = []
    param_lowers = {p["lower"] for p in params}
    if model.estimate:
        for nm in model.estimate:
            key = nm.lower()
            if key not in param_lowers or "(" in key:
                messages.append({"level": "error", "code": "err_est_not_param",
                                 "args": {"name": nm}})
                blocked = True
                continue
            estimate_names.append(key)
    data_info: List[tuple] = []
    if model.data:
        var_keys = ({v["lower"] for v in variables}
                    | {v["base"] for v in variables})
        for d in model.data:
            key = d.name.lower()
            if key not in var_keys:
                messages.append({"level": "error", "code": "err_data_not_var",
                                 "args": {"name": d.name}})
                blocked = True
                continue
            pts = []
            if d.file:
                loaded = _load_csv_data(d.file)
                if loaded is None:
                    messages.append({"level": "error", "code": "err_data_file",
                                     "args": {"name": d.file}})
                    blocked = True
                    continue
                pts = loaded
            else:
                for t_node, v_node in d.points:
                    tv = _eval_const(t_node, env)
                    vv = _eval_const(v_node, env)
                    if tv is None or vv is None:
                        messages.append({"level": "error", "code": "err_data_bad",
                                         "args": {"name": d.name}})
                        blocked = True
                        continue
                    pts.append((float(tv), float(vv)))
            data_info.append((key, pts))
        if not model.estimate:
            messages.append({"level": "info", "code": "info_data_ignored",
                             "args": {}})
    if model.estimate and not model.data:
        messages.append({"level": "error", "code": "err_est_no_data",
                         "args": {}})
        blocked = True
    if is_estimation and objective is not None:
        messages.append({"level": "error", "code": "err_est_opt", "args": {}})
        blocked = True

    # ---- degrees of freedom & validation -----------------------------------------
    if model.task_phases:
        for pname, ph in model.task_phases.items():
            pc = sum(count_leaves(e) for e in ph.equations)
            if common_count + pc != len(unknowns):
                messages.append({"level": "error", "code": "err_phase_dof",
                                 "args": {"phase": pname, "u": len(unknowns),
                                          "e": common_count + pc}})
                blocked = True
    dof = len(unknowns) - eq_count
    is_dynamic = bool(derivs_used)
    is_optimization = objective is not None

    if not equations and not model.task_phases:
        messages.append({"level": "error", "code": "err_no_equations", "args": {}})
        blocked = True

    unsupported = sorted(f for f in funcs_used
                         if f not in SUPPORTED_FUNCS and f not in PROP_FUNCS
                         and f != "INT")
    for f in unsupported:
        messages.append({"level": "error", "code": "err_unsupported_func",
                         "args": {"name": f}})
    if unsupported:
        blocked = True
    if "INT" in funcs_used:
        messages.append({"level": "error", "code": "err_int_domain",
                         "args": {}})
        blocked = True

    # indexed access on a name that is not a declared ARRAY
    for b in sorted(indexed_bases - set(array_sizes)):
        messages.append({"level": "error", "code": "err_not_array",
                         "args": {"name": b}})
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
            if s not in has_initial and base_map.get(s, s) not in has_initial:
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
        "distributed": [{"name": dv["name"], "domain": dv["domain"],
                         "points": domains[dv["domain"]].points,
                         "extent": domains[dv["domain"]].extent}
                        for dv in dist_decls if dv["domain"] in domains],
        "estimated": [p["name"] for p in params if p["lower"] in estimate_names],
        "data": [{"var": d.name,
                  "points": [{"t": pt[0], "value": pt[1]}
                             for pt in data_info[i][1]]}
                 for i, d in enumerate(model.data)
                 if data_info and i < len(data_info)],
        "phases": [{"name": p, "start": phase_intervals[p][0],
                    "end": phase_intervals[p][1],
                    "event": (model.task_phases[p].when is not None
                              if p in model.task_phases else False)}
                   for p in sorted(phase_intervals)],
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
        "_array_sizes": array_sizes,
        "_is_estimation": is_estimation,
        "_estimate": estimate_names,
        "_data": data_info,
        "_phases": phase_intervals,
    }
