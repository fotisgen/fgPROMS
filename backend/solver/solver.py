"""Numerical solvers for fgPROMS models (NumPy/SciPy based).

* Steady-state:  Newton-type root finding on the full algebraic system.
* Dynamic:       implicit Euler (first-order BDF) on the DAE system,
                 with consistent algebraic initialisation at t0 —
                 the same conceptual pipeline gPROMS uses internally.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import root, minimize

from ..parser.gproms_parser import (
    Num, Str, Var, Deriv, UnOp, BinOp, Func, IfExpr,
    EqEquation, IfEquation, Model,
)
from ..properties import PROP_FUNCS, make_prop_call
from .jacobian import (build_jacobian, build_gradient, jacobian_is_available,
                       compile_jacobian)

# --------------------------------------------------------------------------
# Safe math wrappers (keep Newton iterations inside the domain)
# --------------------------------------------------------------------------
def safe_ln(x: float) -> float:
    if x > 1e-10:
        return math.log(x)
    return math.log(1e-10) + (x - 1e-10) / 1e-10


def safe_log10(x: float) -> float:
    if x > 1e-10:
        return math.log10(x)
    return math.log10(1e-10) + (x - 1e-10) / 1e-10


def safe_sqrt(x: float) -> float:
    return math.sqrt(x) if x > 0.0 else 0.0


def safe_div(a: float, b: float) -> float:
    if b == 0.0 or abs(b) < 1e-300:
        return a / 1e-12
    return a / b


def safe_pow(a: float, b: float) -> float:
    if a > 0:
        try:
            p = b * math.log(a)
            return math.exp(p) if p < 700 else 1e300
        except (OverflowError, ValueError):
            return 1e300
    if b == round(b):
        return a ** int(round(b))
    return 0.0


def safe_exp(x: float) -> float:
    return math.exp(max(-745.0, min(x, 700.0)))


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


SUPPORTED_FUNCS = {"EXP", "LN", "LOG", "SQRT", "ABS", "SIN", "COS", "TAN",
                   "SIGN", "MAX", "MIN"}

_UNARY_FUNCS = {
    "EXP": safe_exp, "LN": safe_ln, "LOG": safe_log10, "SQRT": safe_sqrt,
    "ABS": abs, "SIN": math.sin, "COS": math.cos, "TAN": math.tan,
    "SIGN": _sign,
}

# --------------------------------------------------------------------------
# AST -> Python evaluator
# --------------------------------------------------------------------------
def make_eval(node) -> object:
    """Compile an AST node into a callable f(env) -> float.

    Note: no global caching — node identities are not stable across parses,
    so each compilation is fresh and scoped to its model.
    """
    if isinstance(node, Num):
        v = float(node.v)
        fn = lambda env: v
    elif isinstance(node, Var):
        name = node.name.lower()
        fn = lambda env, _n=name: env.get(_n, 0.0)
    elif isinstance(node, Deriv):
        name = "$" + node.name.lower()
        fn = lambda env, _n=name: env.get(_n, 0.0)
    elif isinstance(node, UnOp):
        f = make_eval(node.x)
        if node.op == "-":
            fn = lambda env, _f=f: -_f(env)
        else:  # NOT
            fn = lambda env, _f=f: 1.0 if _f(env) == 0.0 else 0.0
    elif isinstance(node, BinOp):
        l = make_eval(node.l)
        r = make_eval(node.r)
        op = node.op
        if op == "+":
            fn = lambda env, _l=l, _r=r: _l(env) + _r(env)
        elif op == "-":
            fn = lambda env, _l=l, _r=r: _l(env) - _r(env)
        elif op == "*":
            fn = lambda env, _l=l, _r=r: _l(env) * _r(env)
        elif op == "/":
            fn = lambda env, _l=l, _r=r: safe_div(_l(env), _r(env))
        elif op in ("^", "**"):
            fn = lambda env, _l=l, _r=r: safe_pow(_l(env), _r(env))
        elif op == "=":
            fn = lambda env, _l=l, _r=r: _l(env) - _r(env)
        elif op == "<>":
            fn = lambda env, _l=l, _r=r: 1.0 if _l(env) != _r(env) else 0.0
        elif op == "<":
            fn = lambda env, _l=l, _r=r: 1.0 if _l(env) < _r(env) else 0.0
        elif op == "<=":
            fn = lambda env, _l=l, _r=r: 1.0 if _l(env) <= _r(env) else 0.0
        elif op == ">":
            fn = lambda env, _l=l, _r=r: 1.0 if _l(env) > _r(env) else 0.0
        elif op == ">=":
            fn = lambda env, _l=l, _r=r: 1.0 if _l(env) >= _r(env) else 0.0
        elif op == "AND":
            fn = lambda env, _l=l, _r=r: 1.0 if (_l(env) != 0.0 and _r(env) != 0.0) else 0.0
        elif op == "OR":
            fn = lambda env, _l=l, _r=r: 1.0 if (_l(env) != 0.0 or _r(env) != 0.0) else 0.0
        else:
            fn = lambda env: 0.0
    elif isinstance(node, Func):
        if node.name in PROP_FUNCS and node.args and isinstance(node.args[0], Str):
            fluid = node.args[0].value
            arg_fns = [make_eval(a) for a in node.args[1:]]
            fn = make_prop_call(node.name, fluid, arg_fns)
        elif node.name in _UNARY_FUNCS and len(node.args) == 1:
            f = make_eval(node.args[0])
            u = _UNARY_FUNCS[node.name]
            fn = lambda env, _f=f, _u=u: _u(_f(env))
        elif node.name in ("MAX", "MIN"):
            fs = [make_eval(a) for a in node.args]
            u = max if node.name == "MAX" else min
            fn = lambda env, _fs=fs, _u=u: _u(f(env) for f in _fs)
        else:
            fn = lambda env: 0.0
    elif isinstance(node, IfExpr):
        c = make_eval(node.cond)
        a = make_eval(node.then)
        b = make_eval(node.els)
        fn = lambda env, _c=c, _a=a, _b=b: _a(env) if _c(env) != 0.0 else _b(env)
    else:
        fn = lambda env: 0.0

    return fn


# --------------------------------------------------------------------------
# Model compilation: build all residual closures once per model
# --------------------------------------------------------------------------
def compile_model(model: Model) -> list:
    """Compile the equation list into a flat structure of closures.
    Each entry is one of:
      ('eq',  lhs_fn, rhs_fn, leaf_id, lhs_node, rhs_node)
      ('ineq', leaf_id)                       # skipped by the solver
      ('if',  cond_fn, then_list, else_list, cond_node)
    The AST nodes are kept so the SymPy Jacobian builder can translate
    the equations symbolically.
    """
    def c(eq):
        if isinstance(eq, EqEquation):
            if eq.op == "=":
                return ("eq", make_eval(eq.lhs), make_eval(eq.rhs), eq.leaf_id,
                        eq.lhs, eq.rhs)
            return ("ineq", eq.leaf_id)
        cond = make_eval(eq.cond)
        t = [c(e) for e in eq.then_eqs]
        e = [c(e) for e in eq.else_eqs]
        return ("if", cond, t, e, eq.cond)

    return [c(e) for e in model.equations]


# --------------------------------------------------------------------------
# Residual evaluation
# --------------------------------------------------------------------------
def residuals(compiled: list, env: Dict[str, float]) -> Tuple[np.ndarray, List[int]]:
    """Evaluate residuals of a compiled (possibly conditional) equation list.

    Returns (values, leaf_ids). leaf_id refers to the id assigned by the
    analyzer, so the frontend can match residuals to displayed equations.
    """
    out: List[float] = []
    ids: List[int] = []

    def walk(eq) -> None:
        if eq[0] == "eq":
            out.append(eq[1](env) - eq[2](env))
            ids.append(eq[3])
        elif eq[0] == "if":
            branch = eq[2] if eq[1](env) != 0.0 else eq[3]
            for e in branch:
                walk(e)

    for e in compiled:
        walk(e)
    return np.array(out, dtype=float), ids


# --------------------------------------------------------------------------
# Damped Newton with analytic Jacobian
# --------------------------------------------------------------------------
def _newton(fun, jac_fn, x0, maxiter: int = 200, tol: float = 1e-10):
    """Damped Newton with backtracking line search on max|f|.
    Returns (x, success, iterations)."""
    x = np.array(x0, dtype=float)
    f = np.asarray(fun(x), dtype=float)
    for it in range(maxiter):
        norm = float(np.max(np.abs(f)))
        if norm < tol:
            return x, True, it
        J = jac_fn(x)
        if J is None:
            return x, False, it
        J = np.asarray(J, dtype=float)
        if J.shape != (len(x), len(x)) or not np.all(np.isfinite(J)):
            return x, False, it
        # row equilibration for better conditioning (the safe-domain
        # extensions produce very large derivative entries)
        D = np.maximum(np.max(np.abs(J), axis=1), 1e-300)
        try:
            dx = np.linalg.solve(J / D[:, None], -f / D)
        except np.linalg.LinAlgError:
            return x, False, it
        if not np.all(np.isfinite(dx)):
            return x, False, it
        accepted = False
        alpha = 1.0
        for _ in range(40):
            xn = x + alpha * dx
            fn = np.asarray(fun(xn), dtype=float)
            if np.max(np.abs(fn)) < norm:
                x, f = xn, fn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return x, False, it
    return x, bool(np.max(np.abs(f)) < tol), maxiter


# --------------------------------------------------------------------------
# Generic root solving
# --------------------------------------------------------------------------
def _root_solve(names: List[str], x0_map: Dict[str, float], base: Dict[str, float],
                compiled: list, time_var: Optional[str] = None,
                t: float = 0.0, _disable_newton: bool = False
                ) -> Tuple[bool, Optional[np.ndarray], str, List[Tuple[int, float]]]:
    """Solve the system for the unknowns in `names`. Returns
    (success, solution, method, residual_id_pairs)."""
    x0 = np.array([x0_map.get(n, 0.0) for n in names], dtype=float)

    def fun(x):
        env = dict(base)
        if time_var:
            env[time_var] = t
        for n, v in zip(names, x):
            env[n] = float(v)
        try:
            r, _ = residuals(compiled, env)
            return np.where(np.isfinite(r), r, 1e10)
        except Exception:
            return np.full(len(names), 1e10)

    # quick square-system check
    try:
        r0, _ = residuals(compiled, {**base, **{n: x0_map.get(n, 0.0) for n in names}})
        if len(r0) != len(names):
            return False, None, "dof_mismatch", []
    except Exception:
        pass

    # ---- attempt analytic Newton (SymPy Jacobian) first -------------------
    if not _disable_newton:
        try:
            jac = build_jacobian(compiled, names, base)
            if jac is not None and jacobian_is_available(jac, x0, names, base):
                x_newt, ok_newt, _ = _newton(fun, jac, x0)
                if ok_newt:
                    env = dict(base)
                    if time_var:
                        env[time_var] = t
                    for n, v in zip(names, x_newt):
                        env[n] = float(v)
                    r, ids = residuals(compiled, env)
                    pairs = [(i, float(v)) for i, v in zip(ids, r)]
                    return True, x_newt, "NEWTON", pairs
        except Exception:
            pass

    sol = None
    method = ""
    for m in ("hybr", "lm"):
        try:
            opts = {"maxfev": 4000} if m == "hybr" else {"maxiter": 4000}
            cand = root(fun, x0, method=m, options=opts)
            if cand.success:
                # verify the residual actually went to zero (guards against
                # "successful" convergence to a flat NaN-penalised function)
                env_c = dict(base)
                if time_var:
                    env_c[time_var] = t
                for n, v in zip(names, cand.x):
                    env_c[n] = float(v)
                rc, _ = residuals(compiled, env_c)
                if np.all(np.isfinite(rc)) and np.max(np.abs(rc)) < 1e-6:
                    sol = cand
                    method = m
                    break
        except Exception:
            continue

    if sol is None:
        try:
            r, _ = residuals(compiled, {**base, **{n: x0_map.get(n, 0.0) for n in names}})
            detail = f"max residual {float(np.max(np.abs(r))):.3e}"
        except Exception:
            detail = "no convergence"
        return False, None, detail, []

    env = dict(base)
    if time_var:
        env[time_var] = t
    for n, v in zip(names, sol.x):
        env[n] = float(v)
    r, ids = residuals(compiled, env)
    pairs = [(i, float(v)) for i, v in zip(ids, r)]
    return True, np.array(sol.x, dtype=float), method, pairs


# --------------------------------------------------------------------------
# Steady-state solve
# --------------------------------------------------------------------------
def solve_steady(model: Model, info: dict) -> dict:
    names: List[str] = info["_unknowns"]
    base: Dict[str, float] = dict(info["_fixed_values"])
    guesses: Dict[str, float] = info["_initial_guess"]
    compiled = compile_model(model)

    ok, sol, method, pairs = _root_solve(names, guesses, base, compiled)

    if not ok:
        if method == "dof_mismatch":
            r0, _ = residuals(compiled, {**base, **guesses})
            msg = {"level": "error", "code": "err_dof_mismatch",
                   "args": {"ne": len(r0), "nu": len(names)}}
        else:
            msg = {"level": "error", "code": "solve_failed", "args": {"detail": method}}
        return {"success": False, "messages": [msg], "values": None, "residuals": [],
                "maxResidual": None, "method": None}

    values = {n: float(v) for n, v in zip(names, sol)}
    maxr = max((abs(v) for _, v in pairs), default=0.0)
    msg = {"level": "info", "code": "solved_ok",
           "args": {"method": method.upper(), "n": len(names), "res": f"{maxr:.2e}"}}
    return {"success": True, "messages": [msg], "values": values,
            "residuals": [{"id": i, "value": v} for i, v in pairs],
            "maxResidual": maxr, "method": method.upper()}


# --------------------------------------------------------------------------
# Dynamic solve — BDF2 with adaptive step control.
#
#   * First step: implicit Euler (BDF1) — no previous point exists yet.
#   * Later steps: variable-step BDF2, whose derivative approximation is
#         x' ~ c1*(x - x_n) + c2*(x_n - x_{n-1}),  r = h/h_prev
#         c1 = (2r+1)/(h(r+1)),  c2 = -r^2/(h(r+1))
#   * Local error estimate: embedded BDF1 solution at the same step
#     (difference estimator) — conservative and simple.
#   * Step control: err_norm = max_i |x2-x1| / (atol + rtol*|x2|)
#     h_new = 0.9 * h * min(2, err^-1/3); the SCHEDULE STEP acts as the
#     maximum (and initial) step, so output has at least (t1-t0)/STEP points.
#   * fixed_h mode (tests): constant-step BDF2, no step control.
# --------------------------------------------------------------------------
def solve_dynamic(model: Model, info: dict, fixed_h: Optional[float] = None) -> dict:
    sch = info["_schedule"]
    t0, t1, h0 = sch["t0"], sch["t1"], sch["step"]
    tv = sch["var"].lower()
    states: List[str] = info["_states"]
    allv: List[str] = info["_unknowns"]
    alg = [n for n in allv if n not in states]
    base: Dict[str, float] = dict(info["_fixed_values"])
    compiled = compile_model(model)
    messages: List[dict] = []
    idx = {n: i for i, n in enumerate(allv)}
    nv = len(allv)

    # pre-compile the analytic Jacobian ONCE — the structure is
    # step-independent, only the BDF coefficients change per step
    jeval = None
    try:
        jeval = compile_jacobian(compiled, allv, base,
                                 deriv_names=tuple("$" + s for s in states),
                                 param_names=(tv,))
    except Exception:
        jeval = None

    steps_est = int(round((t1 - t0) / h0))
    if steps_est < 1:
        return {"success": False,
                "messages": [{"level": "error", "code": "solve_failed",
                              "args": {"detail": "t1 must be greater than t0"}}],
                "values": None, "series": None, "maxResidual": None, "method": None}
    if steps_est > 200000:
        return {"success": False,
                "messages": [{"level": "error", "code": "err_too_many_steps",
                              "args": {"n": steps_est}}],
                "values": None, "series": None, "maxResidual": None, "method": None}

    prev: Dict[str, float] = {n: info["_initial_guess"].get(n, 0.0) for n in allv}

    # ---- consistent initialisation: freeze states, free the algebraic
    #      variables and the state derivatives -----------------------------
    init_names = alg + ["$" + s for s in states]
    init_x0: Dict[str, float] = {}
    for n in alg:
        init_x0[n] = prev[n]
    for s in states:
        init_x0["$" + s] = 0.0
    if init_names:
        init_base = dict(base)
        init_base[tv] = t0
        for s in states:
            init_base[s] = prev[s]
        ok, sol, method, pairs = _root_solve(init_names, init_x0, init_base, compiled)
        if not ok:
            if method == "dof_mismatch":
                msg = {"level": "error", "code": "err_dof_mismatch",
                       "args": {"ne": len(init_names), "nu": len(init_names)}}
            else:
                msg = {"level": "error", "code": "solve_failed",
                       "args": {"detail": "consistent initialisation: " + method}}
            return {"success": False, "messages": [msg], "values": None,
                    "series": None, "maxResidual": None, "method": None}
        for n, v in zip(init_names, sol):
            if not n.startswith("$"):
                prev[n] = float(v)
        messages.append({"level": "info", "code": "info_consistent_init", "args": {}})

    # ---- BDF stepping -------------------------------------------------------
    adaptive = fixed_h is None
    h_cur = h0 if adaptive else fixed_h
    h_max = h0 if adaptive else fixed_h
    h_min = h_max * 1e-5
    hprev = h_cur
    rtol, atol = 1e-6, 1e-8
    max_steps = 500000

    times = [t0]
    series = {n: [prev[n]] for n in allv}
    prevprev = dict(prev)
    t = t0
    first = True
    steps_taken = 0

    def make_env(x, tnext, hcur, xn, xnm1, order):
        env = dict(base)
        env[tv] = tnext
        for n, v in zip(allv, x):
            env[n] = float(v)
        for s in states:
            i = idx[s]
            if order == 2:
                r = hcur / hprev
                c1 = (2.0 * r + 1.0) / (hcur * (r + 1.0))
                c2 = -(r * r) / (hcur * (r + 1.0))
                env["$" + s] = c1 * (x[i] - xn[s]) + c2 * (xn[s] - xnm1[s])
            else:
                env["$" + s] = (x[i] - xn[s]) / hcur
        return env

    def bdf_res(x, tnext, hcur, xn, xnm1, order):
        env = make_env(x, tnext, hcur, xn, xnm1, order)
        try:
            r, _ = residuals(compiled, env)
            return np.where(np.isfinite(r), r, 1e10)
        except Exception:
            return np.full(nv, 1e10)

    def solve_step(tnext, hcur, xn, xnm1, order):
        x0 = np.array([(2.0 * xn[n] - xnm1[n]) if n in states else xn[n]
                       for n in allv], dtype=float)
        fun = lambda x: bdf_res(x, tnext, hcur, xn, xnm1, order)
        jac = None
        if jeval is not None:
            if order == 2:
                r = hcur / hprev
                c1 = (2.0 * r + 1.0) / (hcur * (r + 1.0))
            else:
                c1 = 1.0 / hcur
            c1_map = {"$" + s: c1 for s in states}

            def jac_fn(x, _t=tnext, _c=c1_map):
                return jeval.eval(x, {tv: _t}, _c)

            jac = jac_fn
        if jac is not None and not jacobian_is_available(jac, x0, allv, base):
            jac = None
        if jac is not None:
            xs, ok_n, _ = _newton(fun, jac, x0)
            if ok_n:
                return True, xs
        for m in ("hybr", "lm"):
            try:
                opts = {"maxfev": 2000} if m == "hybr" else {"maxiter": 2000}
                cand = root(fun, x0, method=m, options=opts)
                if cand.success:
                    env_c = make_env(cand.x, tnext, hcur, xn, xnm1, order)
                    rc, _ = residuals(compiled, env_c)
                    if np.all(np.isfinite(rc)) and np.max(np.abs(rc)) < 1e-6:
                        return True, cand.x
            except Exception:
                continue
        return False, None

    while t < t1 - 1e-12:
        tnext = t + h_cur
        final_step = False
        if tnext >= t1:
            tnext = t1
            h_cur = tnext - t
            final_step = True
        order = 1 if first else 2
        ok, x2 = solve_step(tnext, h_cur, prev, prevprev, order)
        if not ok:
            if adaptive and h_cur > h_min * 2:
                h_cur = max(h_min, h_cur * 0.5)
                continue
            messages.append({"level": "error", "code": "dynamic_step_failed",
                             "args": {"t": f"{t:.4g}",
                                      "detail": "no convergence"}})
            env = make_env(
                np.array([prev[n] for n in allv], dtype=float),
                t, h_cur, prev, prevprev, order)
            r, ids = residuals(compiled, env)
            maxr = max((abs(v) for v in r), default=0.0)
            return {"success": False, "messages": messages, "values": dict(prev),
                    "series": {"t": times, "variables": series},
                    "residuals": [{"id": i, "value": float(v)}
                                  for i, v in zip(ids, r)],
                    "maxResidual": maxr, "method": "BDF2 adaptive"}

        err = 0.0
        if adaptive and order == 2:
            ok1, x1 = solve_step(tnext, h_cur, prev, prevprev, 1)
            if ok1:
                # |BDF2 - BDF1| is O(h^2) while the true BDF2 LTE is O(h^3),
                # so scale by h to obtain an O(h^3) error estimate
                err = max(h_cur * abs(x2[i] - x1[i]) / (atol + rtol * abs(x2[i]))
                          for i in range(nv))
            if err > 1.0:
                h_cur = max(h_min, h_cur * 0.9 * (1.0 / err) ** (1.0 / 3.0))
                continue  # reject, retry from the same previous point
            h_next = h_cur * 0.9 * min(2.0, (1.0 / max(err, 1e-12)) ** (1.0 / 3.0))
        else:
            h_next = h_cur

        new = {n: float(v) for n, v in zip(allv, x2)}
        prevprev = dict(prev)
        prev = new
        hprev = h_cur
        t = tnext
        first = False
        times.append(t)
        for n in allv:
            series[n].append(prev[n])
        steps_taken += 1
        if steps_taken > max_steps:
            messages.append({"level": "error", "code": "dynamic_step_failed",
                             "args": {"t": f"{t:.4g}",
                                      "detail": "step limit reached"}})
            break
        h_cur = min(h_max, max(h_min, h_next))
        if final_step:
            break

    env = dict(base)
    env[tv] = t1
    for n in allv:
        env[n] = prev[n]
    r, ids = residuals(compiled, env)
    maxr = max((abs(v) for v in r), default=0.0)
    method_label = "BDF2 adaptive" if adaptive else "BDF2 fixed"
    messages.append({"level": "info", "code": "dynamic_ok",
                     "args": {"n": len(times), "t0": f"{t0:.6g}", "t1": f"{t1:.6g}",
                              "h": f"{h_max:.6g}"}})
    success = steps_taken <= max_steps and all(
        m["code"] != "dynamic_step_failed" for m in messages)
    return {"success": success, "messages": messages, "values": dict(prev),
            "series": {"t": times, "variables": series},
            "residuals": [{"id": i, "value": float(v)} for i, v in zip(ids, r)],
            "maxResidual": maxr, "method": method_label}


# --------------------------------------------------------------------------
# Optimization (scipy SLSQP) — the EQUATION section acts as the constraint
# set, the OPTIMIZE section defines the objective, and BOUNDS (optional)
# bound the free variables.
# --------------------------------------------------------------------------
def solve_optimize(model: Model, info: dict) -> dict:
    names: List[str] = info["_unknowns"]
    base: Dict[str, float] = dict(info["_fixed_values"])
    guesses: Dict[str, float] = info["_initial_guess"]
    bounds_map: Dict[str, tuple] = info["_bounds"]
    obj_meta = info["_objective"]
    compiled = compile_model(model)

    obj_fn = make_eval(model.objective.expr)
    sign = -1.0 if model.objective.kind == "maximize" else 1.0
    x0 = np.array([guesses.get(n, 0.0) for n in names], dtype=float)

    n_cons = 0
    try:
        r0, _ = residuals(compiled, {**base, **guesses})
        n_cons = len(r0)
    except Exception:
        pass

    def fun(x):
        env = dict(base)
        for n, v in zip(names, x):
            env[n] = float(v)
        return sign * float(obj_fn(env))

    def cons(x):
        env = dict(base)
        for n, v in zip(names, x):
            env[n] = float(v)
        try:
            r, _ = residuals(compiled, env)
            return np.where(np.isfinite(r), r, 1e10)
        except Exception:
            return np.full(max(n_cons, 1), 1e10)

    bounds = [(bounds_map.get(n, (None, None))) for n in names]
    constraints = [{"type": "eq", "fun": cons}]
    jac = None
    try:
        jac = build_jacobian(compiled, names, base)
        if jac is not None and jacobian_is_available(jac, x0, names, base):
            constraints[0]["jac"] = jac
        else:
            jac = None
    except Exception:
        jac = None
    grad = build_gradient(model.objective.expr, names, base)
    if grad is not None:
        _g0, _sign0 = grad, sign
        grad = lambda x: _sign0 * np.asarray(_g0(x), dtype=float)

    try:
        res = minimize(fun, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, jac=grad,
                       options={"maxiter": 1000, "ftol": 1e-12})
    except Exception as e:
        return {"success": False, "messages": [{"level": "error",
                "code": "optimize_failed", "args": {"detail": str(e)}}],
                "values": None, "objective": None, "residuals": [],
                "maxResidual": None, "method": "SLSQP"}

    values = {n: float(v) for n, v in zip(names, res.x)}
    env = dict(base)
    for n, v in zip(names, res.x):
        env[n] = float(v)
    r, ids = residuals(compiled, env)
    maxr = max((abs(v) for v in r), default=0.0)
    obj_value = float(obj_fn(env))

    if res.success and maxr < 1e-6:
        messages = [{"level": "info", "code": "optimize_ok",
                     "args": {"method": "SLSQP", "obj": f"{obj_value:.8g}",
                              "res": f"{maxr:.2e}"}}]
        success = True
    else:
        detail = res.message or "no convergence"
        messages = [{"level": "error", "code": "optimize_failed",
                     "args": {"detail": detail}}]
        success = False

    return {"success": success, "messages": messages, "values": values,
            "objective": {"kind": obj_meta["kind"], "value": obj_value},
            "residuals": [{"id": i, "value": float(v)} for i, v in zip(ids, r)],
            "maxResidual": maxr, "method": "SLSQP"}
