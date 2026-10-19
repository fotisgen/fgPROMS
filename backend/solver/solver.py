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
    Num, Str, Var, Deriv, UnOp, BinOp, Func, IfExpr, Indexed,
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
                   "SIGN", "MAX", "MIN", "SIGMA"}

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
        elif node.name == "SIGMA" and len(node.args) == 4:
            expr_fn = make_eval(node.args[0])
            idx_name = (node.args[1].name.lower()
                        if isinstance(node.args[1], Var) else None)
            lo_fn = make_eval(node.args[2])
            hi_fn = make_eval(node.args[3])

            def fn(env, _e=expr_fn, _l=lo_fn, _h=hi_fn, _i=idx_name):
                if _i is None:
                    return 0.0
                lo = int(round(_l(env)))
                hi = int(round(_h(env)))
                total = 0.0
                for k in range(lo, hi + 1):
                    env[_i] = float(k)
                    total += _e(env)
                env.pop(_i, None)
                return total
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
    elif isinstance(node, Indexed):
        base = node.name.lower()
        idx_fns = [make_eval(i) for i in node.indices]

        def fn(env, _b=base, _fs=idx_fns):
            key = (_b + "("
                   + ",".join(str(int(round(f(env)))) for f in _fs) + ")")
            return env.get(key, 0.0)
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
                t: float = 0.0, _disable_newton: bool = False,
                array_sizes: Optional[Dict[str, int]] = None
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
            jac = build_jacobian(compiled, names, base, array_sizes=array_sizes)
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

    ok, sol, method, pairs = _root_solve(names, guesses, base, compiled,
                                         array_sizes=info.get("_array_sizes"))

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
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Dynamic solve — variable-order BDF (orders 1..5) with adaptive step control
# and event-driven task phases.
#
#   * BDF(k) on a non-uniform mesh via Lagrange interpolant derivative.
#   * Local error estimate: embedded BDF(k-1) solution, h-scaled.
#   * Order climbing up to 5 when the higher-order estimate passes.
#   * TASK phases integrate one at a time; a phase with `END WHEN cond`
#     terminates when cond crosses zero (detected by sign change +
#     bisection refinement), and the next phase chains from the crossing.
# --------------------------------------------------------------------------
def _bdf_coeffs(t_new: float, t_hist: List[float], k: int) -> List[float]:
    """Derivative coefficients of BDF(k) on a non-uniform mesh.

    Interpolation points: t_new followed by t_hist[0..k-1] (newest first).
    Returns c[0..k] such that  x'(t_new) = sum_j c[j] * x(points[j]).
    """
    pts = [t_new] + list(t_hist[:k])
    n = len(pts)
    c = []
    for j in range(n):
        tj = pts[j]
        others = [m for m in range(n) if m != j]
        s = 0.0
        for m in others:
            prod = 1.0
            for l in others:
                if l != m:
                    prod *= (t_new - pts[l]) / (tj - pts[l])
            s += (1.0 / (tj - pts[m])) * prod
        c.append(s)
    return c


def _bdf1_step_solve(cur, je, allv, states, idx, base, tv, t_target,
                     x_anchor, h):
    """Solve one BDF1 step from the anchor state x_anchor at time t_target."""
    nv = len(allv)
    x0 = np.array([x_anchor[n] for n in allv], dtype=float)

    def fun(x):
        env = dict(base)
        env[tv] = t_target
        for n, v in zip(allv, x):
            env[n] = float(v)
        for s in states:
            i = idx[s]
            env["$" + s] = (x[i] - x_anchor[s]) / h
        r, _ = residuals(cur, env)
        return np.where(np.isfinite(r), r, 1e10)

    jac = None
    if je is not None:
        c1_map = {"$" + s: 1.0 / h for s in states}

        def jac_fn(x, _t=t_target, _c=c1_map):
            return je.eval(x, {tv: _t}, _c)

        jac = jac_fn
        if not jacobian_is_available(jac, x0, allv, base):
            jac = None
    if jac is not None:
        xs, ok, _ = _newton(fun, jac, x0)
        if ok:
            return True, xs
    for m in ("hybr", "lm"):
        try:
            opts = {"maxfev": 2000} if m == "hybr" else {"maxiter": 2000}
            cand = root(fun, x0, method=m, options=opts)
            if cand.success:
                env_c = dict(base)
                env_c[tv] = t_target
                for n, v in zip(allv, cand.x):
                    env_c[n] = float(v)
                rc, _ = residuals(cur, env_c)
                if np.all(np.isfinite(rc)) and np.max(np.abs(rc)) < 1e-6:
                    return True, cand.x
        except Exception:
            continue
    return False, None


def _integrate_phase(cur, je, allv, states, idx, base, tv, t_start, t_end,
                     x_start, cond_fn, times, series, h_max, adaptive,
                     f0=None, max_steps=500000, rtol=1e-6, atol=1e-8):
    """Adaptive variable-order BDF over [t_start, t_end] for ONE equation
    set. Optional event detection: when cond_fn crosses zero upward the
    phase stops at the (bisected) crossing point.

    Returns (ok, t_final, x_final, f_final, err_msg, max_order_used)."""
    nv = len(allv)
    h_cur = h_max
    h_min = h_max * 1e-5
    t = t_start
    x_hist: List[Dict[str, float]] = [dict(x_start)]
    t_hist: List[float] = [t_start]
    f_hist: List[Dict[str, float]] = []
    if f0 is not None:
        f_hist.append(dict(f0))
    order = 1
    max_order_used = 1
    steps_taken = 0

    def make_env(x, tnext, hcur, k):
        env = dict(base)
        env[tv] = tnext
        for n, v in zip(allv, x):
            env[n] = float(v)
        if k == 1:
            for s in states:
                i = idx[s]
                env["$" + s] = (x[i] - x_hist[0][s]) / hcur
        else:
            c = _bdf_coeffs(tnext, t_hist, k)
            for s in states:
                i = idx[s]
                val = c[0] * x[i]
                for j in range(1, k + 1):
                    val += c[j] * x_hist[j - 1][s]
                env["$" + s] = val
        return env

    def bdf_res(x, tnext, hcur, k):
        env = make_env(x, tnext, hcur, k)
        try:
            r, _ = residuals(cur, env)
            return np.where(np.isfinite(r), r, 1e10)
        except Exception:
            return np.full(nv, 1e10)

    def solve_step(tnext, hcur, k):
        x0 = np.array([(2.0 * x_hist[0][n] - x_hist[1][n])
                       if (n in states and len(x_hist) > 1) else x_hist[0][n]
                       for n in allv], dtype=float)
        fun = lambda x: bdf_res(x, tnext, hcur, k)
        jac = None
        if je is not None:
            if k == 1:
                c0 = 1.0 / hcur
            else:
                c0 = _bdf_coeffs(tnext, t_hist, k)[0]
            c1_map = {"$" + s: c0 for s in states}

            def jac_fn(x, _t=tnext, _c=c1_map):
                return je.eval(x, {tv: _t}, _c)

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
                    env_c = make_env(cand.x, tnext, hcur, k)
                    rc, _ = residuals(cur, env_c)
                    if np.all(np.isfinite(rc)) and np.max(np.abs(rc)) < 1e-6:
                        return True, cand.x
            except Exception:
                continue
        return False, None

    def err_norm(xa, xb, hcur):
        return max(hcur * abs(xa[i] - xb[i]) / (atol + rtol * abs(xa[i]))
                   for i in range(nv))

    cond_prev = None
    if cond_fn is not None:
        env0 = dict(base)
        env0[tv] = t_start
        for n in allv:
            env0[n] = x_start[n]
        try:
            cond_prev = float(cond_fn(env0))
        except Exception:
            cond_prev = None

    while t < t_end - 1e-12:
        tnext = t + h_cur
        final_step = False
        if tnext >= t_end:
            tnext = t_end
            h_cur = tnext - t
            final_step = True
            if h_cur < h_min:   # vanishing final step: accept current state
                t = t_end
                break
        ok, xk = solve_step(tnext, h_cur, order)
        if not ok:
            if adaptive and h_cur > h_min * 2:
                h_cur = max(h_min, h_cur * 0.5)
                order = max(1, order - 1)
                continue
            return False, t, dict(x_hist[0]), f_hist[0] if f_hist else {}, \
                "no convergence", max_order_used

        # ---- error estimate for the current order -------------------------
        if order == 1 and f_hist:
            # Euler predictor difference: |x1 - (x_n + h*f_n)|
            err = max(
                h_cur * abs(xk[i] - (x_hist[0][n]
                                     + h_cur * f_hist[0].get(n, 0.0)))
                / (atol + rtol * abs(xk[i]))
                for i, n in enumerate(allv) if n in states)
            if not states:
                err = 0.2
        elif order == 1:
            err = 0.2
        else:
            ok1, x1 = solve_step(tnext, h_cur, order - 1)
            if ok1:
                err = err_norm(xk, x1, h_cur)
            else:
                err = 0.5

        adopted = xk
        if (adaptive and not final_step and order < 5
                and len(t_hist) >= order + 1):
            okp, xp = solve_step(tnext, h_cur, order + 1)
            if okp:
                errp = err_norm(xp, xk, h_cur)
                if errp < 1.0:
                    adopted = xp
                    order += 1
                    err = errp

        if adaptive and err > 1.0:
            h_cur = max(h_min, h_cur * 0.9 *
                        (1.0 / err) ** (1.0 / (order + 1)))
            order = max(1, order - 1)
            continue

        if adaptive and not final_step:
            h_next = h_cur * 0.9 * min(
                2.0, (1.0 / max(err, 1e-12)) ** (1.0 / (order + 1)))
        else:
            h_next = h_cur

        # ---- accept ---------------------------------------------------------
        t_pre = t
        x_pre = dict(x_hist[0])
        new = {n: float(v) for n, v in zip(allv, adopted)}
        env_acc = make_env(np.array([new[n] for n in allv], dtype=float),
                           tnext, h_cur, order)
        f_new = {s: float(env_acc["$" + s]) for s in states}
        x_hist.insert(0, new)
        t_hist.insert(0, tnext)
        f_hist.insert(0, f_new)
        if len(x_hist) > 6:
            x_hist.pop()
            t_hist.pop()
            f_hist.pop()
        max_order_used = max(max_order_used, order)
        if not adaptive and steps_taken >= 1:
            order = 2          # constant-step BDF2 (tests / fallback)
        t = tnext
        times.append(t)
        for n in allv:
            series[n].append(new[n])
        steps_taken += 1
        if steps_taken > max_steps:
            return False, t, new, f_new, "step limit reached", max_order_used

        # ---- event detection (END WHEN) -------------------------------------
        event_hit = False
        if cond_fn is not None and cond_prev is not None:
            envn = dict(base)
            envn[tv] = t
            for n in allv:
                envn[n] = new[n]
            try:
                cond_new = float(cond_fn(envn))
            except Exception:
                cond_new = cond_prev
            if cond_prev <= 0.0 < cond_new:
                ta_t, xa_d, va = t_pre, x_pre, cond_prev
                tb_t, vb = t, cond_new
                for _ in range(45):
                    tm = 0.5 * (ta_t + tb_t)
                    hb = tm - ta_t
                    if hb < 1e-12:
                        break
                    okm, xm = _bdf1_step_solve(cur, je, allv, states, idx,
                                               base, tv, tm, xa_d, hb)
                    if not okm:
                        break
                    envm = dict(base)
                    envm[tv] = tm
                    for n, v in zip(allv, xm):
                        envm[n] = float(v)
                    try:
                        vm = float(cond_fn(envm))
                    except Exception:
                        break
                    if vm <= 0.0:
                        ta_t, xa_d, va = tm, dict(zip(allv, xm)), vm
                    else:
                        tb_t, vb = tm, vm
                # replace the accepted point with the refined crossing
                t = ta_t
                new = dict(xa_d)
                x_hist[0] = new
                t_hist[0] = ta_t
                times[-1] = ta_t
                for n in allv:
                    series[n][-1] = xa_d[n]
                cond_new = va
                event_hit = True
            cond_prev = cond_new

        h_cur = min(h_max, max(h_min, h_next))
        if final_step or event_hit:
            break

    return True, t, dict(x_hist[0]), f_hist[0] if f_hist else {}, None, \
        max_order_used


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

    steps_est = int(round((t1 - t0) / h0))
    if steps_est < 1:
        return {"success": False,
                "messages": [{"level": "error", "code": "solve_failed",
                              "args": {"detail": "t1 must be greater than t0"}}],
                "values": None, "series": None, "maxResidual": None,
                "method": None, "maxOrder": 1}
    if steps_est > 200000:
        return {"success": False,
                "messages": [{"level": "error", "code": "err_too_many_steps",
                              "args": {"n": steps_est}}],
                "values": None, "series": None, "maxResidual": None,
                "method": None, "maxOrder": 1}

    # pre-compile the analytic Jacobian ONCE for the common equations
    jeval = None
    try:
        jeval = compile_jacobian(compiled, allv, base,
                                 deriv_names=tuple("$" + s for s in states),
                                 param_names=(tv,),
                                 array_sizes=info.get("_array_sizes"))
    except Exception:
        jeval = None

    prev: Dict[str, float] = {n: info["_initial_guess"].get(n, 0.0) for n in allv}

    # ---- consistent initialisation: freeze states, free the algebraic
    #      variables and the state derivatives -----------------------------
    init_names = alg + ["$" + s for s in states]
    init_x0: Dict[str, float] = {}
    for n in alg:
        init_x0[n] = prev[n]
    for s in states:
        init_x0["$" + s] = 0.0
    f_init: Dict[str, float] = {}
    if init_names:
        init_base = dict(base)
        init_base[tv] = t0
        for s in states:
            init_base[s] = prev[s]
        init_compiled = compiled
        phases0 = info.get("_phases") or {}
        if phases0:
            first_p = next(iter(model.task_phases))
            init_compiled = compiled + compile_model(
                Model(name=model.name, line=model.line,
                      equations=model.task_phases[first_p].equations))
        ok, sol, method, pairs = _root_solve(init_names, init_x0, init_base,
                                             init_compiled,
                                             array_sizes=info.get("_array_sizes"))
        if not ok:
            if method == "dof_mismatch":
                msg = {"level": "error", "code": "err_dof_mismatch",
                       "args": {"ne": len(init_names), "nu": len(init_names)}}
            else:
                msg = {"level": "error", "code": "solve_failed",
                       "args": {"detail": "consistent initialisation: " + method}}
            return {"success": False, "messages": [msg], "values": None,
                    "series": None, "maxResidual": None, "method": None,
                    "maxOrder": 1}
        for n, v in zip(init_names, sol):
            if not n.startswith("$"):
                prev[n] = float(v)
            else:
                f_init[n[1:]] = float(v)
        messages.append({"level": "info", "code": "info_consistent_init",
                         "args": {}})

    adaptive = fixed_h is None
    times = [t0]
    series = {n: [prev[n]] for n in allv}
    max_order_all = 1

    phases = info.get("_phases") or {}
    last_compiled = compiled

    if phases:
        # ---- phased integration: one BDF run per phase, chained -----------
        t_cur = t0
        f_hand = dict(f_init)
        first_phase = True
        ended_early = False
        for pname, (s0, s1, hp) in phases.items():
            ph = model.task_phases[pname]
            cur = compile_model(Model(name=model.name, line=model.line,
                                      equations=ph.equations))
            last_compiled = cur
            je = None
            try:
                je = compile_jacobian(
                    cur, allv, base,
                    deriv_names=tuple("$" + s for s in states),
                    param_names=(tv,),
                    array_sizes=info.get("_array_sizes"))
            except Exception:
                je = None
            cond_fn = make_eval(ph.when) if ph.when is not None else None
            if ended_early:
                t_start = t_cur          # chain from the previous event
            else:
                t_start = max(s0, t_cur)
            t_end = max(s1, t_start)
            # if the end condition already holds at the start, skip the phase
            if cond_fn is not None:
                env0 = dict(base)
                env0[tv] = t_start
                for n in allv:
                    env0[n] = prev[n]
                try:
                    if float(cond_fn(env0)) > 0.0:
                        t_cur = t_start
                        first_phase = False
                        continue
                except Exception:
                    pass
            hmax = hp if adaptive else fixed_h
            f0 = dict(f_hand) if not first_phase else dict(f_init)
            ok, t_f, prev, f_hand, emsg, mo = _integrate_phase(
                cur, je, allv, states, idx, base, tv, t_start, t_end, prev,
                cond_fn, times, series, hmax, adaptive, f0=f0)
            max_order_all = max(max_order_all, mo)
            first_phase = False
            if not ok:
                messages.append({"level": "error", "code": "dynamic_step_failed",
                                 "args": {"t": f"{t_f:.4g}",
                                          "detail": emsg or "no convergence"}})
                break
            ended_early = t_f < s1 - 1e-9
            t_cur = t_f
    else:
        ok, t_f, prev, f_hand, emsg, mo = _integrate_phase(
            compiled, jeval, allv, states, idx, base, tv, t0, t1, prev,
            None, times, series, h0 if adaptive else fixed_h, adaptive,
            f0=dict(f_init))
        max_order_all = max(max_order_all, mo)
        if not ok:
            messages.append({"level": "error", "code": "dynamic_step_failed",
                             "args": {"t": f"{t_f:.4g}",
                                      "detail": emsg or "no convergence"}})

    env = dict(base)
    env[tv] = t1
    for n in allv:
        env[n] = prev[n]
    r, ids = residuals(last_compiled, env)
    maxr = max((abs(v) for v in r), default=0.0)
    method_label = "BDF adaptive" if adaptive else "BDF fixed"
    messages.append({"level": "info", "code": "dynamic_ok",
                     "args": {"n": len(times), "t0": f"{t0:.6g}",
                              "t1": f"{t1:.6g}", "h": f"{h0:.6g}"}})
    success = all(m["code"] != "dynamic_step_failed" for m in messages)
    return {"success": success, "messages": messages, "values": dict(prev),
            "series": {"t": times, "variables": series},
            "residuals": [{"id": i, "value": float(v)} for i, v in zip(ids, r)],
            "maxResidual": maxr, "method": method_label,
            "maxOrder": max_order_all}


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
        jac = build_jacobian(compiled, names, base,
                             array_sizes=info.get("_array_sizes"))
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


# --------------------------------------------------------------------------
# Parameter estimation (gEST-style): fit ESTIMATE parameters to DATA by
# least squares. The model is solved inside the objective — every candidate
# parameter set runs a full (steady or dynamic) simulation.
# --------------------------------------------------------------------------
def solve_estimation(model: Model, info: dict) -> dict:
    est: List[str] = info["_estimate"]
    data: List[tuple] = info["_data"]
    bounds_map: Dict[str, tuple] = info["_bounds"]
    base_fixed: Dict[str, float] = dict(info["_fixed_values"])
    dynamic = info["_schedule"] is not None and info["_states"]

    x0 = [base_fixed.get(p, 0.0) for p in est]

    def simulate(params_dict: Dict[str, float]):
        info2 = dict(info)
        info2["_fixed_values"] = dict(base_fixed)
        info2["_fixed_values"].update(params_dict)
        if dynamic:
            res = solve_dynamic(model, info2)
            if not res["success"]:
                return None
            ts = res["series"]

            def at(var, t_i):
                return float(np.interp(t_i, ts["t"], ts["variables"][var]))
            return at, res
        res = solve_steady(model, info2)
        if not res["success"]:
            return None

        def at(var, t_i):
            return float(res["values"][var])
        return at, res

    def obj(x):
        p = {n: float(v) for n, v in zip(est, x)}
        sim = simulate(p)
        if sim is None:
            return 1e12
        at, _ = sim
        sse = 0.0
        for var, pts in data:
            for t_i, val in pts:
                sse += (at(var, t_i) - val) ** 2
        return sse

    bounds = [bounds_map.get(p, (None, None)) for p in est]
    try:
        r = minimize(obj, x0, method="SLSQP", bounds=bounds,
                     options={"maxiter": 300, "ftol": 1e-10})
    except Exception as e:
        return {"success": False,
                "messages": [{"level": "error", "code": "err_estimate_failed",
                              "args": {"detail": str(e)}}],
                "values": None, "estimated": None, "sse": None,
                "dataFit": None, "residuals": [], "maxResidual": None,
                "method": "SLSQP", "series": None}

    best = {n: float(v) for n, v in zip(est, r.x)}
    sim = simulate(best)
    if sim is None or not r.success:
        detail = getattr(r, "message", "no convergence")
        return {"success": False,
                "messages": [{"level": "error", "code": "err_estimate_failed",
                              "args": {"detail": detail}}],
                "values": None, "estimated": best, "sse": float(obj(r.x)),
                "dataFit": None, "residuals": [], "maxResidual": None,
                "method": "SLSQP", "series": None}

    at, res_final = sim
    sse = float(obj(r.x))
    data_fit = []
    for var, pts in data:
        for t_i, val in pts:
            data_fit.append({"var": var, "t": t_i, "measured": val,
                             "model": float(at(var, t_i))})
    messages = [{"level": "info", "code": "estimate_ok",
                 "args": {"method": "SLSQP", "sse": f"{sse:.6g}"}}]
    return {"success": True, "messages": messages,
            "values": res_final.get("values"),
            "residuals": res_final.get("residuals"),
            "maxResidual": res_final.get("maxResidual"),
            "method": "SLSQP", "estimated": best, "sse": sse,
            "dataFit": data_fit,
            "series": res_final.get("series")}
