"""Analytic Jacobians via SymPy.

The residual closures in the compiled model carry their AST nodes; this
module translates those nodes into SymPy expressions, differentiates them
with respect to the unknowns, and produces a Jacobian callable.

If any construct cannot be translated (e.g. a function SymPy cannot
differentiate, or a conditional that must stay numeric), the builder returns
None and the solver falls back to its numerical path.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import sympy as sp

from ..parser.gproms_parser import Num, Var, Deriv, UnOp, BinOp, Func

_CUT = sp.Float(1e-10)


def _safe_log(x):
    """Mirror of solver.safe_ln: log(x) for x>eps, linear extension below."""
    return sp.Piecewise(
        (sp.log(x), x > _CUT),
        (sp.log(_CUT) + (x - _CUT) / _CUT, True),
    )


def _safe_log10(x):
    """Mirror of solver.safe_log10."""
    return sp.Piecewise(
        (sp.log(x, 10), x > _CUT),
        (sp.log(_CUT, 10) + (x - _CUT) / (_CUT * sp.log(10)), True),
    )


def _safe_sqrt(x):
    """Mirror of solver.safe_sqrt: sqrt(x) for x>0, else 0."""
    return sp.Piecewise((sp.sqrt(x), x > 0), (0, True))


def _safe_exp(x):
    """Mirror of solver.safe_exp: clipped to [-745, 700]."""
    return sp.exp(sp.Min(x, sp.Float(700.0)))


_UNARY = {
    "EXP": _safe_exp,
    "LN": _safe_log,
    "LOG": _safe_log10,
    "SQRT": _safe_sqrt,
    "ABS": sp.Abs,
    "SIN": sp.sin,
    "COS": sp.cos,
    "TAN": sp.tan,
}


def _to_sympy(node, sym_of: Callable[[str], sp.Expr]) -> sp.Expr:
    """Translate an AST node into a SymPy expression.
    `sym_of(name)` maps canonical variable names to SymPy expressions
    (Symbols for unknowns, Floats for fixed parameters)."""
    if isinstance(node, Num):
        return sp.Float(node.v)
    if isinstance(node, Var):
        return sym_of(node.name.lower())
    if isinstance(node, Deriv):
        return sym_of("$" + node.name.lower())
    if isinstance(node, UnOp):
        x = _to_sympy(node.x, sym_of)
        if node.op == "-":
            return -x
        raise ValueError("NOT is not differentiable")
    if isinstance(node, BinOp):
        op = node.op
        if op in ("=", "<>", "<", "<=", ">", ">=", "AND", "OR"):
            raise ValueError(f"{op} is not differentiable")
        l = _to_sympy(node.l, sym_of)
        r = _to_sympy(node.r, sym_of)
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            return l / r
        if op in ("^", "**"):
            return l ** r
        raise ValueError(f"unknown operator {op}")
    if isinstance(node, Func):
        if node.name in ("MAX", "MIN"):
            args = [_to_sympy(a, sym_of) for a in node.args]
            return sp.Max(*args) if node.name == "MAX" else sp.Min(*args)
        if node.name in _UNARY and len(node.args) == 1:
            return _UNARY[node.name](_to_sympy(node.args[0], sym_of))
        raise ValueError(f"{node.name} is not differentiable")
    raise ValueError(f"unsupported node {type(node).__name__}")


def build_jacobian(compiled: list, names: List[str],
                   fixed: Dict[str, float]) -> Optional[Callable]:
    """Build a jac(x) -> ndarray callable for the active branch structure,
    or None if the symbolic translation is not possible.

    Only steady-state style systems are handled here: every referenced name
    that is not an unknown must be resolvable to a constant (fixed value or
    zero). Derivative placeholders '$name' are not constant and cause a
    graceful fallback.
    """
    syms: Dict[str, sp.Expr] = {n: sp.Symbol(n) for n in names}
    sym_unknowns = [syms[n] for n in names]

    def sym_of(name: str) -> sp.Expr:
        if name in syms:
            return syms[name]
        if name in fixed:
            return sp.Float(fixed[name])
        raise ValueError(f"unresolved name {name}")

    try:
        # Build the symbolic mirror of the compiled structure. Each leaf gets
        # a pre-lambdified gradient row.
        def c(eq):
            if eq[0] == "eq":
                expr = _to_sympy(eq[4], sym_of) - _to_sympy(eq[5], sym_of)
                grad = [sp.diff(expr, s) for s in sym_unknowns]
                lam = sp.lambdify(sym_unknowns, grad, "numpy")
                return ("eq", lam)
            if eq[0] == "ineq":
                return None
            return ("if", eq[1], [c(e) for e in eq[2]], [c(e) for e in eq[3]])

        symbolic = [c(e) for e in compiled]
    except Exception:
        return None

    def jac(x) -> np.ndarray:
        env = dict(fixed)
        for n, v in zip(names, x):
            env[n] = float(v)
        rows: List[np.ndarray] = []

        def walk(eq) -> None:
            if eq is None:
                return
            if eq[0] == "eq":
                try:
                    with np.errstate(all="ignore"):
                        rows.append(np.asarray(eq[1](*[env[n] for n in names]),
                                               dtype=float))
                except Exception:
                    raise ValueError("jacobian evaluation failed")
            else:
                branch = eq[2] if eq[1](env) != 0.0 else eq[3]
                for e in branch:
                    walk(e)

        try:
            for e in symbolic:
                walk(e)
        except ValueError:
            return None
        if len(rows) != len(names):
            return None
        return np.array(rows, dtype=float)

    return jac


def build_gradient(node, names: List[str], fixed: Dict[str, float]) -> Optional[Callable]:
    """Build grad(x) -> ndarray for an objective AST node, or None if the
    symbolic translation is not possible."""
    syms: Dict[str, sp.Expr] = {n: sp.Symbol(n) for n in names}

    def sym_of(name: str) -> sp.Expr:
        if name in syms:
            return syms[name]
        if name in fixed:
            return sp.Float(fixed[name])
        raise ValueError(f"unresolved name {name}")

    try:
        expr = _to_sympy(node, sym_of)
        grad = [sp.diff(expr, syms[n]) for n in names]
        lam = sp.lambdify([syms[n] for n in names], grad, "numpy")
    except Exception:
        return None

    def g(x) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(lam(*[float(v) for v in x]), dtype=float)

    return g


def jacobian_is_available(jac: Optional[Callable], x0, names: List[str],
                          fixed: Dict[str, float]) -> bool:
    """Sanity-check that the built Jacobian evaluates to a finite square
    matrix at the initial guess."""
    if jac is None:
        return False
    try:
        J = np.asarray(jac(x0), dtype=float)
        return J.shape == (len(names), len(names)) and np.all(np.isfinite(J))
    except Exception:
        return False
