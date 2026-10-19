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

from ..parser.gproms_parser import Num, Str, Var, Deriv, UnOp, BinOp, Func, Indexed

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


def _to_sympy(node, sym_of: Callable[[str], sp.Expr],
              arrays: Optional[Dict[str, int]] = None) -> sp.Expr:
    """Translate an AST node into a SymPy expression.
    `sym_of(name)` maps canonical variable names to SymPy expressions
    (Symbols for unknowns, Floats for fixed parameters). `arrays` maps
    array base names to their sizes so symbolic indices (SIGMA) can be
    expanded to piecewise element lookups."""
    if isinstance(node, Num):
        return sp.Float(node.v)
    if isinstance(node, Var):
        return sym_of(node.name.lower())
    if isinstance(node, Deriv):
        return sym_of("$" + node.name.lower())
    if isinstance(node, UnOp):
        x = _to_sympy(node.x, sym_of, arrays)
        if node.op == "-":
            return -x
        raise ValueError("NOT is not differentiable")
    if isinstance(node, BinOp):
        op = node.op
        if op in ("=", "<>", "<", "<=", ">", ">=", "AND", "OR"):
            raise ValueError(f"{op} is not differentiable")
        l = _to_sympy(node.l, sym_of, arrays)
        r = _to_sympy(node.r, sym_of, arrays)
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
        if node.name == "SIGMA" and len(node.args) == 4:
            lo = _to_sympy(node.args[2], sym_of, arrays)
            hi = _to_sympy(node.args[3], sym_of, arrays)
            if not (isinstance(lo, sp.Number) and isinstance(hi, sp.Number)):
                raise ValueError("non-numeric SIGMA bounds")
            idx = node.args[1]
            if not isinstance(idx, Var):
                raise ValueError("SIGMA index must be a variable")
            idx_sym = sp.Symbol(idx.name.lower())

            def idx_sym_of(name):
                if name == idx.name.lower():
                    return idx_sym
                return sym_of(name)

            total = sp.Integer(0)
            for k in range(int(lo), int(hi) + 1):
                total += _to_sympy(node.args[0], idx_sym_of, arrays).subs(
                    idx_sym, sp.Integer(k))
            return total
        if node.name in ("MAX", "MIN"):
            args = [_to_sympy(a, sym_of, arrays) for a in node.args]
            return sp.Max(*args) if node.name == "MAX" else sp.Min(*args)
        if node.name in _UNARY and len(node.args) == 1:
            return _UNARY[node.name](_to_sympy(node.args[0], sym_of, arrays))
        raise ValueError(f"{node.name} is not differentiable")
    if isinstance(node, Indexed):
        base = node.name.lower()
        idxs = [_to_sympy(i, sym_of, arrays) for i in node.indices]

        def lookup(d, acc):
            if d == len(idxs):
                return sym_of(f"{base}({','.join(acc)})")
            s = idxs[d]
            if isinstance(s, sp.Number):
                return lookup(d + 1, acc + [str(int(s))])
            if arrays and base in arrays and d < len(arrays[base]):
                size = arrays[base][d]
                terms = [(lookup(d + 1, acc + [str(k)]),
                          sp.Eq(s, sp.Integer(k)))
                         for k in range(1, size + 1)]
                return sp.Piecewise(*terms, (sp.Integer(0), True))
            raise ValueError("symbolic index outside SIGMA")

        return lookup(0, [])
    raise ValueError(f"unsupported node {type(node).__name__}")


class JacobianEvaluator:
    """Pre-compiled analytic Jacobian.

    Built once per model structure; `eval` then applies the current
    linear derivative couplings (BDF coefficients) numerically:
        J = J_x + sum_j c1_j * dF/d('$s_j')   (chain rule)
    """
    def __init__(self, names, param_names, deriv_names, deriv_cols,
                 rows_x, rows_d, structure):
        self.names = names
        self.param_names = param_names
        self.deriv_names = deriv_names
        self.deriv_cols = deriv_cols
        self.rows_x = rows_x      # list of lambdas -> gradient wrt names
        self.rows_d = rows_d      # list of lambdas -> gradient wrt derivs
        self.structure = structure

    def eval(self, x, params: Dict[str, float],
             c1_map: Dict[str, float]) -> Optional[np.ndarray]:
        env = dict(params)
        for n, v in zip(self.names, x):
            env[n] = float(v)
        rows: List[np.ndarray] = []
        row_iter = iter(range(len(self.rows_x)))
        with np.errstate(all="ignore"):
            try:
                def walk(eq) -> None:
                    if eq[0] == "eq":
                        i = next(row_iter)
                        args = [env[n] for n in self.names] + \
                               [env[p] for p in self.param_names]
                        J = np.asarray(self.rows_x[i](*args), dtype=float)
                        if self.deriv_names:
                            rd = np.asarray(self.rows_d[i](*args), dtype=float)
                            for j, d in enumerate(self.deriv_names):
                                c = c1_map.get(d, 0.0)
                                if c:
                                    J[self.deriv_cols[j]] += c * rd[j]
                        rows.append(J)
                    elif eq[0] == "if":
                        branch = eq[2] if eq[1](env) != 0.0 else eq[3]
                        for e in branch:
                            walk(e)

                for e in self.structure:
                    walk(e)
            except (ValueError, KeyError, IndexError):
                return None
        if len(rows) != len(self.names):
            return None
        return np.array(rows, dtype=float)


def compile_jacobian(compiled: list, names: List[str],
                     fixed: Dict[str, float],
                     deriv_names: tuple = (),
                     param_names: tuple = (),
                     array_sizes: Optional[Dict[str, int]] = None
                     ) -> Optional[JacobianEvaluator]:
    """Compile the analytic Jacobian structure once.

    * `deriv_names`: derivative placeholders ('$s') that are linear in an
      unknown — their numeric coefficients are supplied per evaluation.
    * `param_names`: names treated as evaluation-time constants (e.g. the
      schedule time variable).
    * `array_sizes`: base name -> array length, so symbolic indices inside
      SIGMA can be expanded to piecewise element lookups.
    """
    syms: Dict[str, sp.Expr] = {n: sp.Symbol(n) for n in names}
    dsyms: Dict[str, sp.Expr] = {d: sp.Symbol(d) for d in deriv_names}
    psyms: Dict[str, sp.Expr] = {p: sp.Symbol(p) for p in param_names}
    all_syms = [syms[n] for n in names] + [psyms[p] for p in param_names]
    array_sizes = array_sizes or {}

    def sym_of(name: str) -> sp.Expr:
        if name in syms:
            return syms[name]
        if name in dsyms:
            return dsyms[name]
        if name in psyms:
            return psyms[name]
        if name in fixed:
            return sp.Float(fixed[name])
        raise ValueError(f"unresolved name {name}")

    rows_x: List = []
    rows_d: List = []
    deriv_cols = [names.index(d[1:]) for d in deriv_names
                  if d.startswith("$") and d[1:] in syms]
    if len(deriv_cols) != len(deriv_names):
        return None

    def c(eq) -> bool:
        if eq[0] == "eq":
            try:
                expr = (_to_sympy(eq[4], sym_of, array_sizes)
                        - _to_sympy(eq[5], sym_of, array_sizes))
            except Exception:
                return False
            gx = [sp.diff(expr, syms[n]) for n in names]
            rows_x.append(sp.lambdify(all_syms, gx, "numpy"))
            gd = [sp.diff(expr, dsyms[d]) for d in deriv_names]
            rows_d.append(sp.lambdify(all_syms, gd, "numpy"))
            return True
        if eq[0] == "ineq":
            return True
        return all(c(e) for e in eq[2]) and all(c(e) for e in eq[3])

    try:
        if not all(c(e) for e in compiled):
            return None
    except Exception:
        return None

    return JacobianEvaluator(names, list(param_names), list(deriv_names),
                             deriv_cols, rows_x, rows_d, compiled)


def build_jacobian(compiled: list, names: List[str],
                   fixed: Dict[str, float],
                   extra_consts: Optional[Dict[str, float]] = None,
                   lin_deriv: Optional[Dict[str, tuple]] = None,
                   array_sizes: Optional[Dict[str, int]] = None
                   ) -> Optional[Callable]:
    """Build a jac(x) -> ndarray callable for the active branch structure,
    or None if the symbolic translation is not possible.

    * `extra_consts`: names resolved to numeric constants (e.g. the schedule
      time variable during dynamic solves).
    * `lin_deriv`: derivative placeholders expressed as linear functions of
      an unknown: {'$s': (c1, c0)} means '$s' = c1*s + c0 — used by the
      BDF stepping so the Jacobian includes the discretisation terms.
    * `array_sizes`: base name -> array length (for symbolic SIGMA indices).
    """
    extra_consts = extra_consts or {}
    lin_deriv = lin_deriv or {}
    try:
        ev = compile_jacobian(compiled, names, fixed,
                              deriv_names=tuple(lin_deriv.keys()),
                              param_names=tuple(extra_consts.keys()),
                              array_sizes=array_sizes)
    except Exception:
        return None
    if ev is None:
        return None
    c1_map = {d: float(c1) for d, (c1, _c0) in lin_deriv.items()}

    def jac(x) -> np.ndarray:
        return ev.eval(x, extra_consts, c1_map)

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
