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
import scipy.sparse as spm
import sympy as sp

from ..parser.gproms_parser import Num, Str, Var, Deriv, UnOp, BinOp, Func, Indexed

# Safe-domain guards built from sp.Max/sp.Min: their derivatives are
# Heaviside terms that lambdify cleanly to np.heaviside (no Piecewise
# select constructs, no sign(x)/x singular forms).
_CUT = sp.Float(1e-10)


def _safe_log(x):
    """Mirror of solver.safe_ln: log(x) for x>eps, linear extension below."""
    return sp.log(sp.Max(x, _CUT)) + sp.Min(x - _CUT, 0) / _CUT


def _safe_log10(x):
    """Mirror of solver.safe_log10 (same linear-extension construction)."""
    return (sp.log(sp.Max(x, _CUT), 10)
            + sp.Min(x - _CUT, 0) / (_CUT * sp.log(10)))


def _safe_sqrt(x):
    """Mirror of solver.safe_sqrt: sqrt(x) for x>0, else 0."""
    return sp.sqrt(sp.Max(x, 0))


def _safe_exp(x):
    """Plain exp — the [-745, 700] clip lives only in the NUMERIC evaluator.
    A symbolic Min cap makes sp.diff blow up exponentially on deep
    expressions; overflow here is caught by the finite-Jacobian check."""
    return sp.exp(x)


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


def _resolve_derivatives(expr: sp.Expr) -> sp.Expr:
    """Evaluate unevaluated Derivative nodes (e.g. Abs -> sign) so the
    expression is printable by the numpy codegen printer."""
    return expr.replace(
        lambda e: isinstance(e, sp.Derivative),
        lambda e: e.doit())


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
            idx_sym = sp.Symbol(idx.name.lower(), real=True)

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
    """Pre-compiled analytic Jacobian (sparse-aware).

    Built once per model structure; `eval` then applies the current
    linear derivative couplings (BDF coefficients) numerically:
        J = J_x + sum_j c1_j * dF/d('$s_j')   (chain rule)
    The result is a scipy sparse matrix when the fill ratio is low,
    otherwise a dense array.
    """
    def __init__(self, names, param_names, deriv_names, deriv_cols,
                 rows, structure):
        self.names = names
        self.param_names = param_names
        self.deriv_names = deriv_names
        self.deriv_cols = deriv_cols
        # rows: list of (base_cols, base_lam, der_present, der_lam)
        self.rows = rows
        self.structure = structure
        self.n = len(names)

    def eval(self, x, params: Dict[str, float],
             c1_map: Dict[str, float]):
        env = dict(params)
        for n, v in zip(self.names, x):
            env[n] = float(v)
        ii: List[int] = []
        jj: List[int] = []
        vv: List[float] = []
        rows_walked = [0]
        with np.errstate(all="ignore"):
            try:
                def walk(eq) -> None:
                    if eq[0] == "eq":
                        i = rows_walked[0]
                        rows_walked[0] += 1
                        row = self.rows[i]
                        if row[0] == "num":
                            # numeric central-difference row (property
                            # functions and other non-differentiable leaves)
                            _lhs, _rhs = row[1], row[2]
                            for j in range(self.n):
                                xj = float(env[self.names[j]])
                                h = 1e-6 * max(1.0, abs(xj))
                                e1 = dict(env)
                                e1[self.names[j]] = xj + h
                                e2 = dict(env)
                                e2[self.names[j]] = xj - h
                                try:
                                    d = (float(_lhs(e1)) - float(_rhs(e1))
                                         - (float(_lhs(e2))
                                            - float(_rhs(e2)))) / (2 * h)
                                except (ValueError, TypeError):
                                    continue
                                if np.isfinite(d):
                                    ii.append(i)
                                    jj.append(j)
                                    vv.append(float(d))
                            return
                        base_cols, base_lam, der_present, der_lam, \
                            used_names = row[1:]
                        args = [env[n] for n in used_names]
                        if base_cols:
                            vals = np.asarray(base_lam(args), dtype=float)
                            for j, v in zip(base_cols, vals):
                                if v != 0.0:
                                    ii.append(i)
                                    jj.append(j)
                                    vv.append(float(v))
                        if der_present:
                            dvals = np.asarray(der_lam(args), dtype=float)
                            for jj2, v in zip(der_present, dvals):
                                c = c1_map.get(self.deriv_names[jj2], 0.0)
                                if c and v != 0.0:
                                    ii.append(i)
                                    jj.append(self.deriv_cols[jj2])
                                    vv.append(float(c * v))
                    elif eq[0] == "if":
                        branch = eq[2] if eq[1](env) != 0.0 else eq[3]
                        for e in branch:
                            walk(e)

                for e in self.structure:
                    walk(e)
            except (ValueError, KeyError, IndexError, TypeError,
                    ArithmeticError):
                return None
        n_rows = rows_walked[0]
        mat = spm.coo_matrix((vv, (ii, jj)), shape=(n_rows, self.n))
        if len(vv) <= 0.6 * n_rows * self.n:
            return mat.tocsr()
        return mat.toarray()


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
    syms: Dict[str, sp.Expr] = {n: sp.Symbol(n, real=True) for n in names}
    dsyms: Dict[str, sp.Expr] = {d: sp.Symbol(d, real=True) for d in deriv_names}
    psyms: Dict[str, sp.Expr] = {p: sp.Symbol(p, real=True) for p in param_names}
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

    rows: List = []
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
                # non-differentiable leaf (property functions, ...):
                # fall back to a numeric central-difference row
                rows.append(("num", eq[1], eq[2]))
                return True
            # structural sparsity: only the columns whose symbols appear
            base_cols = [j for j, n in enumerate(names)
                         if syms[n] in expr.free_symbols]
            der_present = [j for j, d in enumerate(deriv_names)
                           if dsyms[d] in expr.free_symbols]
            # only the symbols actually used by the DIFFERENTIATED
            # expressions are passed to lambdify (after diff, '$x' terms
            # vanish, keeping the generated functions tiny and fast)
            g_base = [sp.diff(expr, syms[names[j]]) for j in base_cols]
            g_der = [sp.diff(expr, dsyms[deriv_names[j]])
                     for j in der_present]
            g_base = [_resolve_derivatives(g) for g in g_base]
            g_der = [_resolve_derivatives(g) for g in g_der]
            used = set()
            for g in g_base + g_der:
                used |= g.free_symbols
            used = sorted(used, key=str)
            used_names = [str(s) for s in used]
            try:
                base_lam = sp.lambdify([used], g_base, "numpy")
                der_lam = sp.lambdify([used], g_der, "numpy")
            except Exception:
                rows.append(("num", eq[1], eq[2]))
                return True
            rows.append(("sym", base_cols, base_lam, der_present, der_lam,
                         used_names))
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
                             deriv_cols, rows, compiled)


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
        J = ev.eval(x, extra_consts, c1_map)
        if J is None:
            return None
        if spm.issparse(J):
            return J.toarray()
        return J

    return jac


def build_gradient(node, names: List[str], fixed: Dict[str, float]) -> Optional[Callable]:
    """Build grad(x) -> ndarray for an objective AST node, or None if the
    symbolic translation is not possible."""
    syms: Dict[str, sp.Expr] = {n: sp.Symbol(n, real=True) for n in names}

    def sym_of(name: str) -> sp.Expr:
        if name in syms:
            return syms[name]
        if name in fixed:
            return sp.Float(fixed[name])
        raise ValueError(f"unresolved name {name}")

    try:
        expr = _to_sympy(node, sym_of)
        grad = [sp.diff(expr, syms[n]) for n in names]
        grad = [_resolve_derivatives(g) for g in grad]
        used = set()
        for g in grad:
            used |= g.free_symbols
        used = sorted(used, key=str)
        used_names = [str(s) for s in used]
        lam = sp.lambdify([used], grad, "numpy")
    except Exception:
        return None

    def g(x) -> np.ndarray:
        with np.errstate(all="ignore"):
            envmap = dict(zip(names, [float(v) for v in x]))
            return np.asarray(lam([envmap[n] for n in used_names]),
                              dtype=float)

    return g


def jacobian_is_available(jac: Optional[Callable], x0, names: List[str],
                          fixed: Dict[str, float]) -> bool:
    """Sanity-check that the built Jacobian evaluates to a finite square
    matrix at the initial guess."""
    if jac is None:
        return False
    try:
        J = jac(x0)
        if J is None:
            return False
        if spm.issparse(J):
            return (J.shape == (len(names), len(names))
                    and bool(np.all(np.isfinite(J.data))))
        J = np.asarray(J, dtype=float)
        return J.shape == (len(names), len(names)) and np.all(np.isfinite(J))
    except Exception:
        return False
