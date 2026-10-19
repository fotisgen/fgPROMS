"""Distributed (1D) system expansion — method of lines.

A MODEL may declare a DISTRIBUTION domain (name, extent L, N points) and
variables of type DISTRIBUTION(Axial). Spatial operators PARTIAL(T, Axial)
and PARTIAL(T, Axial, Axial) are replaced by finite-difference stencils:

    interior (2..N-1):  central differences
    boundary (1, N):    one-sided differences

Equations containing distributed references expand into one equation per
interior point; BOUNDARY equations (with an `AT Axial = pos` clause) expand
at the corresponding boundary point. Everything downstream (indexed
variables, SIGMA, Jacobians, BDF) works on the expanded system unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .parser.gproms_parser import (
    Model, Num, Str, Var, Deriv, UnOp, BinOp, Func, IfExpr, Indexed,
    EqEquation, IfEquation, expr_str,
)


@dataclass
class DomainInfo:
    name: str          # display name
    extent: float
    points: int
    dz: float


def _contains_distributed(node, dist_vars: Set[str]) -> bool:
    if isinstance(node, Var):
        return node.name.lower() in dist_vars
    if isinstance(node, Deriv):
        return node.name.lower() in dist_vars
    if isinstance(node, Func):
        return any(_contains_distributed(a, dist_vars) for a in node.args)
    if isinstance(node, UnOp):
        return _contains_distributed(node.x, dist_vars)
    if isinstance(node, BinOp):
        return (_contains_distributed(node.l, dist_vars)
                or _contains_distributed(node.r, dist_vars))
    if isinstance(node, IfExpr):
        return (_contains_distributed(node.cond, dist_vars)
                or _contains_distributed(node.then, dist_vars)
                or _contains_distributed(node.els, dist_vars))
    if isinstance(node, Indexed):
        return any(_contains_distributed(i, dist_vars) for i in node.indices)
    return False


def _eq_contains(eq, dist_vars: Set[str]) -> bool:
    if isinstance(eq, EqEquation):
        return (_contains_distributed(eq.lhs, dist_vars)
                or _contains_distributed(eq.rhs, dist_vars))
    return (_contains_distributed(eq.cond, dist_vars)
            or any(_eq_contains(e, dist_vars) for e in eq.then_eqs)
            or any(_eq_contains(e, dist_vars) for e in eq.else_eqs))


def _div(a, b):
    return BinOp("/", a, b)


def _num(v: float) -> Num:
    return Num(v)


def _stencil(var_name: str, k: int, domain: DomainInfo,
             order: int, at_boundary: Optional[int]) -> Node:
    """Difference stencil for PARTIAL(T, A[, A]) at point k.
    at_boundary: None (interior), 1 (first point), N (last point)."""
    n = domain.points
    dz = domain.dz

    def T(j: int) -> Node:
        return Indexed(var_name, [_num(j)])

    if at_boundary is None:
        if order == 1:
            return _div(BinOp("-", T(k + 1), T(k - 1)), _num(2.0 * dz))
        return _div(BinOp("+", BinOp("-", T(k + 1), BinOp("*", _num(2.0), T(k))),
                             T(k - 1)), _num(dz * dz))
    if at_boundary == 1:
        if order == 1:
            return _div(BinOp("-", T(2), T(1)), _num(dz))
        return _div(BinOp("+", BinOp("-", T(3), BinOp("*", _num(2.0), T(2))),
                             T(1)), _num(dz * dz))
    if order == 1:
        return _div(BinOp("-", T(n), T(n - 1)), _num(dz))
    return _div(BinOp("+", BinOp("-", T(n), BinOp("*", _num(2.0), T(n - 1))),
                         T(n - 2)), _num(dz * dz))


def _expand_at(node, k: int, domain: DomainInfo, dist_vars: Set[str],
               at_boundary: Optional[int]):
    if isinstance(node, (Num, Str)):
        return node
    if isinstance(node, Var):
        if node.name.lower() in dist_vars:
            return Indexed(node.name, [_num(k)])
        return node
    if isinstance(node, Deriv):
        if node.name.lower() in dist_vars:
            return Deriv(f"{node.name}({k})")
        return node
    if isinstance(node, Indexed):
        return Indexed(node.name, [_expand_at(i, k, domain, dist_vars,
                                              at_boundary)
                                   for i in node.indices])
    if isinstance(node, UnOp):
        return UnOp(node.op, _expand_at(node.x, k, domain, dist_vars,
                                        at_boundary))
    if isinstance(node, BinOp):
        return BinOp(node.op,
                     _expand_at(node.l, k, domain, dist_vars, at_boundary),
                     _expand_at(node.r, k, domain, dist_vars, at_boundary))
    if isinstance(node, Func):
        if node.name == "PARTIAL":
            if len(node.args) == 2:
                v = node.args[0]
                if not isinstance(v, Var):
                    raise ValueError("PARTIAL requires a variable")
                return _stencil(v.name, k, domain, 1, at_boundary)
            if len(node.args) == 3:
                v = node.args[0]
                if not isinstance(v, Var):
                    raise ValueError("PARTIAL requires a variable")
                return _stencil(v.name, k, domain, 2, at_boundary)
            raise ValueError("PARTIAL takes 2 or 3 arguments")
        return Func(node.name, [_expand_at(a, k, domain, dist_vars,
                                           at_boundary) for a in node.args])
    if isinstance(node, IfExpr):
        return IfExpr(_expand_at(node.cond, k, domain, dist_vars, at_boundary),
                      _expand_at(node.then, k, domain, dist_vars, at_boundary),
                      _expand_at(node.els, k, domain, dist_vars, at_boundary))
    raise ValueError(f"cannot expand {type(node).__name__}")


def _expand_eq(eq, k: int, domain: DomainInfo, dist_vars: Set[str],
               at_boundary: Optional[int]) -> Equation:
    if isinstance(eq, EqEquation):
        return EqEquation(
            _expand_at(eq.lhs, k, domain, dist_vars, at_boundary),
            _expand_at(eq.rhs, k, domain, dist_vars, at_boundary), eq.op)
    return IfEquation(
        _expand_at(eq.cond, k, domain, dist_vars, at_boundary),
        [_expand_eq(e, k, domain, dist_vars, at_boundary)
         for e in eq.then_eqs],
        [_expand_eq(e, k, domain, dist_vars, at_boundary)
         for e in eq.else_eqs])


def _resolve_boundary_point(at_expr, domain: DomainInfo, env) -> Optional[int]:
    """Evaluate the AT position: 0 -> first point, extent -> last point."""
    if at_expr is None:
        return None
    try:
        from .solver.solver import make_eval
        v = float(make_eval(at_expr)(env))
    except Exception:
        return None
    if abs(v) < 1e-9 * max(1.0, abs(domain.extent)):
        return 1
    if abs(v - domain.extent) < 1e-9 * max(1.0, abs(domain.extent)):
        return domain.points
    return None


def expand_equations(model: Model, domains: Dict[str, DomainInfo],
                     dist_vars: Dict[str, str], env) -> List[Equation]:
    """Expand EQUATION and BOUNDARY sections into concrete per-point
    equations. Equations without distributed references pass through."""
    out: List[Equation] = []
    for eq in model.equations:
        if not _eq_contains(eq, set(dist_vars)):
            out.append(eq)
            continue
        # find the single domain involved
        involved = {dist_vars[v] for v in dist_vars
                    if _eq_contains(eq, {v})}
        if len(involved) != 1:
            raise ValueError("equation mixes multiple distribution domains")
        dom_key = next(iter(involved))
        if dom_key not in domains:
            raise ValueError(f"undeclared distribution domain {dom_key}")
        domain = domains[dom_key]
        for k in range(2, domain.points):
            out.append(_expand_eq(eq, k, domain, set(dist_vars), None))

    for bc in model.boundaries:
        if not bc.at_domain or bc.at_expr is None:
            raise ValueError("boundary equation requires AT <domain> = <pos>")
        dom_key = bc.at_domain.lower()
        if dom_key not in domains:
            raise ValueError(f"unknown distribution domain {bc.at_domain}")
        domain = domains[dom_key]
        if not _eq_contains(bc.eq, set(dist_vars)):
            raise ValueError("boundary equation has no distributed variable")
        pt = _resolve_boundary_point(bc.at_expr, domain, env)
        if pt is None:
            raise ValueError(
                f"boundary position must be 0 or the domain extent")
        out.append(_expand_eq(bc.eq, pt, domain, set(dist_vars), pt))
    return out
