"""Distributed (1D/2D) system expansion — method of lines.

A MODEL may declare DISTRIBUTION domains (name, extent L, N points) and
variables of type DISTRIBUTION(Axial[, Radial]) — one dimension per domain.
Spatial operators PARTIAL(T, dom) and PARTIAL(T, dom, dom) are replaced by
finite-difference stencils:

    interior:      central differences
    domain edges:  one-sided differences (boundary equations)

Equations containing distributed references expand into one equation per
interior grid point; BOUNDARY equations (with `AT dom = pos`) expand along
the whole edge. Corner points receive boundary equations from every edge
they touch; exact duplicates are deduplicated.

INT(expr, dom) computes a trapezoidal integral over a 1D domain.
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


def _eq_vars(eq, dist_vars: Set[str]) -> Set[str]:
    out: Set[str] = set()
    if isinstance(eq, EqEquation):
        for node in (eq.lhs, eq.rhs):
            for v in dist_vars:
                if _contains_distributed(node, {v}):
                    out.add(v)
    else:
        for v in dist_vars:
            if _eq_contains(eq, {v}):
                out.add(v)
    return out


def _div(a, b):
    return BinOp("/", a, b)


def _num(v: float) -> Num:
    return Num(v)


def _stencil(var_name: str, pt: Dict[str, int], dom: str, order: int,
             edge: Optional[str], domains: Dict[str, DomainInfo],
             dist_vars: Dict[str, tuple]) -> Node:
    """Difference stencil for PARTIAL(T, dom[, dom]) at the current point."""
    dims = dist_vars[var_name.lower()]
    d = domains[dom]
    n, dz = d.points, d.dz
    k = pt[dom]

    def T_at(idx: int) -> Node:
        return Indexed(var_name,
                       [_num(idx if dd == dom else pt[dd]) for dd in dims])

    at_first = (edge == dom and k == 1)
    at_last = (edge == dom and k == n)
    if at_first:
        if order == 1:
            return _div(BinOp("-", T_at(2), T_at(1)), _num(dz))
        return _div(BinOp("+", BinOp("-", T_at(3), BinOp("*", _num(2.0),
                                                          T_at(2))),
                          T_at(1)), _num(dz * dz))
    if at_last:
        if order == 1:
            return _div(BinOp("-", T_at(n), T_at(n - 1)), _num(dz))
        return _div(BinOp("+", BinOp("-", T_at(n),
                                     BinOp("*", _num(2.0), T_at(n - 1))),
                          T_at(n - 2)), _num(dz * dz))
    if order == 1:
        return _div(BinOp("-", T_at(k + 1), T_at(k - 1)), _num(2.0 * dz))
    return _div(BinOp("+", BinOp("-", T_at(k + 1),
                                 BinOp("*", _num(2.0), T_at(k))),
                      T_at(k - 1)), _num(dz * dz))


def _expand_at(node, pt: Dict[str, int], domains: Dict[str, DomainInfo],
               dist_vars: Dict[str, tuple], edge: Optional[str]):
    if isinstance(node, (Num, Str)):
        return node
    if isinstance(node, Var):
        if node.name.lower() in dist_vars:
            dims = dist_vars[node.name.lower()]
            return Indexed(node.name, [_num(pt[d]) for d in dims])
        return node
    if isinstance(node, Deriv):
        if node.name.lower() in dist_vars:
            dims = dist_vars[node.name.lower()]
            return Deriv(f"{node.name}({','.join(str(pt[d]) for d in dims)})")
        return node
    if isinstance(node, Indexed):
        return Indexed(node.name, [_expand_at(i, pt, domains, dist_vars, edge)
                                   for i in node.indices])
    if isinstance(node, UnOp):
        return UnOp(node.op, _expand_at(node.x, pt, domains, dist_vars, edge))
    if isinstance(node, BinOp):
        return BinOp(node.op,
                     _expand_at(node.l, pt, domains, dist_vars, edge),
                     _expand_at(node.r, pt, domains, dist_vars, edge))
    if isinstance(node, Func):
        if node.name == "PARTIAL" and len(node.args) in (2, 3):
            v = node.args[0]
            dom_arg = node.args[1]
            if not isinstance(v, Var) or not isinstance(dom_arg, Var):
                raise ValueError("PARTIAL requires a variable and a domain")
            dom = dom_arg.name.lower()
            if dom not in domains:
                raise ValueError(f"undeclared distribution domain "
                                 f"{dom_arg.name}")
            if v.name.lower() not in dist_vars or dom not in dist_vars[v.name.lower()]:
                raise ValueError(f"{v.name} is not distributed over "
                                 f"{dom_arg.name}")
            order = 2 if len(node.args) == 3 else 1
            return _stencil(v.name, pt, dom, order, edge, domains, dist_vars)
        return Func(node.name, [_expand_at(a, pt, domains, dist_vars, edge)
                                for a in node.args])
    if isinstance(node, IfExpr):
        return IfExpr(_expand_at(node.cond, pt, domains, dist_vars, edge),
                      _expand_at(node.then, pt, domains, dist_vars, edge),
                      _expand_at(node.els, pt, domains, dist_vars, edge))
    raise ValueError(f"cannot expand {type(node).__name__}")


def _expand_eq(eq, pt, domains, dist_vars, edge) -> Equation:
    if isinstance(eq, EqEquation):
        return EqEquation(
            _expand_at(eq.lhs, pt, domains, dist_vars, edge),
            _expand_at(eq.rhs, pt, domains, dist_vars, edge), eq.op)
    return IfEquation(
        _expand_at(eq.cond, pt, domains, dist_vars, edge),
        [_expand_eq(e, pt, domains, dist_vars, edge) for e in eq.then_eqs],
        [_expand_eq(e, pt, domains, dist_vars, edge) for e in eq.else_eqs])


def _product_interior(doms: List[str],
                      domains: Dict[str, DomainInfo]) -> List[Dict[str, int]]:
    """All interior point combinations over the given domains."""
    pts: List[Dict[str, int]] = [{}]
    for d in doms:
        n = domains[d].points
        new = []
        for pt in pts:
            for k in range(2, n):
                p2 = dict(pt)
                p2[d] = k
                new.append(p2)
        pts = new
    return pts


def _product_full(doms: List[str],
                  domains: Dict[str, DomainInfo]) -> List[Dict[str, int]]:
    pts: List[Dict[str, int]] = [{}]
    for d in doms:
        n = domains[d].points
        new = []
        for pt in pts:
            for k in range(1, n + 1):
                p2 = dict(pt)
                p2[d] = k
                new.append(p2)
        pts = new
    return pts


def _resolve_boundary_point(at_expr, domain: DomainInfo, env) -> Optional[int]:
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


# --------------------------------------------------------------------------
# INT(expr, dom): trapezoidal integral over a 1D domain
# --------------------------------------------------------------------------
def _expand_int(node, domains: Dict[str, DomainInfo],
                dist_vars: Dict[str, tuple]) -> Node:
    if isinstance(node, Func) and node.name == "INT" and len(node.args) == 2:
        expr = node.args[0]
        dom_arg = node.args[1]
        if not isinstance(dom_arg, Var):
            raise ValueError("INT requires a domain name")
        dom = domains.get(dom_arg.name.lower())
        if dom is None:
            raise ValueError(f"undeclared distribution domain {dom_arg.name}")
        # the integrated expression must be 1D over exactly this domain
        involved = set()
        for v, dims in dist_vars.items():
            if _contains_distributed(expr, {v}):
                involved |= set(dims)
        if involved != {dom_arg.name.lower()}:
            raise ValueError("INT requires a 1D variable over the same domain")
        terms = []
        for k in range(1, dom.points + 1):
            pt = {dom_arg.name.lower(): k}
            ek = _expand_at(expr, pt, domains, dist_vars, None)
            if k == 1 or k == dom.points:
                terms.append(BinOp("*", _num(0.5), ek))
            else:
                terms.append(ek)
        total = terms[0]
        for t in terms[1:]:
            total = BinOp("+", total, t)
        return BinOp("*", _num(dom.dz), total)
    if isinstance(node, UnOp):
        return UnOp(node.op, _expand_int(node.x, domains, dist_vars))
    if isinstance(node, BinOp):
        return BinOp(node.op, _expand_int(node.l, domains, dist_vars),
                     _expand_int(node.r, domains, dist_vars))
    if isinstance(node, Func):
        return Func(node.name, [_expand_int(a, domains, dist_vars)
                                for a in node.args])
    if isinstance(node, IfExpr):
        return IfExpr(_expand_int(node.cond, domains, dist_vars),
                      _expand_int(node.then, domains, dist_vars),
                      _expand_int(node.els, domains, dist_vars))
    if isinstance(node, Indexed):
        return Indexed(node.name, [_expand_int(i, domains, dist_vars)
                                   for i in node.indices])
    return node


def _expand_int_eq(eq, domains, dist_vars) -> Equation:
    if isinstance(eq, EqEquation):
        return EqEquation(_expand_int(eq.lhs, domains, dist_vars),
                          _expand_int(eq.rhs, domains, dist_vars), eq.op)
    return IfEquation(_expand_int(eq.cond, domains, dist_vars),
                      [_expand_int_eq(e, domains, dist_vars)
                       for e in eq.then_eqs],
                      [_expand_int_eq(e, domains, dist_vars)
                       for e in eq.else_eqs])


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def expand_equations(model: Model, domains: Dict[str, DomainInfo],
                     dist_vars: Dict[str, tuple], env) -> List[Equation]:
    """Expand EQUATION and BOUNDARY sections into concrete per-point
    equations. INT nodes become trapezoidal sums first."""
    out: List[Equation] = []
    for eq in model.equations:
        eq = _expand_int_eq(eq, domains, dist_vars)
        vars_ = _eq_vars(eq, set(dist_vars))
        if not vars_:
            out.append(eq)
            continue
        involved: List[str] = []
        for v in vars_:
            for d in dist_vars[v]:
                if d not in domains:
                    raise ValueError(f"undeclared distribution domain {d}")
                if d not in involved:
                    involved.append(d)
        for pt in _product_interior(involved, domains):
            out.append(_expand_eq(eq, pt, domains, dist_vars, None))

    seen: Set[str] = set()
    for bc in model.boundaries:
        if not bc.at_domain or bc.at_expr is None:
            raise ValueError("boundary equation requires AT <domain> = <pos>")
        dom_key = bc.at_domain.lower()
        if dom_key not in domains:
            raise ValueError(f"unknown distribution domain {bc.at_domain}")
        domain = domains[dom_key]
        edge_pos = _resolve_boundary_point(bc.at_expr, domain, env)
        if edge_pos is None:
            raise ValueError(
                f"boundary position must be 0 or the domain extent")
        vars_ = _eq_vars(bc.eq, set(dist_vars))
        if not vars_:
            raise ValueError("boundary equation has no distributed variable")
        other_dims: List[str] = []
        for v in vars_:
            for d in dist_vars[v]:
                if d != dom_key and d not in other_dims:
                    other_dims.append(d)
        for pt in _product_full(other_dims, domains):
            pt2 = dict(pt)
            pt2[dom_key] = edge_pos
            e = _expand_eq(bc.eq, pt2, domains, dist_vars, dom_key)
            key = expr_str(e.lhs) + " = " + expr_str(e.rhs) \
                if isinstance(e, EqEquation) else expr_str(e.cond)
            key += " | " + ",".join(f"{d}:{pt2[d]}" for d in sorted(pt2))
            if key in seen:
                continue   # duplicate corner condition — deduplicate
            seen.add(key)
            out.append(e)
    return out
