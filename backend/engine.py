"""fgPROMS execution engine: parse -> analyze -> solve."""
from __future__ import annotations

from .parser.gproms_parser import parse_file, ParseError
from .analyzer.analyzer import analyze
from .solver.solver import solve_steady, solve_dynamic, solve_optimize, make_eval
from .flowsheet import analyze_process


def _error(code: str, args: dict) -> dict:
    return {"ok": False,
            "messages": [{"level": "error", "code": code, "args": args}],
            "model": None, "results": None, "timeSeries": None}


def run_model(code: str) -> dict:
    """Full pipeline. Returns a JSON-serialisable result dictionary."""
    if not code or not code.strip():
        return _error("err_empty", {})

    try:
        parsed = parse_file(code)
    except ParseError as e:
        if e.code == "err_unsupported_section":
            return _error("err_unsupported_section", {"section": e.msg.split("'")[1]})
        return _error("parse_error",
                      {"line": e.line, "col": e.col, "detail": e.msg})

    # ---- flowsheet (PROCESS) ----------------------------------------------
    if parsed.process is not None:
        info = analyze_process(parsed.process, parsed.models)
        messages = list(info["messages"])
        if info["blocked"]:
            return {"ok": True, "messages": messages, "model": info["model"],
                    "results": None, "timeSeries": None}
        res = solve_steady(info["_synth"], info["_info"])
        messages += res["messages"]
        results = {"mode": "steady", "success": res["success"],
                   "method": res.get("method"),
                   "values": res.get("values"),
                   "residuals": res.get("residuals"),
                   "maxResidual": res.get("maxResidual")}
        return {"ok": True, "messages": messages, "model": info["model"],
                "results": results, "timeSeries": None}

    model = next(iter(parsed.models.values()))
    info = analyze(model)
    messages = list(info["messages"])

    if info["blocked"]:
        return {"ok": True, "messages": messages, "model": info["model"],
                "results": None, "timeSeries": None}

    # ---- optimization ------------------------------------------------------
    if info["_is_optimization"]:
        if info["_dof"] == 0:
            # degenerate case: system fully determined -> solve, then
            # evaluate the objective at the solution point
            res = solve_steady(model, info)
            messages += res["messages"]
            obj_value = None
            if res.get("values"):
                env = dict(info["_fixed_values"])
                env.update(res["values"])
                obj_value = float(make_eval(model.objective.expr)(env))
            results = {"mode": "optimization", "success": res["success"],
                       "method": res.get("method"),
                       "values": res.get("values"),
                       "residuals": res.get("residuals"),
                       "maxResidual": res.get("maxResidual"),
                       "objective": {"kind": model.objective.kind,
                                     "value": obj_value}}
        else:
            res = solve_optimize(model, info)
            messages += res["messages"]
            results = {"mode": "optimization", "success": res["success"],
                       "method": res.get("method"),
                       "values": res.get("values"),
                       "residuals": res.get("residuals"),
                       "maxResidual": res.get("maxResidual"),
                       "objective": res.get("objective")}
        return {"ok": True, "messages": messages, "model": info["model"],
                "results": results, "timeSeries": None}

    dynamic = info["model"]["isDynamic"] and info["_schedule"] is not None

    if dynamic:
        res = solve_dynamic(model, info)
        messages += res["messages"]
        results = {"mode": "dynamic", "success": res["success"],
                   "method": res.get("method"),
                   "values": res.get("values"),
                   "residuals": res.get("residuals"),
                   "maxResidual": res.get("maxResidual")}
        ts = res.get("series")
    else:
        res = solve_steady(model, info)
        messages += res["messages"]
        results = {"mode": "steady", "success": res["success"],
                   "method": res.get("method"),
                   "values": res.get("values"),
                   "residuals": res.get("residuals"),
                   "maxResidual": res.get("maxResidual")}
        ts = None

    return {"ok": True, "messages": messages, "model": info["model"],
            "results": results, "timeSeries": ts}
