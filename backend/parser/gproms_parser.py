"""Parser for the fgPROMS language — a practical subset of the gPROMS
modelling language.

Supported subset (Phase 1):
  * MODEL name ... END
  * PARAMETER / VARIABLE declarations with AS <type> and optional DEFAULT
  * SET section            (name := value ;)
  * EQUATION section       (expr = expr ;  and  IF cond THEN ... ELSE ... END)
  * INITIAL section        (name = value ;)
  * SCHEDULE section       (SEQUENCE CONTINUE FOR t := t0 TO t1 STEP h END)
  * '#' comments, '$var' time derivatives, standard math functions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ParseError(Exception):
    def __init__(self, msg: str, line: int = 0, col: int = 0, code: str = "parse_error"):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.col = col
        self.code = code


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------
@dataclass
class Token:
    kind: str          # 'IDENT' | 'NUMBER' | 'OP' | 'DERIV' | 'EOF'
    value: str
    line: int
    col: int


SECTION_KEYWORDS = {
    "PARAMETER", "VARIABLE", "EQUATION", "INITIAL", "SCHEDULE", "SET",
    "OPTIMIZE", "BOUNDS",
    "SELECTOR", "DISTRIBUTION_DOMAIN", "UNIT", "PROCESS", "STREAM",
    "SOLUTIONPARAMETER",
}

RESERVED = SECTION_KEYWORDS | {
    "MODEL", "END", "DEFAULT", "IF", "THEN", "ELSE", "CASE", "SWITCH", "WHEN",
    "CONTINUE", "FOR", "TO", "STEP", "SEQUENCE", "AS", "REAL", "INTEGER",
    "LOGICAL", "TRUE", "FALSE", "AND", "OR", "NOT", "OF", "WITHIN",
    "MAXIMIZE", "MINIMIZE", "MAXIMISE", "MINIMISE",
}


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(text)
    line, col = 1, 1

    def advance(k: int):
        nonlocal i, col
        i += k
        col += k

    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            col = 1
            i += 1
            continue
        if c in " \t\r":
            advance(1)
            continue
        if c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c in "\"'":
            quote = c
            j = i + 1
            while j < n and text[j] != quote:
                j += 1
            if j >= n:
                raise ParseError("Unterminated string literal", line, col)
            tokens.append(Token("STRING", text[i + 1:j], line, col))
            advance(j - i + 1)
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and text[j].isdigit():
                j += 1
            if j < n and text[j] == ".":
                j += 1
                while j < n and text[j].isdigit():
                    j += 1
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k < n and text[k].isdigit():
                    j = k
                    while j < n and text[j].isdigit():
                        j += 1
            tokens.append(Token("NUMBER", text[i:j], line, col))
            advance(j - i)
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token("IDENT", text[i:j], line, col))
            advance(j - i)
            continue
        if c == "$":
            tokens.append(Token("DERIV", "$", line, col))
            advance(1)
            continue
        two = text[i:i + 2]
        if two in (":=", "<>", "<=", ">=", "**"):
            tokens.append(Token("OP", two, line, col))
            advance(2)
            continue
        if c in "+-*/^()[]{}.,;:=<>":
            tokens.append(Token("OP", c, line, col))
            advance(1)
            continue
        raise ParseError(f"Unexpected character {c!r}", line, col)

    tokens.append(Token("EOF", "", line, col))
    return tokens


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------
class Node:
    pass


@dataclass
class Num(Node):
    v: float


@dataclass
class Str(Node):
    value: str


@dataclass
class Var(Node):
    name: str


@dataclass
class Deriv(Node):          # $name  ->  d(name)/dt
    name: str


@dataclass
class UnOp(Node):
    op: str                 # '-' | 'NOT'
    x: Node


@dataclass
class BinOp(Node):
    op: str                 # + - * / ^ ** AND OR = <> < <= > >=
    l: Node
    r: Node


@dataclass
class Func(Node):
    name: str
    args: List[Node]


@dataclass
class IfExpr(Node):
    cond: Node
    then: Node
    els: Node


@dataclass
class Equation:
    pass


@dataclass
class EqEquation(Equation):
    lhs: Node
    rhs: Node
    op: str = "="
    leaf_id: int = -1       # filled in by the analyzer


@dataclass
class IfEquation(Equation):
    cond: Node
    then_eqs: List[Equation]
    else_eqs: List[Equation]


@dataclass
class Decl:
    names: List[str]
    type_text: str
    default: Optional[Node]
    line: int


@dataclass
class Assignment:
    name: str
    value: Node
    line: int


@dataclass
class ScheduleStep:
    var: str
    t0: Node
    t1: Node
    step: Optional[Node]


@dataclass
class Objective:
    kind: str                 # 'maximize' | 'minimize'
    expr: Node
    line: int


@dataclass
class Bound:
    name: str
    lo: Node
    hi: Node
    line: int


@dataclass
class UnitDecl:
    name: str
    type_name: str
    sets: List[Assignment]
    line: int


@dataclass
class ProcessDef:
    name: str
    line: int
    units: List[UnitDecl] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    sets: List[Assignment] = field(default_factory=list)
    initial: List[Assignment] = field(default_factory=list)
    schedule: List[ScheduleStep] = field(default_factory=list)


@dataclass
class FileDef:
    models: dict                        # name_lower -> Model
    process: Optional[ProcessDef] = None


@dataclass
class Model:
    name: str
    line: int
    parameters: List[Decl] = field(default_factory=list)
    variables: List[Decl] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    initial: List[Assignment] = field(default_factory=list)
    sets: List[Assignment] = field(default_factory=list)
    schedule: List[ScheduleStep] = field(default_factory=list)
    objective: Optional[Objective] = None
    bounds: List[Bound] = field(default_factory=list)


# --------------------------------------------------------------------------
# Expression pretty-printer
# --------------------------------------------------------------------------
_BIN_PREC = {
    "OR": 1, "AND": 2,
    "=": 3, "<>": 3, "<": 3, "<=": 3, ">": 3, ">=": 3,
    "+": 4, "-": 4, "*": 5, "/": 5, "^": 6, "**": 6,
}
_SYM = {"**": "^"}


def _join_type(toks: List[Token]) -> str:
    """Join type tokens into a readable type string, e.g.
    'ARRAY(10) OF REAL' instead of 'ARRAY ( 10 ) OF REAL'."""
    out = ""
    for i, t in enumerate(toks):
        v = t.value
        if i == 0:
            out = v
        elif v in ")]" or v == "(":
            out += v
        elif toks[i - 1].value in "([":
            out += v
        else:
            out += " " + v
    return out


def _num_str(v: float) -> str:
    if abs(v - round(v)) < 1e-12 and abs(v) < 1e15:
        return str(int(round(v)))
    return format(v, ".6g")


def expr_str(n: Node, outer: float = 0.0) -> str:
    if isinstance(n, Num):
        s = _num_str(n.v)
        return f"({s})" if n.v < 0 and outer < 7 else s
    if isinstance(n, Str):
        return f"'{n.value}'"
    if isinstance(n, Var):
        return n.name
    if isinstance(n, Deriv):
        return "$" + n.name
    if isinstance(n, UnOp):
        if n.op == "NOT":
            return f"NOT {expr_str(n.x, 2)}"
        return "-" + expr_str(n.x, 7)
    if isinstance(n, BinOp):
        p = _BIN_PREC[n.op]
        if n.op in ("^", "**"):
            l, r = expr_str(n.l, p), expr_str(n.r, p)
        else:
            l, r = expr_str(n.l, p), expr_str(n.r, p + 1)
        s = f"{l} {_SYM.get(n.op, n.op)} {r}"
        return f"({s})" if p < outer else s
    if isinstance(n, Func):
        return f"{n.name}({', '.join(expr_str(a) for a in n.args)})"
    if isinstance(n, IfExpr):
        return f"IF {expr_str(n.cond)} THEN {expr_str(n.then)} ELSE {expr_str(n.els)} END"
    return "?"


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
class Parser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.pos = 0

    # -- token helpers -----------------------------------------------------
    def peek(self, k: int = 0) -> Token:
        idx = min(self.pos + k, len(self.tokens) - 1)
        return self.tokens[idx]

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def at_op(self, op: str) -> bool:
        t = self.peek()
        return t.kind == "OP" and t.value == op

    def at_kw(self, word: str) -> bool:
        t = self.peek()
        return t.kind == "IDENT" and t.value.upper() == word

    def expect_op(self, op: str) -> Token:
        t = self.peek()
        if t.kind == "OP" and t.value == op:
            return self.next()
        raise ParseError(f"Expected '{op}' but found '{t.value}'", t.line, t.col)

    def expect_kw(self, word: str) -> Token:
        t = self.peek()
        if t.kind == "IDENT" and t.value.upper() == word:
            return self.next()
        raise ParseError(f"Expected '{word}' but found '{t.value}'", t.line, t.col)

    def expect_ident(self) -> Token:
        t = self.peek()
        if t.kind == "IDENT":
            return self.next()
        raise ParseError(f"Expected an identifier but found '{t.value}'", t.line, t.col)

    def at_boundary(self) -> bool:
        t = self.peek()
        if t.kind == "EOF" or t.kind == "END":
            return True
        if t.kind == "IDENT" and t.value.upper() in RESERVED:
            return True
        return False

    # -- model --------------------------------------------------------------
    def parse(self) -> Model:
        self.expect_kw("MODEL")
        return self.parse_model_body()

    def parse_file(self) -> FileDef:
        """Parse a source file that may contain multiple MODEL definitions
        plus an optional PROCESS flowsheet."""
        models: dict = {}
        process: Optional[ProcessDef] = None
        while self.peek().kind != "EOF":
            t = self.peek()
            if t.kind == "IDENT" and t.value.upper() == "MODEL":
                self.next()
                m = self.parse_model_body()
                models[m.name.lower()] = m
            elif t.kind == "IDENT" and t.value.upper() == "PROCESS":
                self.next()
                process = self.parse_process_body()
            else:
                raise ParseError(f"Expected MODEL or PROCESS, found '{t.value}'",
                                 t.line, t.col)
        if not models and not process:
            tok = self.peek()
            raise ParseError("The file contains no MODEL or PROCESS",
                             tok.line, tok.col)
        return FileDef(models, process)

    def parse_model_body(self) -> Model:
        name = self.expect_ident()
        model = Model(name=name.value, line=name.line)

        while True:
            t = self.peek()
            if t.kind == "EOF":
                raise ParseError("Unexpected end of file — expected END", t.line, t.col)
            if t.kind == "IDENT":
                u = t.value.upper()
                if u == "END":
                    self.next()
                    break
                if u == "PARAMETER":
                    self.next()
                    model.parameters = self.parse_decls()
                elif u == "VARIABLE":
                    self.next()
                    model.variables = self.parse_decls()
                elif u == "EQUATION":
                    self.next()
                    model.equations = self.parse_eq_list()
                elif u == "INITIAL":
                    self.next()
                    model.initial = self.parse_assignments()
                elif u == "SET":
                    self.next()
                    model.sets = self.parse_assignments()
                elif u == "SCHEDULE":
                    self.next()
                    model.schedule = self.parse_schedule()
                elif u == "OPTIMIZE":
                    self.next()
                    model.objective = self.parse_objective()
                elif u == "BOUNDS":
                    self.next()
                    model.bounds = self.parse_bounds()
                elif u in SECTION_KEYWORDS:
                    raise ParseError(
                        f"Section '{u}' is not supported by fgPROMS (Phase 1)",
                        t.line, t.col, code="err_unsupported_section")
                elif u == "MODEL":
                    raise ParseError("Nested MODEL declarations are not supported",
                                     t.line, t.col)
                else:
                    raise ParseError(f"Unexpected identifier '{t.value}'", t.line, t.col)
            else:
                raise ParseError(f"Unexpected token '{t.value}'", t.line, t.col)

        return model

    # -- process (flowsheet) ---------------------------------------------------
    def parse_process_body(self) -> ProcessDef:
        name = self.expect_ident()
        p = ProcessDef(name=name.value, line=name.line)
        while True:
            t = self.peek()
            if t.kind == "EOF":
                raise ParseError("Unexpected end of file — expected END",
                                 t.line, t.col)
            if t.kind == "IDENT":
                u = t.value.upper()
                if u == "END":
                    self.next()
                    break
                if u == "UNIT":
                    self.next()
                    p.units = self.parse_units()
                elif u == "EQUATION":
                    self.next()
                    p.equations = self.parse_eq_list()
                elif u == "SET":
                    self.next()
                    p.sets = self.parse_assignments()
                elif u == "INITIAL":
                    self.next()
                    p.initial = self.parse_assignments()
                elif u == "SCHEDULE":
                    self.next()
                    p.schedule = self.parse_schedule()
                else:
                    raise ParseError(f"Unexpected identifier '{t.value}' "
                                     f"in PROCESS", t.line, t.col)
            else:
                raise ParseError(f"Unexpected token '{t.value}'", t.line, t.col)
        return p

    def parse_units(self) -> List[UnitDecl]:
        units: List[UnitDecl] = []
        while (self.peek().kind == "IDENT"
               and self.peek().value.upper() not in RESERVED):
            name_tok = self.next()
            self.expect_kw("AS")
            type_tok = self.expect_ident()
            sets: List[Assignment] = []
            # unit-level SET is recognised only when it is followed by an
            # undotted assignment (`SET V := 20`); a PROCESS-level SET
            # section (`SET R1.V := 20`) is left for the process body
            if (self.at_kw("SET") and self.peek(1).kind == "IDENT"
                    and self.peek(2).kind == "OP"
                    and self.peek(2).value in (":=", "=")):
                self.next()
                while (self.peek().kind == "IDENT"
                       and self.peek().value.upper() not in RESERVED
                       and self.peek(1).kind == "OP"
                       and self.peek(1).value in (":=", "=")):
                    n_tok = self.next()
                    t2 = self.peek()
                    if t2.kind == "OP" and t2.value in (":=", "="):
                        self.next()
                    else:
                        raise ParseError("Expected ':=' in unit SET",
                                         t2.line, t2.col)
                    v = self.parse_expr()
                    if self.at_op(";"):
                        self.next()
                    sets.append(Assignment(n_tok.value, v, n_tok.line))
            units.append(UnitDecl(name_tok.value, type_tok.value, sets,
                                  name_tok.line))
        return units

    # -- declarations --------------------------------------------------------
    def parse_decls(self) -> List[Decl]:
        decls: List[Decl] = []
        while self.peek().kind == "IDENT" and self.peek().value.upper() not in RESERVED:
            line = self.peek().line
            names: List[str] = []
            while True:
                tok = self.expect_ident()
                names.append(tok.value)
                if self.at_op(","):
                    self.next()
                    continue
                break
            self.expect_kw("AS")
            type_toks: List[Token] = []
            while True:
                t = self.peek()
                if t.kind == "EOF":
                    break
                if t.kind == "OP" and t.value == ";":
                    break
                if t.kind == "IDENT":
                    u = t.value.upper()
                    if u == "DEFAULT":
                        break
                    if u in SECTION_KEYWORDS or u == "END":
                        break
                    p1 = self.peek(1)
                    if p1.kind == "IDENT" and p1.value.upper() == "AS":
                        break
                type_toks.append(self.next())
            type_text = _join_type(type_toks).strip() or "REAL"
            default = None
            if self.at_kw("DEFAULT"):
                self.next()
                default = self.parse_expr()
            if self.at_op(";"):
                self.next()
            decls.append(Decl(names, type_text, default, line))
        return decls

    # -- assignments (INITIAL / SET) ------------------------------------------
    def parse_assignments(self) -> List[Assignment]:
        assigns: List[Assignment] = []
        while self.peek().kind == "IDENT" and self.peek().value.upper() not in RESERVED:
            tok = self.next()
            line = tok.line
            name = tok.value
            # dotted name: Unit.parameter (PROCESS-level SET)
            while self.at_op(".") and self.peek(1).kind == "IDENT":
                self.next()
                name += "." + self.next().value
            t = self.peek()
            if t.kind == "OP" and t.value in (":=", "="):
                self.next()
            else:
                raise ParseError("Expected ':=' or '=' in assignment", t.line, t.col)
            value = self.parse_expr()
            if self.at_op(";"):
                self.next()
            assigns.append(Assignment(name, value, line))
        return assigns

    # -- equations -------------------------------------------------------------
    def parse_eq_list(self) -> List[Equation]:
        eqs: List[Equation] = []
        while True:
            if self.at_kw("ELSE") or self.at_kw("END") or self.peek().kind == "EOF":
                break
            if self.peek().kind == "IDENT" and self.peek().value.upper() in SECTION_KEYWORDS:
                break
            eqs.append(self.parse_equation())
            self.expect_op(";")
        return eqs

    def parse_equation(self) -> Equation:
        if self.at_kw("IF"):
            self.next()
            cond = self.parse_expr()
            self.expect_kw("THEN")
            then_eqs = self.parse_eq_list()
            else_eqs: List[Equation] = []
            if self.at_kw("ELSE"):
                self.next()
                else_eqs = self.parse_eq_list()
            self.expect_kw("END")
            return IfEquation(cond, then_eqs, else_eqs)
        lhs = self.parse_expr()
        if isinstance(lhs, BinOp) and lhs.op in ("=", "<>", "<", "<=", ">", ">="):
            return EqEquation(lhs.l, lhs.r, lhs.op)
        t = self.peek()
        if t.kind == "OP" and t.value in ("=", "<>", "<", "<=", ">", ">="):
            self.next()
            rhs = self.parse_expr()
            return EqEquation(lhs, rhs, t.value)
        raise ParseError("An equation must contain '='", t.line, t.col)

    # -- schedule ----------------------------------------------------------------
    def parse_schedule(self) -> List[ScheduleStep]:
        steps: List[ScheduleStep] = []
        if self.at_kw("SEQUENCE"):
            self.next()
        while not self.at_kw("END"):
            if self.peek().kind == "EOF":
                raise ParseError("Unexpected end of file inside SCHEDULE",
                                 self.peek().line, self.peek().col)
            self.expect_kw("CONTINUE")
            self.expect_kw("FOR")
            var = self.expect_ident()
            self.expect_op(":=")
            t0 = self.parse_expr()
            self.expect_kw("TO")
            t1 = self.parse_expr()
            step = None
            if self.at_kw("STEP"):
                self.next()
                step = self.parse_expr()
            steps.append(ScheduleStep(var.value, t0, t1, step))
        self.expect_kw("END")
        return steps

    # -- objective / bounds (optimization) ----------------------------------------
    def parse_objective(self) -> Objective:
        t = self.peek()
        if t.kind == "IDENT" and t.value.upper() in ("MAXIMIZE", "MINIMIZE",
                                                     "MAXIMISE", "MINIMISE"):
            kind = "maximize" if t.value.upper().startswith("MAX") else "minimize"
            self.next()
            expr = self.parse_expr()
            if self.at_op(";"):
                self.next()
            return Objective(kind, expr, t.line)
        raise ParseError("Expected MAXIMIZE or MINIMIZE", t.line, t.col)

    def parse_bounds(self) -> List[Bound]:
        bounds: List[Bound] = []
        while (self.peek().kind == "IDENT"
               and self.peek().value.upper() not in RESERVED):
            name_tok = self.next()
            self.expect_op(":")
            lo = self.parse_expr()
            self.expect_op(":")
            hi = self.parse_expr()
            if self.at_op(";"):
                self.next()
            bounds.append(Bound(name_tok.value, lo, hi, name_tok.line))
        return bounds

    # -- expressions (precedence climbing) --------------------------------------
    def parse_expr(self, min_prec: float = 0.0) -> Node:
        t = self.peek()
        if t.kind == "OP" and t.value in ("-", "NOT"):
            self.next()
            left = UnOp(t.value.upper(), self.parse_expr(6))
        elif t.kind == "OP" and t.value == "+":
            self.next()
            left = self.parse_expr(6)
        elif t.kind == "IDENT" and t.value.upper() == "IF":
            self.next()
            cond = self.parse_expr()
            self.expect_kw("THEN")
            a = self.parse_expr()
            self.expect_kw("ELSE")
            b = self.parse_expr()
            self.expect_kw("END")
            left = IfExpr(cond, a, b)
        else:
            left = self.parse_primary()
        while True:
            t = self.peek()
            if t.kind == "OP" and t.value.upper() in _BIN_PREC:
                prec = _BIN_PREC[t.value.upper()]
                if prec < min_prec:
                    break
                self.next()
                rprec = prec if t.value in ("^", "**") else prec + 1
                right = self.parse_expr(rprec)
                left = BinOp(t.value.upper(), left, right)
            else:
                break
        return left

    def parse_primary(self) -> Node:
        t = self.peek()
        if t.kind == "NUMBER":
            self.next()
            try:
                return Num(float(t.value))
            except ValueError:
                raise ParseError(f"Bad number '{t.value}'", t.line, t.col)
        if t.kind == "STRING":
            self.next()
            return Str(t.value)
        if t.kind == "DERIV":
            self.next()
            name = self.expect_ident()
            return Deriv(name.value)
        if t.kind == "IDENT":
            u = t.value.upper()
            if u == "TRUE":
                self.next()
                return Num(1.0)
            if u == "FALSE":
                self.next()
                return Num(0.0)
            self.next()
            if self.at_op("("):
                self.next()
                args: List[Node] = []
                if not self.at_op(")"):
                    while True:
                        args.append(self.parse_expr())
                        if self.at_op(","):
                            self.next()
                            continue
                        break
                self.expect_op(")")
                return Func(u, args)
            # dotted name: unit.variable (flowsheet connections)
            name = t.value
            while self.at_op(".") and self.peek(1).kind == "IDENT":
                self.next()
                name += "." + self.next().value
            return Var(name)
        if t.kind == "OP" and t.value == "(":
            self.next()
            x = self.parse_expr()
            self.expect_op(")")
            return x
        raise ParseError(f"Unexpected token '{t.value}'", t.line, t.col)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def parse_model(text: str) -> Model:
    """Parse fgPROMS source text into a single Model AST."""
    return Parser(text).parse()


def parse_file(text: str) -> FileDef:
    """Parse fgPROMS source text into a FileDef (models + optional process)."""
    return Parser(text).parse_file()
