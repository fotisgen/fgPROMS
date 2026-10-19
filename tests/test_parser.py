"""Parser unit tests: tokens, precedence, declarations, errors."""
import pytest

from backend.parser.gproms_parser import (
    tokenize, Parser, parse_model, ParseError, expr_str,
    Num, Var, Deriv, UnOp, BinOp, Func, EqEquation, IfEquation,
)


def parse_eq(s: str):
    return Parser(s).parse_equation()


def parse_expr(s: str):
    return Parser(s).parse_expr()


# ---------------- tokenizer ----------------
def test_tokenize_numbers():
    toks = tokenize("1 2.5 .5 1e3 2.5E-2 3.")
    nums = [t.value for t in toks if t.kind == "NUMBER"]
    assert nums == ["1", "2.5", ".5", "1e3", "2.5E-2", "3."]


def test_tokenize_comments_skipped():
    toks = tokenize("# a comment\nx # trailing")
    assert [t.value for t in toks if t.kind != "EOF"] == ["x"]


def test_tokenize_derivative():
    toks = tokenize("$x")
    assert toks[0].kind == "DERIV"
    assert toks[1].kind == "IDENT" and toks[1].value == "x"


def test_tokenize_positions():
    toks = tokenize("a\n  b")
    assert toks[1].line == 2 and toks[1].col == 3


# ---------------- precedence ----------------
def test_mult_binds_tighter_than_plus():
    e = parse_expr("a + b * c")
    assert isinstance(e, BinOp) and e.op == "+"
    assert isinstance(e.r, BinOp) and e.r.op == "*"


def test_unary_minus_times():
    e = parse_expr("-k * CA^n")
    assert isinstance(e, BinOp) and e.op == "*"
    assert isinstance(e.l, UnOp) and e.l.op == "-"
    assert isinstance(e.r, BinOp) and e.r.op == "^"


def test_power_right_associative():
    e = parse_expr("a^b^c")
    assert isinstance(e, BinOp) and e.op == "^"
    assert isinstance(e.r, BinOp) and e.r.op == "^"


def test_unary_minus_binds_below_power():
    e = parse_expr("-x^2")
    assert isinstance(e, UnOp) and e.op == "-"
    assert isinstance(e.x, BinOp) and e.x.op == "^"


def test_parens_override():
    e = parse_expr("(a + b) * c")
    assert isinstance(e, BinOp) and e.op == "*"
    assert isinstance(e.l, BinOp) and e.l.op == "+"


def test_comparison_lowest():
    e = parse_expr("a + b > c")
    assert isinstance(e, BinOp) and e.op == ">"


# ---------------- equations ----------------
def test_equation_split():
    e = parse_eq("a = b + c ;")
    assert isinstance(e, EqEquation) and e.op == "="
    assert isinstance(e.lhs, Var) and e.lhs.name == "a"


def test_inequality_flagged():
    e = parse_eq("a < b ;")
    assert isinstance(e, EqEquation) and e.op == "<"


def test_derivative_equation():
    e = parse_eq("$h = (Fin - Fout) / A ;")
    assert isinstance(e.lhs, Deriv) and e.lhs.name == "h"


def test_if_equation_branches():
    e = parse_eq("IF T > 300 THEN a = 1 ; ELSE a = 2 ; END ;")
    assert isinstance(e, IfEquation)
    assert len(e.then_eqs) == 1 and len(e.else_eqs) == 1


def test_if_equation_multiple_in_branch():
    e = parse_eq("IF c THEN a = 1 ; b = 2 ; ELSE a = 3 ; b = 4 ; END ;")
    assert len(e.then_eqs) == 2 and len(e.else_eqs) == 2


def test_expr_str_rendering():
    m = parse_model("MODEL M\nVARIABLE\nCA AS REAL\nk AS REAL\nEQUATION\n$CA = -k * CA^n ;\nEND")
    text = expr_str(m.equations[0].rhs)
    assert text == "-k * CA ^ n"


# ---------------- declarations ----------------
def test_multi_name_declaration():
    p = Parser("A, B AS REAL")
    decls = p.parse_decls()
    assert decls[0].names == ["A", "B"]


def test_default_value_parsed():
    p = Parser("k AS REAL DEFAULT 0.5")
    decls = p.parse_decls()
    assert isinstance(decls[0].default, Num) and decls[0].default.v == 0.5


def test_type_text_parsed():
    p = Parser("T AS Temperature\nP AS ARRAY(10) OF REAL")
    decls = p.parse_decls()
    assert decls[0].type_text == "Temperature"
    assert decls[1].type_text == "ARRAY(10) OF REAL"


def test_case_insensitive_keywords():
    m = parse_model("model x\nvariable\ny as real\nequation\ny = 1 ;\nend")
    assert m.name == "x"
    assert len(m.variables) == 1 and len(m.equations) == 1


def test_greek_identifiers():
    m = parse_model("MODEL Θ\nVARIABLE\nθ AS REAL\nEQUATION\nθ = 25 ;\nEND")
    assert m.variables[0].names == ["θ"]


# ---------------- errors ----------------
def test_missing_end_raises_with_position():
    with pytest.raises(ParseError) as e:
        parse_model("MODEL X\nVARIABLE\na AS REAL\n")
    assert e.value.line >= 1


def test_bad_character_raises():
    with pytest.raises(ParseError):
        parse_model("MODEL X\nEQUATION\na = 1 ; @ ;\nEND")


def test_missing_semicolon_raises():
    with pytest.raises(ParseError):
        parse_model("MODEL X\nVARIABLE\na AS REAL\nEQUATION\na = 1\nb = 2 ;\nEND")


def test_equation_without_equals_raises():
    with pytest.raises(ParseError):
        parse_eq("a b ;")


def test_unbalanced_parens_raises():
    with pytest.raises(ParseError):
        parse_expr("(1 + 2")


def test_unsupported_section_raises_code():
    with pytest.raises(ParseError) as e:
        parse_model("MODEL X\nSELECTOR\nA AS REAL\nEND")
    assert e.value.code == "err_unsupported_section"
