"""Robustness tests: malformed / extreme inputs must never crash —
they must always return a structured result with diagnostic messages."""
from backend.engine import run_model

from conftest import has_msg


def test_empty_code():
    r = run_model("")
    assert r["ok"] is False
    assert has_msg(r, "err_empty", "error")


def test_whitespace_only():
    r = run_model("   \n\t  ")
    assert has_msg(r, "err_empty", "error")


def test_model_without_end():
    r = run_model("MODEL X\nVARIABLE\na AS REAL\n")
    assert has_msg(r, "parse_error", "error")
    assert "line" in r["messages"][0]["args"]


def test_no_equations():
    r = run_model("MODEL X\nVARIABLE\na AS REAL\nEND")
    assert has_msg(r, "err_no_equations", "error")
    assert r["results"] is None


def test_garbage_characters():
    r = run_model("MODEL X\nEQUATION\na = 1 ; @@@ ;\nEND")
    assert has_msg(r, "parse_error", "error")


def test_unbalanced_parens_in_equation():
    r = run_model("MODEL X\nVARIABLE\ny AS REAL\nEQUATION\ny = (1 + 2 ;\nEND")
    assert has_msg(r, "parse_error", "error")


def test_if_without_end():
    r = run_model("MODEL X\nVARIABLE\na AS REAL\nEQUATION\nIF a > 0 THEN a = 1 ; ELSE a = 2 ;\nEND")
    assert has_msg(r, "parse_error", "error")


def test_unsupported_section():
    r = run_model("MODEL X\nSOLUTIONPARAMETER\na AS REAL\nEND")
    assert has_msg(r, "err_unsupported_section", "error")


def test_division_by_zero_parameter_no_crash():
    # 1/a with a=0: the safe-division guard keeps the solve finite
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 0.0
  VARIABLE
    y AS REAL
  EQUATION
    y = 1 / a ;
END
""")
    assert r["ok"] is True
    assert r["results"] is not None          # may "solve" to a huge value, but no crash


def test_log_of_negative_no_crash():
    r = run_model("""
MODEL M
  VARIABLE
    y AS REAL
    x AS REAL
  EQUATION
    y = LN(x) ;
    x = -1 ;
END
""")
    assert r["ok"] is True
    assert r["results"] is not None


def test_too_many_steps_blocked():
    r = run_model("""
MODEL M
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
    x AS REAL
  EQUATION
    $x = -a * x ;
  INITIAL
    x = 1.0 ;
  SCHEDULE
    SEQUENCE
      CONTINUE FOR t := 0 TO 100000 STEP 0.001
    END
END
""")
    assert has_msg(r, "err_too_many_steps", "error")
    assert r["results"] is None


def test_huge_numbers_in_equations():
    r = run_model("""
MODEL M
  VARIABLE
    x AS REAL
  EQUATION
    x = 1.0E300 * 1.0E-300 ;
END
""")
    assert r["ok"] is True
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["x"] - 1.0) < 1e-9


def test_greek_and_unicode_identifiers():
    r = run_model("""
MODEL Εναλλάκτης
  PARAMETER
    ΔΤ AS REAL DEFAULT 25.0
  VARIABLE
    Θ AS REAL
  EQUATION
    Θ = ΔΤ + 5 ;
END
""")
    assert r["ok"] is True
    assert r["results"]["success"]
    assert abs(r["results"]["values"]["θ"] - 30.0) < 1e-9


def test_long_expression_chain():
    n = 50
    decls = "\n".join(f"    x{i} AS REAL" for i in range(n))
    identities = "\n".join(f"    x{i} = a ;" for i in range(1, n))
    terms = " + ".join(f"x{i}" for i in range(n))
    r = run_model(f"""
MODEL Big
  PARAMETER
    a AS REAL DEFAULT 1.0
  VARIABLE
{decls}
  EQUATION
    {terms} = a * {n} ;
{identities}
END
""")
    assert r["ok"] is True
    assert r["results"]["success"]
    v = r["results"]["values"]
    assert abs(v["x0"] - 1.0) < 1e-9
    assert abs(v["x49"] - 1.0) < 1e-9


def test_fuzz_random_garbage_never_raises():
    import random
    random.seed(42)
    alphabet = "MODEL END VARIABLE EQUATION a b x y = ; ( ) + - * / $ IF THEN 1.0 #\n"
    for _ in range(60):
        length = random.randint(10, 200)
        s = "".join(random.choice(alphabet.split()) for _ in range(length))
        r = run_model(s)
        assert isinstance(r, dict)
        assert "messages" in r
        assert isinstance(r["messages"], list)
