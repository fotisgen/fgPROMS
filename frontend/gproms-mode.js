/* CodeMirror mode: gPROMS language (fgPROMS subset) */
CodeMirror.defineMode("gproms", function () {
  const KW = [
    "MODEL", "END", "PARAMETER", "VARIABLE", "EQUATION", "INITIAL", "SCHEDULE",
    "SET", "DEFAULT", "IF", "THEN", "ELSE", "CASE", "SWITCH", "CONTINUE",
    "FOR", "TO", "STEP", "SEQUENCE", "AS", "TRUE", "FALSE", "AND", "OR",
    "NOT", "REAL", "INTEGER", "LOGICAL", "OPTIMIZE", "MAXIMIZE", "MINIMIZE",
    "BOUNDS",
  ];
  const FUNC = ["EXP", "LN", "LOG", "SQRT", "ABS", "SIN", "COS", "TAN",
                "SIGN", "MAX", "MIN", "SIGMA"];
  return {
    token: function (stream) {
      if (stream.eatSpace()) return null;
      if (stream.match(/^#.*/)) return "comment";
      if (stream.match(/^\$[A-Za-z_\u0370-\u03ff]/)) return "variable-2";
      if (stream.match(/^[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?/)) return "number";
      if (stream.match(/^[A-Za-z_\u0370-\u03ff][A-Za-z0-9_\u0370-\u03ff]*/)) {
        const w = stream.current().toUpperCase();
        if (KW.indexOf(w) >= 0) return "keyword";
        if (FUNC.indexOf(w) >= 0) return "builtin";
        return "variable";
      }
      stream.next();
      return null;
    },
  };
});
CodeMirror.defineMIME("text/x-gproms", "gproms");
