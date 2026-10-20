# Πίνακας κάλυψης gPROMS

Αυτό το αρχείο είναι η **μετρημένη**, ελεγξιμη μήτρα κάλυψης του fgPROMS
έναντι της λειτουργικής επιφάνειας του gPROMS.

Το τεστ `tests/test_coverage.py` διαβάζει αυτόν τον πίνακα και υπολογίζει
αυτόματα το σταθμισμένο ποσοστό — αν η μήτρα αλλάξει χωρίς να ενημερωθεί
ο δηλωμένος αριθμός, το τεστ αποτυγχάνει.

## Συμβάσεις

* Κάθε ομάδα δηλώνεται ως `## Ομάδα Χ — τίτλος (βάρος NN%)`.
* Κάθε γραμμή πίνακα ξεκινά με την κατάσταση στην πρώτη στήλη:
  * `✅` πλήρες (1.0) — υλοποιημένο και αποδεδειγμένο με τεστ
  * `◐` μερικό (0.5) — υλοποιημένο μερικώς/απλοποιημένα
  * `✗` απών (0.0) — δεν υλοποιείται (και γιατί, όπου έχει σημασία)
* Η στήλη «Απόδειξη» δείχνει πού αποδεικνύεται κάθε στοιχείο.

Τρέχουσα συνολική κάλυψη: **69%**

## Ομάδα Α — Γλώσσα μοντελοποίησης (βάρος 35%)

| | Χαρακτηριστικό | Απόδειξη |
|---|---|---|
| ✅ | Ενότητες PARAMETER / VARIABLE / EQUATION | `tests/test_parser.py` |
| ✅ | SET / DEFAULT / INITIAL / BOUNDS | `tests/test_analyzer.py`, `tests/test_optimization.py` |
| ◐ | Τύποι INTEGER / LOGICAL (δηλώνονται, χωρίς πλήρη σημασιολογία) | parser, `backend/units.py` |
| ✅ | Πίνακες ARRAY πολυδιάστατοι | `tests/test_multidim.py` |
| ✅ | Κατανεμημένα DISTRIBUTION 1D/2D | `tests/test_distributed.py` |
| ✅ | Αριθμητικές παραστάσεις + EXP/LN/SQRT/SIN/COS/TAN/ABS/SIGN/LOG/MAX/MIN | `tests/test_parser.py` |
| ✅ | Εξισώσεις IF/THEN/ELSE και IF εκφράσεις | `tests/test_analyzer.py`, `tests/test_robustness.py` |
| ✅ | Βρόχοι FOR σε εξισώσεις | `tests/test_envelope.py` |
| ✅ | TASK / PHASE / SCHEDULE / END WHEN | `tests/test_phases.py` |
| ◐ | WITHIN (χρονικά παράθυρα στο SCHEDULE — συντακτική υποστήριξη) | `tests/test_language_extras.py` |
| ✅ | CASE / SWITCH εξισώσεις (επέκταση σε IF-αλυσίδα) | `tests/test_case_switch.py` |
| ✅ | SELECTOR (επιλογή EOS backend σε όλα τα property calls) | `tests/test_language_extras.py` |
| ◐ | REINITIALIZE (RESTART όχι ακόμη) | `tests/test_reinitialize.py` |
| ✗ | PIECEWISE γραμμικές εξισώσεις / πίνακες | — |
| ✗ | ORDERED_SET | — |
| ✅ | Τύποι STRING (παραμετροποίηση ρευστών κ.λπ.) | `tests/test_language_extras.py` |
| ◐ | Μονάδες: διαστατικός έλεγχος κάθε εξίσωσης (χωρίς μετατροπές) | `tests/test_units.py` |
| ✗ | FOREIGN_OBJECT / FOREIGN_PROCESS (αρχιτεκτονικά εκτός σκοπού) | — |
| ✅ | Τελεστές PARTIAL / INT | `tests/test_distributed.py` |
| ✅ | STREAM AS MATERIAL με ιδιότητες H/CP/RHO | `tests/test_streams.py` |

## Ομάδα Β — Αριθμητικά (βάρος 30%)

| | Χαρακτηριστικό | Απόδειξη |
|---|---|---|
| ✅ | Newton με αναλυτικό Jacobian (SymPy) | `tests/test_solver_steady.py` |
| ✅ | Αραιοί Jacobians (structural sparsity) | `tests/test_distributed.py::test_2d_jacobian_is_sparse` |
| ✅ | DAE ολοκλήρωση BDF 1–5 μεταβλητής τάξης με έλεγχο σφάλματος | `tests/test_solver_dynamic.py` |
| ✅ | Ανίχνευση γεγονότων με διχοτόμηση | `tests/test_phases.py` |
| ◐ | Παραμετρική εκτίμηση DATA/ESTIMATE (1 πείραμα, χωρίς στατιστικά) | `tests/test_estimation.py` |
| ✅ | Βελτιστοποίηση SLSQP + Ipopt (NLP rescue) | `tests/test_optimization.py`, `tests/test_highs_ipopt.py` |
| ✅ | LP / MILP επιλυτές (HiGHS) | `tests/test_highs_ipopt.py` |
| ◐ | Δυναμική βελτιστοποίηση (gOPT-lite: σταθερός έλεγχος) | `tests/test_dynamic_opt.py` |
| ✗ | High-index DAE (ενεργό ερευνητικό πεδίο — εκτός σκοπού) | — |
| ◐ | Κλιμάκωση (ισορρόπηση γραμμών, όχι πλήρης) | `backend/solver/solver.py` |
| ✗ | Εύρωστη αρχικοποίηση (multi-start, ομοτοπία) | — |
| ✅ | Αριθμητικό Jacobian fallback | `backend/solver/jacobian.py` |
| ✗ | Ανάλυση συστήματος (gO:GlobalSpec — εκτός σκοπού) | — |

## Ομάδα Γ — Θερμοδυναμική (βάρος 15%)

| | Χαρακτηριστικό | Απόδειξη |
|---|---|---|
| ✅ | Ιδιότητες καθαρών ρευστών (CoolProp) | `tests/test_properties.py` |
| ✅ | Μίγματα (CoolProp HEOS) | `tests/test_properties.py` |
| ✅ | Σύνθεση ως μεταβλητή: `CP('A&B', z, T, P)` | `tests/test_properties.py` (M25) |
| ✅ | Flash / Q / TBUBBLE / TDEW / PBUBBLE / PDEW | `tests/test_properties.py` |
| ✅ | Φάκελος φάσεων (bubble/dew καμπύλες) | `tests/test_envelope.py` |
| ✅ | Κυβικά EOS (PR/SRK) ως μοντέλα + fugacity + επιλογή backend | `tests/test_cubic_eos.py` |
| ✅ | Μοντέλα ενεργότητας (NRTL, Wilson) — γ-φ VLE, αζεότροπο | `tests/test_activity_models.py` |
| ✅ | Χημική ισορροπία (K-values, ελαχιστοποίηση Gibbs) | `tests/test_chemical_equilibrium.py` |
| ✅ | Διαγράμματα T-s / P-h δεδομένα + heatmap 2D στο GUI | `tests/test_cycles.py`, `tests/test_gui.py` |
| ✅ | Κύκλοι ισχύος (Rankine) με ισοεντροπικές επιλύσεις | `tests/test_cycles.py` |
| ✅ | Θερμοδυναμική συνέπεια (Maxwell) | `tests/test_thermo_consistency.py` |
| ✗ | gSAFT / Multiflash (εμπορικά — εκτός σκοπού) | — |
| ✗ | UNIFAC / custom property packages | — |
| ◐ | Ιδιότητες μεταφοράς (μόνο VISC) | `backend/properties.py` |
| ✗ | Στερεά / ηλεκτρολύτες | — |

## Ομάδα Δ — Flowsheets / βιβλιοθήκη μονάδων (βάρος 10%)

| | Χαρακτηριστικό | Απόδειξη |
|---|---|---|
| ✅ | PROCESS / UNIT / STREAM | `tests/test_flowsheet.py` |
| ✅ | Επίλυση συνδεσιμότητας + DOF | `tests/test_flowsheet.py` |
| ✅ | Βιβλιοθήκη: heater, mixer, splitter, pump, flash_drum | `tests/test_library.py`, `tests/test_thermo_consistency.py` |
| ◐ | Στήλη απόσταξης (McCabe-Thiele, σταθερή σχετική πτητικότητα) | `tests/test_distillation.py` |
| ✗ | Αντιδραστήρες PFR / σειρές CSTR | — |
| ✗ | Ελεγκτές PID | — |
| ✗ | Εναλλάκτες λεπτομερείς (UA, LMTD) | — |
| ◐ | Recycle/tearing (ταυτόχρονη επίλυση, χωρίς dedicated tearing) | `tests/test_flowsheet.py` |
| ✅ | Ισοζύγια μάζας/ενέργειας μονάδων | `tests/test_flowsheet.py` |

## Ομάδα Ε — Οικοσύστημα (βάρος 10%)

| | Χαρακτηριστικό | Απόδειξη |
|---|---|---|
| ✅ | Web GUI με γραφήματα (Plotly) | `tests/test_gui.py` |
| ✅ | Δίγλωσσο UI (Ελληνικά/English) | `frontend/i18n.js` |
| ✅ | Export CSV | `tests/test_gui.py` |
| ✅ | Διαχείριση projects | `tests/test_projects.py` |
| ✅ | Tutorials / τεκμηρίωση στα ελληνικά | `tests/test_tutorials.py` |
| ✅ | MCP server (Model Context Protocol) | `tests/test_mcp.py` |
| ✗ | Διεπαφή Excel / Matlab | — |
| ✗ | Cloud / multi-user | — |
| ✗ | Debugger / trace εξισώσεων | — |
| ✗ | Αναφορές (PDF) | — |

## Πώς υπολογίζεται το ποσοστό

Για κάθε ομάδα: `βάρος × (✅ + 0.5·◐) / (✅ + ◐ + ✗)`, άθροισμα ανά ομάδες.
Τα βάρη αθροίζουν στο 100%. Ο έλεγχος γίνεται από το
`tests/test_coverage.py` — η μήτρα και ο αριθμός είναι πάντα συνεπείς.
