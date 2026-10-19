# Αναφορά έρευνας αποθετηρίων (M1)

Ημερομηνία: 19/10/2026 · Ερευνητής: fgPROMS project
Κανόνας: κάθε στοιχείο χαρακτηρίζεται **[VERIFIED]** (διαβάστηκε απευθείας σε
LICENSE/README του repo) ή **[DOCUMENTED]** (από επίσημη τεκμηρίωση/γνώση,
χωρίς επιτόπια ανάγνωση κώδικα σε αυτή τη συνεδρία). Τίποτα δεν υπονοείται.

---

## 1. IDAES (idaes-pse)

**Τι είναι:** Framework μοντελοποίησης διεργασιών βασισμένο στο Pyomo, από
DOE/US ερευνητικά ιδρύματα (LBNL, Sandia, Carnegie Mellon κ.ά.).

### Στοιχεία
| Ερώτημα | Εύρημα |
|---|---|
| Αριθμητικές δυνατότητες | **[DOCUMENTED]** Μοντελοποίηση με Pyomo AML: μονάδες (unit models), ιδιότητες (property packages), flowsheets, DAE μέσω pyomo.dae, βελτιστοποίηση μέσω Pyomo solvers |
| Αλγεβρικές / ODE / DAE / PDE / βελτιστοποίηση | **[DOCUMENTED]** Ναι σε όλα (PDE μέσω διακριτοποίησης) |
| User-defined μοντέλα | **[DOCUMENTED]** Ναι — είναι framework, όχι κλειστό εργαλείο |
| Προγραμματική κλήση | **[VERIFIED]** Είναι Python package (`pip install idaes-pse`), προφανώς καλείται προγραμματικά |
| Εξαρτήσεις / solvers | **[VERIFIED, README]** `idaes get-extensions` κατεβάζει έτοιμα δυαδικά solvers (Ipopt κ.ά.). Optional extras: `ui`, `omlt`, `grid`, `coolprop` |
| Python υποστήριξη | **[VERIFIED, README]** 3.10 – 3.14, Windows 10 |
| Ωριμότητα | **[VERIFIED, README]** Μεγάλο ενεργό project (CI, codecov, discussions, εκδόσεις σε PyPI) |
| Άδεια | **[VERIFIED, LICENSE.md]** BSD-3-Clause-στυλ + ρήτρα παραχώρησης βελτιώσεων (enhancement grant). ΔΕΝ είναι LGPL (αρχική υπόθεση διόρθωθηκε) |

### Συμπέρασμα για το fgPROMS
- ✅ **Χρήσιμο ως προμηθευτής δυαδικών solvers** (`get-extensions`) αν ποτέ
  θελήσουμε Ipopt χωρίς compilation — χωρίς να εξαρτηθούμε από το framework.
- ✅ Πηγή έμπνευσης για **property packages** και συμβάσεις DOF.
- ❌ Όχι ως κύριος engine: θα μας δέσμευε στο Pyomo AML, βαρύ dependency tree,
  αχρείαστο για εκπαιδευτικό εργαλείο αυτής της κλίμακας.

---

## 2. Pyomo

**Τι είναι:** Algebraic Modeling Language (AML) σε Python — δηλωτική δόμηση
βελτιστοποίησης, με solver plugins (NLP/MINLP/LP) και pyomo.dae.

### Στοιχεία
| Ερώτημα | Εύρημα |
|---|---|
| Αριθμητικές δυνατότητες | **[DOCUMENTED]** Διαμόρφωση μοντέλων + σύνδεση σε εξωτερικούς solvers (Ipopt, GLPK, CBC…). Δεν περιέχει δικούς του NLP solvers (μόνο interfaces) |
| AEs / ODEs / DAEs | **[DOCUMENTED]** pyomo.dae: διακριτοποίηση DAE (collocation, finite differences) |
| Βελτιστοποίηση | **[VERIFIED από φύση του εργαλείου]** Είναι ο σκοπός του |
| Προγραμματική κλήση | **[VERIFIED]** Python package |
| Εξαρτήσεις | **[VERIFIED, LICENSE.md notices]** PLY (BSD), PyUtilib (BSD), + προαιρετικοί solvers |
| Ωριμότητα | Πολύ ώριμο (Sandia, ευρεία χρήση) |
| Άδεια | **[VERIFIED, LICENSE.md]** BSD-3-Clause (Sandia) + notices για ενσωματωμένα BSD-στυλ modules |

### Συμπέρασμα για το fgPROMS
- ✅ **Υποψήφιο για το μελλοντικό optimization engine** (μεταφραστής
  gPROMS AST → Pyomo ConcreteModel) αν χρειαστούμε πραγματικό NLP solver
  (Ipopt μέσω IDAES binaries).
- ⚠️ Για το M5 (πρώτη βελτιστοποίηση) ΔΕΝ το χρειαζόμαστε: scipy SLSQP
  καλύπτει μικρά εκπαιδευτικά NLP χωρίς καμία νέα εξάρτηση.

---

## 3. DWSIM / dwsim10

**Τι είναι:** Ολοκληρωμένος open-source process simulator. **Σημαντική
διόρθωση:** το dwsim10 ΔΕΝ είναι Python rewrite — είναι .NET 10.

### Στοιχεία
| Ερώτημα | Εύρημα |
|---|---|
| Τι είναι | **[VERIFIED, README]** C#/.NET 10, 63 projects, Avalonia desktop UI, version 10.2.0.0 |
| Δυνατότητες | **[VERIFIED, README]** Steady-state + dynamic, thermodynamics, unit operations, flowsheet solver, automation + fluent APIs, **MCP server** (27 tools, stdio ή HTTP/SSE), managed IPOPT solver |
| Έλεγχοι | **[VERIFIED, README]** 211 tests, 14 sample flowsheets |
| Προγραμματική κλήση | **[DOCUMENTED]** Μόνο μέσω .NET APIs / MCP — όχι Python library |
| Άδεια | **[VERIFIED, README]** **GPL-3.0** (COPYING) |

### Συμπέρασμα για το fgPROMS
- ❌ **Δεν ενσωματώνεται**: GPL-3.0 (copyleft) + .NET runtime. Ως ξεχωριστή
  διεργασία (π.χ. μέσω MCP) θα ήταν «mere aggregation», αλλά προσθέτει
  τεράστιο runtime για μικρό όφελος.
- 💡 Χρήσιμο ως **σημείο αναφοράς** για thermo/unit operations και για το
  concept ενός MCP server (ιδέα που μπορεί να μας χρησιμεύσει αργότερα).

---

## 4. MOSAICmodeling

**Τι είναι:** Δωρεάν **web** πλατφόρμα μοντελοποίησης (TU Berlin, DBTA):
LaTeX-style δηλωτική περιγραφή μοντέλων, συμβολική επεξεργασία και **code
generation** σε AMPL/GAMS/gPROMS/MATLAB/Python/Julia.

### Στοιχεία
| Ερώτημα | Εύρημα |
|---|---|
| Δυνατότητες | **[VERIFIED, ιστότοπος]** AEs/ODEs/PDEs, συνδυασμός σε συστήματα, transformations, connectors/ports/streams, βελτιστοποίηση, code generation |
| Μοντέλα χρήστη | **[VERIFIED]** Ναι — όλο το σύστημα είναι user-defined μοντέλα |
| Προγραμματική κλήση | **[VERIFIED, ιστότοπος]** Δεν είναι library — web πλατφόρμα με λογαριασμό (εγγραφή). Ο server είναι downloadable αλλά όχι Python API |
| Άδεια | **[DOCUMENTED]** Free to use με όρους χρήσης· δεν είναι ανοικτού κώδικα βιβλιοθήκη |

### Συμπέρασμα για το fgPROMS
- 💡 **Έμπνευση αρχιτεκτονικής**, όχι εξάρτηση: η φιλοσοφία
  «δηλωτικό μοντέλο → συμβολική ανάλυση → παραγωγή κώδικα» είναι ακριβώς
  η δική μας. Σημειώνει και τη σημασία των **connectors/ports** για
  flowsheets (χρήσιμο για το M7).
- 📖 Θα το αναφέρουμε στη βιβλιογραφία/τεκμηρίωση.

---

## 5. SciPy

| Ερώτημα | Εύρημα |
|---|---|
| Ρόλος | **[VERIFIED, ήδη σε χρήση]** Ο αριθμητικός πυρήνας του fgPROMS: `scipy.optimize.root` (hybr/lm), `scipy.integrate.solve_ivp` στα tests αναφοράς |
| Άδεια | BSD — καμία σύγκρουση |
| Έκδοση σε χρήση | 1.18.1 (Python 3.14, Windows) |

---

## 6. Πρόσθετα υποψήφια (εκτός του αρχικού prompt)

| Package | Ρόλος στο fgPROMS | Άδεια | Κατάσταση |
|---|---|---|---|
| **SymPy** | Συμβολική παραγώγιση → αναλυτικά Jacobians (M4) | BSD | Έτοιμο για χρήση |
| **CoolProp** | Θερμοφυσικές ιδιότητες (M6) | MIT (προς επαλήθευση κατά την εγκατάσταση) | Έλεγχος wheels για Py3.14 κατά το M6 |
| **CasADi** | Auto-diff + IDAS (BDF) + Ipopt σε ένα wheel | LGPL | Μόνο αν επιβεβαιωθούν wheels για 3.14 — δεν προγραμματίζεται άμεσα |

---

## Τελική απόφαση ενσωμάτωσης

| Repo | Απόφαση | Λόγος |
|---|---|---|
| SciPy | ✅ Παραμένει ο μοναδικός αριθμητικός πυρήνας | Ήδη δουλεύει, BSD, αρκετό για την κλίμακα |
| SymPy | ✅ Ενσωμάτωση (M4) | Αναλυτικά Jacobians, BSD, μηδενικό ρίσκο |
| CoolProp | ✅ Προαιρετικό backend (M6) | Πραγματικές ιδιότητες, 1 dependency |
| Pyomo | ⏸ Αναβολή — μόνο αν SLSQP δεν επαρκέσει | Δεν διπλασιάζουμε εργαλεία (αρχή του prompt) |
| IDAES | ⏸ Μόνο ως προμηθευτής Ipopt binaries στο μέλλον | Το framework θα μας δέσμευε |
| DWSIM | ❌ Αποκλεισμός | GPL-3.0 + .NET runtime |
| MOSAICmodeling | 💡 Έμπνευση (ports/streams για M7) + βιβλιογραφία | Δεν είναι library |
