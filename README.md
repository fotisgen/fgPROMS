# fgPROMS

**fgPROMS** — ένα τοπικό στούντιο μοντελοποίησης διεργασιών, εμπνευσμένο από τη
γλώσσα μοντελοποίησης του gPROMS. Γράφεις μοντέλα σε γλώσσα τύπου gPROMS,
τα αναλύεις, τα επιλύεις (steady-state ή δυναμικά) και βλέπεις αποτελέσματα
και γραφήματα — όλα τοπικά στον υπολογιστή σου, χωρίς internet.

> ⚠️ **Σημείωση:** Το fgPROMS είναι ανεξάρτητο, προσωπικό/εκπαιδευτικό εργαλείο.
> Δεν σχετίζεται με τη Siemens/PSE και δεν αποτελεί προϊόν της. Το "gPROMS"
> είναι εμπορικό σήμα της Siemens Process Systems Engineering.

---

## 🚀 Εκκίνηση

1. Κάνε **διπλό κλικ στο `run.bat`**
2. Την πρώτη φορά δημιουργείται εικονικό περιβάλλον Python και εγκαθίστανται
   τα απαραίτητα πακέτα (numpy, scipy, fastapi, uvicorn) — θέλει 1-2 λεπτά.
3. Ανοίγει αυτόματα ο browser στη διεύθυνση `http://127.0.0.1:8765`
4. Για να σταματήσεις: `Ctrl+C` στο παράθυρο του server.

Απαιτείται Python 3.10+ (αν υπάρχει, το `run.bat` τη βρίσκει μόνο του).

## 🖥️ Χρήση

- **Επεξεργαστής** (αριστερά): γράφεις ή φορτώνεις ένα αρχείο `.gPROMS`
- **Παραδείγματα**: έτοιμα μοντέλα από το μενού (CSTR, εναλλάκτης, δεξαμενή,
  batch reactor)
- **Εκτέλεση**: το κουμπί «Εκτέλεση» αναλύει και επιλύει το μοντέλο
- **Καρτέλες** (δεξιά):
  - *Μοντέλο* — παράμετροι, μεταβλητές, εξισώσεις, βαθμοί ελευθερίας
  - *Αποτελέσματα* — τιμές μεταβλητών και υπόλοιπα εξισώσεων
  - *Γραφήματα* — διαγράμματα δυναμικών προσομοιώσεων (διαδραστικά)
  - *Μηνύματα* — διαγνωστικά, προειδοποιήσεις, σφάλματα
- **Ελληνικά/Αγγλικά**: εναλλαγή από το κουμπί επάνω δεξιά
- **Άνοιγμα/Αποθήκευση** αρχείων τοπικά

## 📜 Υποστηριζόμενη γλώσσα (Φάση 2)

```gproms
MODEL CSTR
  PARAMETER
    Fin  AS REAL  DEFAULT 1.0     # m3/h
    V    AS REAL  DEFAULT 10.0
    k    AS REAL  DEFAULT 0.5
  VARIABLE
    Fout AS REAL
    CA   AS REAL
  EQUATION
    Fout = Fin ;
    Fin * CAin - Fout * CA - V * k * CA = 0 ;
END
```

| Στοιχείο | Υποστήριξη |
|---|---|
| `MODEL … END`, `PARAMETER`, `VARIABLE`, `EQUATION` | ✅ |
| `DEFAULT` τιμές, `SET` (καθορισμός μεταβλητών) | ✅ |
| `IF … THEN … ELSE … END` (συνθήκες σε εξισώσεις) | ✅ |
| Δυναμικά: παράγωγοι `$x` = dx/dt, `INITIAL`, `SCHEDULE` | ✅ |
| **Δυναμικός επιλυτής BDF μεταβλητής τάξης (1–5)** με προσαρμοστικό βήμα | ✅ |
| **Βελτιστοποίηση**: `OPTIMIZE` (MAXIMIZE/MINIMIZE), `BOUNDS` — SLSQP | ✅ |
| **Flowsheets**: `PROCESS`, `UNIT name AS Type [SET …]`, συνδέσεις `R2.x = R1.y` | ✅ |
| **Dynamic flowsheets**: μονάδες με `$` + `SCHEDULE`/`INITIAL` στο PROCESS | ✅ |
| **Πίνακες**: `ARRAY(3)`, `ARRAY(3,4)`, `z(i,j)`, `SIGMA` (και nested) | ✅ |
| **Κατανεμημένα (PDE, 1D)**: `DISTRIBUTION`, `PARTIAL(T,Axial[,Axial])`, `BOUNDARY … AT`, μέθοδος γραμμών | ✅ |
| **Εκτίμηση παραμέτρων**: `DATA` + `ESTIMATE`, ελάχιστα τετράγωνα (gEST-style) | ✅ |
| **Φάσεις batch**: `TASK`/`PHASE`, `WITH phase` στο SCHEDULE, επανεκκίνηση στην αλλαγή | ✅ |
| **Ιδιότητες**: μίγματα CoolProp `'n-Butane[0.4]&Propane[0.6]'`, `Q`, bubble/dew, `U`, `G` | ✅ |
| **Βιβλιοθήκη μονάδων**: `library/` (Heater, Pump, Mixer, Splitter, Flash) | ✅ |
| **Projects**: αποθήκευση/φόρτωση αρχείων στον φάκελο `projects/` | ✅ |
| **Εξαγωγή CSV** αποτελεσμάτων | ✅ |
| Αναλυτικά Jacobians (SymPy) + damped Newton | ✅ |
| Steady-state (Newton/LM) + δυναμική (DAE, BDF adaptive) | ✅ |
| `SELECTOR`, `STREAM` με ιδιότητες, N-D διακριτοποίηση, βιομηχανικά πακέτα | ❌ (μελλοντικά) |
| Ανισότητες (`>`, `<`) σε εξισώσεις | ⚠️ αγνοούνται με προειδοποίηση |

## 📚 Εκπαιδευτικό υλικό

- **Tutorials** (`docs/`): 4 μαθήματα στα ελληνικά — πρώτο μοντέλο, δυναμική
  προσομοίωση, βελτιστοποίηση, flowsheets. Τα παραδείγματα κώδικα μέσα στα
  tutorials **εκτελούνται αυτόματα από τη σουίτα ελέγχων** (δεν μπορούν να
  ξεπεραστούν).
- **Βιβλιοθήκη μονάδων** (`library/`): Heater, Pump, Mixer, Splitter, Flash —
  αντέγραψε όποια μονάδα θες στο δικό σου μοντέλο.

## 📁 Δομή

```
fgPROMS/
├── run.bat               ← διπλό κλικ για εκκίνηση
├── launch.py             ← server + αυτόματο άνοιγμα browser
├── backend/
│   ├── app.py            ← FastAPI server
│   ├── engine.py         ← parse → analyze → solve
│   ├── parser/           ← parser γλώσσας gPROMS
│   ├── analyzer/         ← δομική ανάλυση, βαθμοί ελευθερίας
│   └── solver/           ← αριθμητικοί επιλυτές (NumPy/SciPy)
├── frontend/             ← γραφικό περιβάλλον (editor, καρτέλες, Plotly)
├── examples/             ← παραδείγματα .gPROMS
└── test_backend.py       ← αυτοματοποιημένος έλεγχος παραδειγμάτων
```

## 🧪 Έλεγχος λειτουργικότητας

Δύο επίπεδα ελέγχου:

### Αυτοματοποιημένη σουίτα (160 έλεγχοι + 8 browser tests)

Διπλό κλικ στο **`test.bat`** (ή `.venv\Scripts\python -m pytest tests -v`):

| Ομάδα | Περιεχόμενο |
|---|---|
| `tests/test_parser.py` | 20 έλεγχοι σύνταξης, προτεραιοτήτων τελεστών, σφαλμάτων |
| `tests/test_analyzer.py` | 12 έλεγχοι DOF, διαγνωστικών, συνθηκών αποκλεισμού |
| `tests/test_solver_steady.py` | αναλυτικές λύσεις: CSTR, εναλλάκτης, Rachford-Rice vs bisection, Jacobians |
| `tests/test_solver_dynamic.py` | εκθετική απόσβεση vs e^(−at), **τάξη σύγκλισης 2**, Van der Pol vs scipy, ισοζύγιο μάζας |
| `tests/test_optimization.py` | SLSQP vs αναλυτικά βέλτιστα, όρια, εκφυλισμένες περιπτώσεις |
| `tests/test_properties.py` | CoolProp: cp νερού vs βιβλιογραφία, ιδιότητες σε μοντέλα/βελτιστοποίηση |
| `tests/test_flowsheet.py` | 2 CSTR σε σειρά vs αναλυτική λύση, **δύο δεξαμενές (dynamic)**, SET overrides |
| `tests/test_sigma.py` | SIGMA/ARRAY: N-component flash vs bisection, αθροίσματα |
| `tests/test_library.py` | μονάδες βιβλιοθήκης vs αναλυτικές λύσεις |
| `tests/test_tutorials.py` | τα παραδείγματα των tutorials εκτελούνται πραγματικά |
| `tests/test_projects.py` | project storage + CSV export |
| `tests/test_gui.py` | **πραγματικός browser (Playwright)**: editor, αποτελέσματα, γραφήματα, διάγραμμα, γλώσσα, projects, CSV |
| `tests/test_robustness.py` | 15 έλεγχοι: χαλασμένα αρχεία → καθαρά μηνύματα, ποτέ crash |
| `tests/test_api.py` | endpoints με FastAPI TestClient |

Τα benchmarks (`benchmarks/`) είναι μοντέλα με **κλειστές αναλυτικές λύσεις**
ή ανεξάρτητες λύσεις αναφοράς (scipy/bisection), ώστε κάθε τρέξιμο να
επαληθεύει πραγματικά την αριθμητική ορθότητα.

### Χειροκίνητη λίστα GUI

Δες το **`CHECKLIST_GUI.md`** — ~30 βήματα ελέγχου του γραφικού περιβάλλοντος
(εκκίνηση, editor, παραδείγματα, καρτέλες, γραφήματα, γλώσσα, αρχεία).

## 🎯 Αναμενόμενες τιμές παραδειγμάτων

- **CSTR**: `Fout = 1`, `CA = 0.3333`, `r = 0.1667`
- **BatchReactor** (n=1.5): αναλυτική λύση `CA(30) = (1 + 0.5·0.1·30)^(−2) = 0.16`
