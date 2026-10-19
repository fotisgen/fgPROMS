# Μάθημα 5 — Κατανεμημένα συστήματα (PDE, 1D)

Σε αυτό το μάθημα λύνουμε μια **μερική διαφορική εξίσωση**: τη μόνιμη
αγωγή θερμότητας σε ράβδο, `d²T/dz² = 0`, με άκρα 300 και 350 K.

## Τα νέα στοιχεία

1. **`DISTRIBUTION`** — τομέας χώρου: `Axial : L : 11` (μήκος L, 11 σημεία)
2. **`DISTRIBUTION(Axial)`** — μεταβλητή με προφίλ κατά μήκος του άξονα
3. **`PARTIAL(T, Axial, Axial)`** — δεύτερη χωρική παράγωγος
4. **`BOUNDARY … AT`** — οριακές συνθήκες στα άκρα

## Το μοντέλο

```gproms
MODEL Conduction
  PARAMETER
    L  AS REAL DEFAULT 1.0
    T0 AS REAL DEFAULT 300.0
    TL AS REAL DEFAULT 350.0
  DISTRIBUTION
    Axial : L : 11 ;
  VARIABLE
    T AS DISTRIBUTION(Axial) OF REAL
  EQUATION
    PARTIAL(T, Axial, Axial) = 0 ;
  BOUNDARY
    T = T0 AT Axial = 0 ;
    T = TL AT Axial = L ;
END
```

## Τι βλέπεις

- Οι 11 τιμές `T(1)…T(11)` σχηματίζουν **γραμμικό προφίλ**:
  `T(z) = 300 + 50·z` — ακριβώς η αναλυτική λύση
- Η εξίσωση `PARTIAL(T, Axial, Axial) = 0` εφαρμόζεται στα 9 εσωτερικά
  σημεία με κεντρικές διαφορές· τα 2 άκρα ορίζονται από το BOUNDARY

## Δοκιμές

1. Αύξησε τα σημεία σε `101` → το προφίλ μένει το ίδιο (ακρίβεια)
2. Δοκίμασε **δυναμική διάχυση**: `$T = alpha * PARTIAL(T, Axial, Axial)`
   με `INITIAL T = 100` και `BOUNDARY T = 0` — βλέπε `diffusion_dynamic.gPROMS`
3. Πρόσθεσε **ολοκλήρωμα**: `Tavg = INT(T, Axial) / L` → 325 ακριβώς

## Πώς δουλεύει

Το fgPROMS χρησιμοποιεί τη **μέθοδο των γραμμών** (όπως το gPROMS):
διακριτοποιεί τον άξονα σε N σημεία και μετατρέπει την PDE σε σύστημα
αλγεβρικών (ή DAE) εξισώσεων που λύνει με τους κανονικούς επιλυτές.
