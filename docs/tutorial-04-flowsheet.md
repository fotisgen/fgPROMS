# Μάθημα 4 — Flowsheets (σύνθεση μονάδων)

Σε αυτό το μάθημα συνδέουμε μονάδες: ένας **Mixer** αναμιγνύει δύο
ρεύματα και ο **Splitter** μοιράζει το αποτέλεσμα 30/70.

## Η ιδέα

- Κάθε `MODEL` είναι μια **μονάδα** με μεταβλητές εισόδου/εξόδου
- Το `PROCESS` δημιουργεί **στιγμιότυπα** (`R1 AS Mixer`)
- Οι **συνδέσεις** είναι εξισώσεις με τελεία: `SP.Fin = MX.Fout`

## Το μοντέλο

```gproms
MODEL Mixer
  VARIABLE
    F1   AS REAL
    F2   AS REAL
    C1   AS REAL
    C2   AS REAL
    Fout AS REAL
    Cout AS REAL
  EQUATION
    Fout = F1 + F2 ;
    Fout * Cout = F1 * C1 + F2 * C2 ;
END

MODEL Splitter
  PARAMETER
    s AS REAL DEFAULT 0.3
  VARIABLE
    Fin AS REAL
    F1  AS REAL
    F2  AS REAL
  EQUATION
    F1 = s * Fin ;
    F2 = (1 - s) * Fin ;
END

PROCESS MixSplit
  UNIT
    MX AS Mixer
    SP AS Splitter
  EQUATION
    MX.F1 = 1.5 ;
    MX.F2 = 0.5 ;
    MX.C1 = 2.0 ;
    MX.C2 = 0.8 ;
    SP.Fin = MX.Fout ;
END
```

## Τι βλέπεις

- Καρτέλα **Μοντέλο**: το **διάγραμμα του δικτύου** (MX → SP) πάνω από
  τον πίνακα μονάδων
- Αναλυτική λύση: `Fout = 2`, `Cout = 1.7`, `SP.F1 = 0.6`, `SP.F2 = 1.4`

## Χρήσιμα

- **Παράμετροι μονάδας**: `R1 AS CSTRUnit SET V := 20` αλλάζει
  παραμέτρους ανά στιγμιότυπο
- Στο φάκελο `library/` υπάρχουν έτοιμες μονάδες (Heater, Pump, Mixer,
  Splitter, Flash) — αντέγραψε όποια θες στο δικό σου αρχείο
- Ολικό DOF = άγνωστοι όλων των μονάδων − εξισώσεις όλων των μονάδων
  και των συνδέσεων — πρέπει να είναι 0
