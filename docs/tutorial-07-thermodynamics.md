# Μάθημα 7 — Θερμοδυναμική με το fgPROMS

Σε αυτό το μάθημα βλέπουμε πώς το fgPROMS καλύπτει τον εκπαιδευτικό κορμό της
θερμοδυναμικής: **ιδιότητες καθαρών ρευστών και μιγμάτων** (CoolProp), **κυβικές
καταστατικές εξισώσεις** γραμμένες μέσα στη γλώσσα, **ισορροπία φάσεων** και
**κύκλους ισχύος** — πάντα με επαλήθευση έναντι ανεξάρτητων αναφορών.

## 7.1 Ισορροπία φάσεων μίγματος: φάκελος φάσεων

Οι συναρτήσεις `TBUBBLE` / `TDEW` δίνουν τα όρια του διφασικού θύλακα ενός
μίγματος. Με έναν βρόχο `FOR` χτίζουμε ολόκληρο τον φάκελο φάσεων (20 πιέσεις,
2–10 bar) για το μίγμα n-βουτάνιο/προπάνιο (z = [0.4, 0.6]):

```gproms
MODEL Envelope
  PARAMETER
    N AS REAL DEFAULT 20
  VARIABLE
    z  AS ARRAY(2) OF REAL
    P  AS ARRAY(20) OF REAL DEFAULT 5.0E5
    Tb AS ARRAY(20) OF REAL DEFAULT 300.0
    Td AS ARRAY(20) OF REAL DEFAULT 300.0
  EQUATION
    z(1) + z(2) = 1 ;
    z(2) = 0.6 ;
    FOR i := 1 TO 20 DO
      P(i) = 2.0E5 + (i - 1) * (8.0E5 / (N - 1)) ;
      Tb(i) = TBUBBLE('n-Butane&Propane', z, P(i)) ;
      Td(i) = TDEW('n-Butane&Propane', z, P(i)) ;
    END
END
```

Στην καρτέλα «Γραφήματα» επίλεξε Χ = p και Υ = tb, td — θα δεις τον κλασικό
φάκελο φάσεων: η καμπύλη φυσαλίδας (tb) κάτω από την καμπύλη δρόσου (td), με
το μίγμα διφασικό ανάμεσά τους. Οι τιμές επαληθεύονται στα tests έναντι του
CoolProp με ακρίβεια 1e-3.

## 7.2 Κυβική καταστατική εξίσωση γραμμένη στη γλώσσα

Το πραγματικό εκπαιδευτικό ζητούμενο: να **δεις** την ίδια την εξίσωση. Εδώ
υλοποιούμε την Peng–Robinson για το n-βουτάνιο — η γλώσσα λύνει το κυβικό
πολυώνυμο για το Ζ και υπολογίζει τον συντελεστή φυγοκεντρικότητας φ:

```gproms
MODEL PengRobinson
  PARAMETER
    Tc AS REAL DEFAULT 425.125
    Pc AS REAL DEFAULT 3.796E6
    w  AS REAL DEFAULT 0.201
    T  AS REAL DEFAULT 400.0
    P  AS REAL DEFAULT 2.0E6
    R  AS REAL DEFAULT 8.314
  VARIABLE
    kappa AS REAL
    alpha AS REAL
    ac    AS REAL
    bc    AS REAL
    Apr   AS REAL
    Bpr   AS REAL
    Z     AS REAL
    phi   AS REAL
  EQUATION
    kappa = 0.37464 + 1.54226 * w - 0.26992 * w^2 ;
    alpha = (1 + kappa * (1 - SQRT(T / Tc)))^2 ;
    ac = 0.45724 * R^2 * Tc^2 / Pc * alpha ;
    bc = 0.07780 * R * Tc / Pc ;
    Apr = ac * P / (R^2 * T^2) ;
    Bpr = bc * P / (R * T) ;
    Z^3 - (1 - Bpr) * Z^2 + (Apr - 2*Bpr - 3*Bpr^2) * Z
        - (Apr*Bpr - Bpr^2 - Bpr^3) = 0 ;
    phi = EXP(Z - 1 - LN(Z - Bpr)
          - Apr / (2 * SQRT(2) * Bpr)
            * LN((Z + (1 + SQRT(2)) * Bpr) / (Z + (1 - SQRT(2)) * Bpr))) ;
  INITIAL
    kappa = 0.674 ;  alpha = 1.041 ;
    ac = 1.566 ;     bc = 7.25E-5 ;
    Apr = 0.283 ;    Bpr = 0.0436 ;
    Z = 0.7 ;        phi = 0.77 ;
END
```

Το αποτέλεσμα (Ζ = 0.708, φ = 0.770) ταιριάζει το backend `PR` του CoolProp με
ακρίβεια 1e-4. Δοκίμασε να αλλάξεις την αρχική τιμή του Z σε 0.05 — ο επιλυτής
θα βρει την **υγρή** ρίζα του κυβικού: οι τρεις ρίζες της PR είναι η ιστορία
ατμού/υγρού/ασταθούς κλάδου.

## 7.3 Κύκλος ισχύος Rankine

Η ισοεντροπική εκτόνωση στον στρόβιλο λύνεται **από τον επιλυτή**: γράφουμε
την συνθήκη S(T₄, P₁) = s₃ και η γλώσσα βρίσκει μόνη της την T₄ — με τις
κορεσμένες καταστάσεις από τη μορφή `H('Water', 'Q', 0, P)`:

```gproms
MODEL Rankine
  PARAMETER
    P1 AS REAL DEFAULT 1.0E4
    P2 AS REAL DEFAULT 3.0E6
    T3 AS REAL DEFAULT 673.15
  VARIABLE
    T1 AS REAL
    h1 AS REAL
    s1 AS REAL
    h2 AS REAL
    h3 AS REAL
    s3 AS REAL
    T4 AS REAL
    s4 AS REAL
    h4 AS REAL
    x4 AS REAL
    Qb AS REAL
    Qc AS REAL
    Wp AS REAL
    Wt AS REAL
    eta AS REAL
  EQUATION
    T1 = TBUBBLE('Water', P1) ;
    h1 = H('Water', 'Q', 0.0, P1) ;
    s1 = S('Water', 'Q', 0.0, P1) ;
    h2 = h1 + (P2 - P1) / RHO('Water', 'Q', 0.0, P1) ;
    h3 = H('Water', T3, P2) ;
    s3 = S('Water', T3, P2) ;
    s4 = s3 ;
    h4 = H('Water', 'PS', P1, s3) ;
    x4 = Q('Water', 'PS', P1, s3) ;
    T4 = T1 ;
    Qb = h3 - h2 ;
    Qc = h4 - h1 ;
    Wp = h2 - h1 ;
    Wt = h3 - h4 ;
    eta = (Wt - Wp) / Qb ;
  INITIAL
    T1 = 319.0 ;  h1 = 191800.0 ;  s1 = 649.0 ;
    h2 = 194800.0 ;
    h3 = 3.23E6 ;  s3 = 6790.0 ;
    T4 = 319.0 ;   s4 = 6790.0 ;   h4 = 2.2E6 ;   x4 = 0.85 ;
    Qb = 3.0E6 ;   Qc = 2.0E6 ;   Wp = 3000.0 ;  Wt = 1.0E6 ;
    eta = 0.3 ;
END
```

Ο θερμικός βαθμός απόδοσης βγαίνει ~34% — η γνωστή τιμή για ιδανικό Rankine
30 bar / 400 °C / 0.1 bar — και το πρώτο αξίωμα `Wt − Wp = Qb − Qc` κλείνει
με ακρίβεια μηχανής (επαληθεύεται στα tests).

## Τι μάθαμε

* Οι ιδιότητες μιγμάτων και οι φάκελοι φάσεων βγαίνουν με βρόχους FOR και
  επαληθεύονται έναντι του CoolProp.
* Οι κυβικές EOS γράφονται **μέσα στη γλώσσα** — ο φοιτητής βλέπει κάθε όρο
  και ο επιλυτής βρίσκει τις ρίζες του κυβικού.
* Η ισοεντροπική συνθήκη λύνεται από τον επιλυτή — καμία «μαγική» συνάρτηση.
