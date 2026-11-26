# Nabor ocen D_lst

`D_lst` je seznam vrstic `[user_id, kombinacija aktivnosti, kontekst, ocena]`:

- **user_id**: oznaka anketiranca iz `data/scores/ml_data_scores_and_wgt.xlsx`,
- **kombinacija aktivnosti**: terka ene do treh aktivnosti (`actID`),
- **kontekst**: za vsako aktivnost pogoji časa in kraja (`C_T1–C_T3`, `C_P1–C_P3`),
- **ocena**: ustreznost kombinacije za uporabnika.

| Podatek | Vrednost |
|---|---|
| Datoteka | `data/D_lst/D_lst_full_20260202_150312.pkl` |
| Velikost | 581.511.163 B (približno 582 MB) |
| SHA-256 | `9b8ae198de039fc07893c1231f3a55de74b7f0c2f7dd94416edd9401dd477203` |
| Vrstice | 3.053.029 |
| Uporabniki | 679 |
| Kombinacije | 6.579 (1–3 izmed 34 aktivnosti: 34 + 561 + 5.984) |
| Ocene | od 0,3 do 5, povprečje 1,541 |

Datoteka ni v gitu (`*.pkl` je v `.gitignore`), ker vsebuje podatke posameznih uporabnikov.

Ponovno se ustvari bitno enako z `src/runGeneratorOfD_v7.py`, če je nastavljeno
`USE_EXISTING_D_LST = False` (zapiše `data/D_lst/D_lst_full_<časovni žig>.pkl`).
Enakost SHA-256 je bila preverjena 2026-11-26.
