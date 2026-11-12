# Revizija D_lst za končni zagon v7 (20261015_150006)

Datum revizije: 2026-10-27. Diagnostika ne spreminja produkcijske kode, vhodov, obstoječega D_lst ali rezultatov.
Skripta: [audit_dlst.py](audit_dlst.py). Strojni izpisi: `results/*.json`, dnevniki korakov: `logs/`.

## 1. Sklep

**Odločitev A.** Obstoječi `data/D_lst/D_lst_full_20260202_150312.pkl` se v celoti ujema s postopkom, ki ga trenutna koda v7 izvede iz trenutnih vhodov. Regeneracija je dala **bajtno enak pickle** (enak SHA-256). Zamenjava D_lst ni potrebna, nov končni zagon zaradi D_lst prav tako ne.

To je tehnična skladnost. Vsebinska utemeljenost pravil (uteži, binarne kompatibilnosti, preslikava `Ac_AB4_8`, zdravstvena primernost) s tem ni potrjena. Glej razdelek 7.

## 2. Kaj je bilo izvedeno in obseg

| Korak | Obseg |
|---|---|
| SHA-256 obstoječega D_lst, kode in vhodov; LFS kazalec v gitu | vse navedene datoteke |
| Strukturni pregled obstoječega D_lst | **vseh 3.053.029 zapisov** |
| Regeneracija s **trenutno** v7 (STEP 2–10 izvedeni iz izvorne kode v7, nato dobesedni klic `erst.get_dataMat(...)` iz veje `else`) | 696 uporabnikov × 6.579 kombinacij = 4.579.584 parov, brez `test_mode`, brez rezanja; 1.738 s |
| Regeneracija z **zgodovinsko** kodo (commit `45d9509`, 2025-07-25: `runGeneratorOfD_v5.py` + tedanji `elderly_recsys_tools.py`, vhoda `ActivityContextGen_v09`/`v10`) | enak obseg; 1.742 s |
| **Neodvisna** vektorska implementacija formule (numpy, brez `elderly_recsys_tools`), z v12 in z v09/v10 | enak obseg |
| Primerjava vseh parov, ocen (bitno), vrstnega reda in zapisa kombinacij | 7 parov primerjav, vsi zapisi |
| Sledljivi primeri | 8 primerov |

Normalizacija pri regeneraciji je potekala v isti funkciji (`get_dataMat`) na celotni množici vključenih zapisov, ne po kosih. Modeli se niso učili. Matplotlib je imel onemogočena `savefig`/`show`; v `outputs/`, `data/` in `src/` ni nove ali spremenjene datoteke (preverjeno po zagonu).

## 3. Primerjava starega in regeneriranega nabora

| Merilo | Obstoječi | Regeneriran (trenutna v7) | Regeneriran (45d9509) | Neodvisni izračun (v12) |
|---|---:|---:|---:|---:|
| SHA-256 pickla | `9b8ae198…477203` | **`9b8ae198…477203`** | **`9b8ae198…477203`** | (ni pickle) |
| Zapisi | 3.053.029 | 3.053.029 | 3.053.029 | 3.053.029 |
| Uporabniki | 679 | 679 | 679 | 679 |
| Različne kombinacije | 6.579 | 6.579 | 6.579 | 6.579 |
| Dolžine 1 / 2 / 3 | 8.312 / 216.804 / 2.827.913 | enako | enako | enako |
| Skupni pari | — | 3.053.029 | 3.053.029 | 3.053.029 |
| Manjkajoči / dodatni pari | — | 0 / 0 | 0 / 0 | 0 / 0 |
| Bitno enake ocene | — | 3.053.029 | 3.053.029 | 3.053.029 |
| Največja absolutna razlika | — | 0,0 | 0,0 | 0,0 |
| Enak vrstni red (po položaju) | — | da, vse vrstice | da | da |
| Kontekstni slovarji po actID | — | enaki | enaki | (ni primerjano) |

Toleranca ni bila potrebna: vse ocene so bitno enake.

Ker je vrstni red identičen, `ShuffleSplit(random_state=42)` na tem naboru da iste indekse razvojnega in testnega dela kot v zagonu 20261015_150006.

## 4. Pregled obstoječega D_lst (vsi zapisi)

- Vsebnik je `list`; vsak zapis je `list` dolžine 4: `[uid (numpy.int64), item (tuple str), context (tuple dict), rating (numpy.float64)]`.
- 0 NaN in 0 neskončnih ocen. Razpon je 0,3–5,0, povprečje 1,541340. Vrednost 0,3 ima 2.314 zapisov, vrednost 5,0 pa 603.
- 34 različnih actID, vsi iz vprašanj `Ac_AB4_{1,2,3,4,5,8}` in vsi prisotni v `ActionLst` v12 in v09.
- Ni ponovljenega actID znotraj kombinacije, ni podvojenih parov uporabnik–kombinacija in ni podvojitev zaradi permutacij (6.579 urejenih = 6.579 neurejenih elementov).
- Vsak kontekstni slovar ima ključe `qID, C_T1..C_T3, C_P1..C_P3`. Njegov `qID` je predpona actID, za isti actID pa je slovar v vseh vrsticah enak.
- uid so nepadajoči. Znotraj uporabnika so kombinacije v vrstnem redu `itertools.combinations` (dolžina 1, nato 2, nato 3; actID v vrstnem redu vrstic Excela). Isto zaporedje sta neodvisno dala regeneracija in neodvisni izračun.

### 696 → 679 uporabnikov

`uIDs` so vsi `person_id` iz `ml_data_scores_and_wgt.xlsx` (696, brez podvojitev, vsi prisotni v vprašalnikih). Izločenih je 17 uporabnikov: **449, 471, 489, 566, 622, 926, 1012, 1065, 1145, 1170, 1393, 1537, 1684, 1689, 1690, 1715, 1725**.

Vsi imajo enak dejanski razlog: **vseh šest odgovorov `Ac_AB4_*` je 0** (nihče nima manjkajočih odgovorov). Zato je vsak prispevek 0, največji raw je 0 ≤ r_T = 0,3 in noben njihov par ne preide praga. Utež `a_F2` pri tem ne igra vloge. Nihče ni izločen zaradi NaN, podvojenega ID-ja ali napake (`results/independent.json`, `missing_users`).

## 5. Postopek nastanka (iz izvedljive kode, ne iz komentarjev)

**Vhodi, ki dejansko vplivajo na ocene:**
- `Activities.xlsx` / `AllAnswers`: stolpci `AB4_1, AB4_2, AB4_3, AB4_4, AB4_5, AB4_8` (z indeksom `S4`, `-1 → NaN`). Odgovori so binarni (0/1).
- `ml_data_scores_and_wgt.xlsx` / `scores_and_wgt`: indeks `person_id` (izbor uporabnikov) in stolpec `a_F2` (utež; 50 različnih vrednosti, razpon 1–5).
- `ActivityContextGen_v12.xlsx` / `ActionLst`: `qID` (z dedovanjem navzdol), `actID` in vrstni red vrstic (34 aktivnosti).
- `ActivityContextGen_v12.xlsx` / `ActionCompatibilities`: blok 34 × 34.

**Vhodi, ki vplivajo samo na zapis konteksta (ne na ocene):** `act_C_T1..3` in `act_C_P1..3` iz `ActionLst`.

**Prebrano brez vpliva na D_lst:** `MentalHealth`, `PhysicalHealth`, `SocialHealth` (samo `join`; za aspekt `activity` se uporabljajo le stolpci `Ac_`), ostali stolpci uteži, `action_prop_*`, `kontekst`/`context_lst` in skupine `ContextG_*`. Podvojeni `S4 = 1298` v vseh štirih vprašalnikih po `join` da 16 vrstic, vendar 1298 ni med 696 uporabniki. Na min-max skaliranje binarnih stolpcev ne vpliva. Filter `['A82_r1','A82_r3','A83_r']` nikoli ne zadene, ker imajo qID predpono `Mh_`/`Sh_`; na aspekt `activity` ne vpliva.

**Izračun** (`get_actID_score_df`, `get_uID_answers_df`, `get_score_estimation`, `get_dataMat`):

1. `S[u,a] = a_F2[u] · x[u,q(a)]`. Tu je `x` surovi odgovor; NaN ostane NaN, neštevilska vrednost da 0.
2. `s[u,q]` je min-max skaliran odgovor, kjer je bil NaN pred skaliranjem nastavljen na 0, skalirnik pa naučen na vseh vrsticah vprašalnika.
3. Prispevek aktivnosti je `v[u,a] = S[u,a] · s[u,q(a)] = a_F2 · x · s`. Odgovor torej nastopa dvakrat. Ker je `x ∈ {0,1}` in `s = x` (preverjeno), velja `v = a_F2 · x`.
4. `raw = Σ v − (1 − Π C[a_i, a_j])` po parih `i<j` v vrstnem redu kombinacije. C je binarna (0/1) in simetrična, brez NaN. Kazen je zato 0 ali **točno 1**, ne glede na število nezdružljivih parov.
5. Vključitev velja za `raw > r_T = 0,3` (NaN je izključen samodejno). Ta prag se uporabi pred normalizacijo.
6. Nato sledi min-max normalizacija vseh vključenih zapisov na `[r_T, 5]`: `0,3 + 4,7·(raw − min)/(max − min)`, kjer je min = 0,388527 in max = 15,0.
7. Kombinacije so `itertools.combinations` (brez permutacij, brez ponavljanja) dolžin 1–3 nad 34 aktivnostmi: 34 + 561 + 5.984 = 6.579.

**Prag:** največji izključeni raw je 0,0, najmanjši vključeni pa 0,388527. Vsak prag v intervalu [0; 0,3885) bi dal isti nabor; v okolici 0,3 ni mejnih primerov.

**Vhod v12 proti v09/v10:** za 34 aktivnosti dasta enake actID, vrstni red, qID, kontekste in kompatibilnosti. Neodvisni izračun z obema različicama in obe regeneraciji dajo isti rezultat. Sprememba `zvečerr → zvečer` v v12 na te zapise nima vpliva.

## 6. Sledljivi primeri (`results/examples.json`)

Normalizacija: min = 0,38852660616873647, max = 15,0. Vsi primeri se z obstoječim D_lst ujemajo bitno.

| Primer | Uporabnik / kombinacija | Prispevki `a_F2·x·s` | C | raw | ocena |
|---|---|---|---|---:|---:|
| 1 aktivnost | 1373 / `AB4_1_Act01` | 1,38853 | — | 1,38853 | 0,62167 |
| 2 aktivnosti, nezdružljivo | 1373 / `AB4_1_Act01, AB4_1_Act02` | 1,38853 + 1,38853 | 0 | 2,77705 − 1 = 1,77705 | 0,74664 |
| 2 aktivnosti, združljivo | 1373 / `AB4_1_Act01, AB4_2_Act01` | 1,38853 + 1,38853 | 1 | 2,77705 | 1,06831 |
| 3 aktivnosti | 1373 / `AB4_1_Act01..03` | 3 × 1,38853 | Π = 0 | 4,16558 − 1 = 3,16558 | 1,19328 |
| prag, najmanjši vključen | 1373 / `AB4_1_Act01, AB4_8_Act01` | 1,38853 + 0 (x=0) | 0 | 0,38853 | 0,3 |
| prag, največji izključen | 2 / `AB4_4_Act01` | 0 (x=0) | — | 0,0 | izključeno |
| največja ocena | 456 / `AB4_1_Act01, AB4_2_Act01, AB4_5_Act01` | 3 × 5,0 | 1 | 15,0 | 5,0 |

Za uporabnika 1373 velja `a_F2 = 1,3885266061687365`, `AB4_1 = 1`, `AB4_2 = 1`, `AB4_8 = 0`. Ocena 0,62167 je `0,3 + 4,7·(1,38853 − 0,38853)/(15 − 0,38853)`.

**Skladnost z opisom:** besedilo magistrske naloge v repozitoriju ni na voljo, zato skladnosti z njim **nisem preveril**. Primerjal sem kodo z opisi, ki so v repozitoriju:
- Docstring v7 (STEP 11) navaja `relevance = score * answer` in kompatibilnost za pare. Izvedba računa `score · answer · scaled(answer)`, kar je pri binarnih odgovorih enako `score · answer`. Kompatibilnost vstopa kot `−(1 − ΠC)`.
- README navaja, da so ocene izračunane iz vprašalniških odgovorov, uteži in združljivosti. To je skladno.

Natančne enačbe za kombinacije (seštevanje, kazen 1 − ΠC, normalizacija na [0,3; 5]) je treba primerjati z besedilom magistrske naloge.

## 7. Vsebinska opažanja (niso tehnične napake regeneracije)

Spodnje lastnosti so dosledno prisotne v kodi in v D_lst. Ali so metodološko sprejemljive, je odločitev avtorja in mentorja.

1. **Preslikava `Ac_AB4_8`.** Vprašanje se glasi »… zavestno ne skrbim za zdravje«. Odgovor 1 poveča relevantnost 11 aktivnosti `AB4_8_Act01..11` z enakim predznakom kot pri drugih vprašanjih. Pri drugih vprašanjih odgovor 1 pomeni, da oseba to že počne. Odgovor 1 na AB4_8 je dalo 38 od 679 vključenih uporabnikov.
2. **Podvojene vsebine.** `AB4_8_Act01..09` imajo enako besedilo kot `AB4_1_Act01..09`. 297 kombinacij oziroma **158.292 zapisov** vsebuje isto aktivnost dvakrat pod različnima ID-jema, na primer »V obrok vključite zelenjavo.« dvakrat.
3. **Člani z ničelnim prispevkom.** **2.709.618 od 3.053.029 zapisov (88,8 %)** vsebuje vsaj eno aktivnost, na katere vprašanje je uporabnik odgovoril 0. V nabor pridejo zaradi drugih članov kombinacije.
4. **Kazen kompatibilnosti.** 258 od 561 parov ima C = 0 (95 znotraj istega vprašanja, 163 med vprašanji). **2.765.081 zapisov (90,6 %)** ima kazen 1 in ostane v naboru. Kombinacija z več nezdružljivimi pari (1.271 kombinacij, 557.381 zapisov) dobi enako kazen kot kombinacija z enim takim parom.
5. **Lestvica in izenačitve.** Ocena ima le 243 različnih vrednosti raw (50 vrednosti uteži × število pritrdilnih članov ± kazen), zato je veliko izenačitev. To vpliva na izbiro referenčnih top-5, kar README že omenja. Najnižja ocena je po konstrukciji natanko r_T.
6. **Zdravstvena primernost** priporočil in ustreznost uteži `a_F2` iz te revizije ne izhajata.

## 8. Izvor (kaj je dokazano)

- **Dokazana identičnost datoteke:** SHA-256 obstoječe datoteke `9b8ae198de039fc07893c1231f3a55de74b7f0c2f7dd94416edd9401dd477203` je enak zapisu v `RUN_20261015_150006.md`. `D_lst_full.pkl` je bajtno enaka kopija. Dnevnik zagona navaja to pot in `rows=3053029`, vendar hasha ob zagonu ni zajel.
- **Skladnost s trenutno kodo:** dokazana z bajtno enako regeneracijo iz trenutne v7 (SHA `76624b73…`), `elderly_recsys_tools.py` (SHA `a6b6cf23…`) in trenutnih vhodov.
- **Zgodovinski nastanek:** LFS kazalec `data/D_lst_full.pkl` v commitu `45d9509` z dne **2025-07-25** ima že isti oid `sha256:9b8ae198…`. Vsebina torej obstaja vsaj od takrat; datum `20260202_150312` v imenu ni datum nastanka vsebine. Koda iz tega commita z v09/v10 da bajtno enak rezultat. Vprašalniki in uteži so enaki tedanjim (enaki git blobi). Kateri od teh postopkov je datoteko dejansko ustvaril, ni dokazano, ker zagona iz julija 2025 ne beleži noben dnevnik. Ker trenutna in zgodovinska koda dasta isti rezultat, to na pravilnost ne vpliva.

## 9. Kaj ostaja neznano

- Ali je bila v zagonu 20261015_150006 res naložena prav ta vsebina: hash ob zagonu ni bil zajet. Ujemanje poti, števila vrstic, števila uporabnikov in elementov, min/max/povprečja ter izpisanih najnižjih in najvišjih zapisov je močan posreden dokaz, ne pa dokaz.
- Ukaz, okolje in različice paketov ob julijskem nastanku in ob zagonu.
- Besedilo magistrske naloge in njena enačba (ni v repozitoriju).
- Vsebinska ustreznost pravil iz razdelka 7.

## 10. Potrebujemo nov D_lst ali nov zagon?

Ne. D_lst je bitno in po vrstnem redu enak regeneriranemu, zato bi nov zagon z regeneriranim naborom dal enake vhode za STEP 12–18. Če bi se zaradi razdelka 7 spremenila pravila (na primer AB4_8, podvojene vsebine, kazen ali izločanje kombinacij s člani brez prispevka), bi bilo treba ponovno ustvariti D_lst in ponoviti STEP 12–18: primerjavo algoritmov, GridSearch, CV, končni SVD, held-out metrike, priporočila 4.1/4.2/M3–M5 in tabelo 4.5. Spremenila bi se namreč množica parov, ocene in s tem tudi delitev.

## 11. Datoteke in ponovitev

V `diagnostics/dlst_audit_20261027/`:
- `audit_dlst.py`: diagnostična skripta
- `REPORT.md`: to poročilo
- `results/`: `provenance.json`, `checks_existing.json`, `checks_current.json`, `checks_hist45d9509.json`, `regen_current.json`, `regen_hist45d9509.json`, `independent.json`, `compare.json`, `examples.json`
- `logs/`: izpisi posameznih korakov, vključno s `content_stats.txt` (števila iz razdelka 7)
- `work/` (velike datoteke, `*.pkl` so v `.gitignore`): `D_lst_regen_current.pkl`, `D_lst_regen_hist45d9509.pkl`, `digest_*.pkl`, `intermediates_*.pkl` in `hist45d9509/` (koda iz commita in kopije tedanjih vhodov)

Ponovitev (približno 35 min za vsako regeneracijo; skupaj okoli 70 min, ker `all` korake izvaja zaporedno; potrebnih je okoli 8 GB RAM):

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -B -W ignore diagnostics\dlst_audit_20261027\audit_dlst.py all
```

Opomba o okolju: med revizijo je Windows (Application Control) občasno zavrnil nalaganje DLL-jev SciPy. Ponovni zagon je težavo odpravil, na rezultate pa nima vpliva.
