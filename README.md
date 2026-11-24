# Kontekstualiziran priporočilni sistem za dnevne aktivnosti starejših

Magistrsko delo, Fakulteta za elektrotehniko Univerze v Ljubljani  
**Avtor:** Gašper Leskovec · **Mentor:** prof. dr. Andrej Košir

Projekt obravnava priporočanje posameznih aktivnosti in kombinacij do treh aktivnosti starejšim. Model SVD napoveduje ocene, izračunane iz vprašalniških odgovorov, uporabniških uteži in združljivosti aktivnosti. Naknadni filtri omejijo izbor glede na čas, kraj ali pripadnost aktivnosti kontekstni skupini.

## Postopek

Končna izvedba je [runGeneratorOfD_v7.py](src/runGeneratorOfD_v7.py). Uporablja modul [elderly_recsys_tools.py](src/elderly_recsys_tools.py) za pripravo podatkov in vrednotenje ter [recommendation_items.py](src/recommendation_items.py) za obravnavo celotnih kombinacij in njihovo filtriranje.

1. Skripta prebere vhodne podatke in naloži pripravljeni nabor ocen `D_lst`.
2. Pred izbiro modela odloži 25 % ocen za končni test. Na razvojnem delu primerja SVD, KNNBasic, BaselineOnly in NMF ter poišče nastavitve SVD. Pri `SKIP_TUNING = True` (privzeto) ta korak preskoči in uporabi `BEST_PARAMS` iz zagona `20261015_150006`.
3. Končni SVD nauči na razvojnem delu in napove ocene testnih parov uporabnik–kombinacija.
4. Za vsakega uporabnika razvrsti vse odložene testne kombinacije. Vrne največ pet kombinacij brez filtriranja, s filtrom časa in kraja ali ločeno po skupinah M3–M5.

Pri kontekstnem filtru morajo vse aktivnosti kombinacije ustrezati istemu simuliranemu času in kraju. Pri skupinskem filtru morajo vse pripadati isti skupini. Seznama po filtriranju ne dopolnjuje z drugimi kandidati, zato je lahko krajši ali prazen.

## Namestitev

Uporabljen je Python 3.11. Ukaze zaženi v PowerShellu iz korena projekta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Različice v `requirements.txt` izhajajo iz popisa lokalnega okolja 21. 10. 2026. V svežem okolju se `requirements.txt` namesti brez napak (preverjeno 2026-11-24).

## Potrebni podatki

Za zagon morajo biti na voljo naslednje datoteke. Poti so relativne na koren projekta.

| Pot | Vsebina oziroma list |
|---|---|
| `data/questionnaires/Activities.xlsx` | odgovori o aktivnostih, `AllAnswers` |
| `data/questionnaires/MentalHealth.xlsx` | `AllAnswers` |
| `data/questionnaires/PhysicalHealth.xlsx` | `AllAnswers` |
| `data/questionnaires/SocialHealth.xlsx` | `AllAnswers` |
| `data/context/ActivityContextGen_v12.xlsx` | `ActionLst`, `ActionCompatibilities`, kontekstne oznake in skupine |
| `data/scores/ml_data_scores_and_wgt.xlsx` | uporabniške ocene in uteži, `scores_and_wgt` |
| `data/D_lst/D_lst_full_20260202_150312.pkl` | pripravljeni nabor ocen, približno 581 MB |

Skripta bere vse štiri vprašalnike; pri nastavitvi `aspect_groups_lst = ['activity']` za izračun ocen uporablja vprašanja o aktivnostih. Nabor `D_lst` se zagotovi ločeno. Brez potrebnih vhodov zagon ni mogoč. Vprašalniki, uteži in individualni izvozi niso namenjeni samodejni javni objavi.

## Zagon

```powershell
$env:MPLBACKEND = "Agg"
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe src\runGeneratorOfD_v7.py
```

Nastavitve so v začetnem delu skripte:

- `USE_EXISTING_D_LST = True` naloži pripravljeni nabor. Vrednost `False` sproži generiranje novega nabora iz vhodov.
- `test_mode = False` izvede polni eksperiment. Vrednost `True` omeji nabor na prvih 100.000 zapisov in mrežo nastavitev na eno kombinacijo; deset ponovitev preverjanja ostane. To je preizkus delovanja, ne reprezentativen vzorec celotnega nabora.
- `SKIP_TUNING = True` preskoči STEP 12–14 (primerjava algoritmov, iskanje nastavitev, preverjanje SVD) in uporabi `BEST_PARAMS`. Vrednost `False` jih ponovi.

Zagon s `SKIP_TUNING = False` vključuje iskanje nastavitev in lahko traja več ur; zagon 15. 10. 2026 je trajal približno 5 h 43 min. Zagon 24. 11. 2026 s `SKIP_TUNING = True` je trajal približno 5 min. Za pregled obstoječih rezultatov ga ni treba ponavljati.

Obstoječi test pomožnega modula:

```powershell
.\.venv\Scripts\python.exe -B tests\test_recommendation_items.py -v
```

## Rezultati in njihova razlaga

Končni zagon ima oznako **`20261124_183727`** in obsega **679 uporabnikov, 6.579 kombinacij ter 3.053.029 ocen**. Razvojni del vsebuje 2.289.771 ocen, testni pa 763.258. Primerjava algoritmov in nastavitve SVD so iz zagona `20261015_150006`.

SVD na odloženem testnem delu doseže **RMSE 0,0819**, **MAE 0,0531** in **FCP 0,9966**. Brez kontekstnega filtra sistem vrne 3.395 priporočil, s filtrom pa 2.534 za 608 od 679 uporabnikov.

| Različica | Natančnost | Priklic | F1 | F1 z izenačenji | Pokritost | Povp. ocena |
|---|---:|---:|---:|---:|---:|---:|
| Brez konteksta (baseline) | 0,339 | 0,339 | 0,339 | 0,982 | 1,000 | 3,011 |
| Popularnost (baseline) | 0,151 | 0,151 | 0,151 | 0,531 | 1,000 | 3,054 |
| S kontekstom, globalna referenca | 0,018 | 0,018 | 0,018 | 0,120 | 0,895 | 2,022 |
| S kontekstom, kontekstna referenca | 0,804 | 0,804 | 0,804 | 0,998 | 0,895 | 2,022 |
| M3 (uteženo po uporabnikih) | 0,623 | 0,623 | 0,623 | 0,988 | -- | 1,928 |
| M4 (uteženo po uporabnikih) | 0,698 | 0,698 | 0,698 | 0,996 | -- | 1,844 |
| M5 (uteženo po uporabnikih) | 0,683 | 0,683 | 0,683 | 0,998 | -- | 1,892 |

Ocene so izračunane po pravilih, zato napake napovedovanja merijo ujemanje s temi ocenami. Rezultati ne potrjujejo zadovoljstva uporabnikov ali zdravstvenih učinkov. Na metrike top-5 vpliva tudi leksikografsko razreševanje izenačenih referenčnih ocen. Kontekstna referenca in M3–M5 uporabljajo ožje reference (M3–M5 s povprečji, uteženimi po uporabnikih; povprečna ocena je neutežena), zato niso neposredno primerljivi z globalno referenco. Pri kontekstni referenci je pričakovani F1 naključnega rangiranja 0,532, na 398 uporabnikih z več kot petimi skladnimi kombinacijami pa 0,285 (SVD 0,804 oziroma 0,700; [extra_stats.txt](diagnostics/eval_fix_20261124/extra_stats.txt)). Visoki F1 z izenačenji so posledica po pravilu izračunanih ocen, ne dokaz uporabnosti.

Izvozi so v `outputs/heldout_evaluation/`: primerjave modelov v `evaluation/`, tabele in slike v `latex/`, priporočila in skupinske metrike v `recommendations/`. Imena vsebujejo časovni žig zagona.

[Zapis končnega zagona](outputs/heldout_evaluation/RUN_20261124_183727.md) vsebuje povezave na izvoze in dnevnik `full_run_v7_20261124_183727.log`, hashe ter omejitve; zagon z izbiro nastavitev opisuje [RUN_20261015_150006.md](outputs/heldout_evaluation/RUN_20261015_150006.md). Izvor pripravljenega `D_lst` ni potrjen; pozneje zajeti hashi ne dokazujejo vsebine datotek ob zgodovinskem zagonu.
