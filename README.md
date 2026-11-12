# Kontekstualiziran priporočilni sistem za dnevne aktivnosti starejših

Magistrsko delo, Fakulteta za elektrotehniko Univerze v Ljubljani  
**Avtor:** Gašper Leskovec · **Mentor:** prof. dr. Andrej Košir

Projekt obravnava priporočanje posameznih aktivnosti in kombinacij do treh aktivnosti starejšim. Model SVD napoveduje ocene, izračunane iz vprašalniških odgovorov, uporabniških uteži in združljivosti aktivnosti. Naknadni filtri omejijo izbor glede na čas, kraj ali pripadnost aktivnosti kontekstni skupini.

## Postopek

Končna izvedba je [runGeneratorOfD_v7.py](src/runGeneratorOfD_v7.py). Uporablja modul [elderly_recsys_tools.py](src/elderly_recsys_tools.py) za pripravo podatkov in vrednotenje ter [recommendation_items.py](src/recommendation_items.py) za obravnavo celotnih kombinacij in njihovo filtriranje.

1. Skripta prebere vhodne podatke in naloži pripravljeni nabor ocen `D_lst`.
2. Pred izbiro modela odloži 25 % ocen za končni test. Na razvojnem delu primerja SVD, KNNBasic, BaselineOnly in NMF ter poišče nastavitve SVD.
3. Končni SVD nauči na razvojnem delu in napove ocene testnih parov uporabnik–kombinacija.
4. Za vsakega uporabnika razvrsti največ 20 testnih kandidatov. Vrne največ pet kombinacij brez filtriranja, s filtrom časa in kraja ali ločeno po skupinah M3–M5.

Pri kontekstnem filtru morajo vse aktivnosti kombinacije ustrezati istemu simuliranemu času in kraju. Pri skupinskem filtru morajo vse pripadati isti skupini. Seznama po filtriranju ne dopolnjuje z drugimi kandidati, zato je lahko krajši ali prazen.

## Namestitev

Uporabljen je Python 3.11. Ukaze zaženi v PowerShellu iz korena projekta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Različice v `requirements.txt` izhajajo iz popisa lokalnega okolja 21. 10. 2026. Namestitev s tem seznamom v sveže okolje še ni preverjena.

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

Nastavitvi sta v začetnem delu skripte:

- `USE_EXISTING_D_LST = True` naloži pripravljeni nabor. Vrednost `False` sproži generiranje novega nabora iz vhodov.
- `test_mode = False` izvede polni eksperiment. Vrednost `True` omeji nabor na prvih 100.000 zapisov in mrežo nastavitev na eno kombinacijo; deset ponovitev preverjanja ostane. To je preizkus delovanja, ne reprezentativen vzorec celotnega nabora.

Polni zagon vključuje učenje in iskanje nastavitev ter lahko traja več ur. Zagon 15. 10. 2026 je trajal približno 5 h 43 min. Za pregled obstoječih rezultatov ga ni treba ponavljati.

Obstoječi test pomožnega modula:

```powershell
.\.venv\Scripts\python.exe -B tests\test_recommendation_items.py -v
```

## Rezultati in njihova razlaga

Končni zagon ima oznako **`20261015_150006`** in obsega **679 uporabnikov, 6.579 kombinacij ter 3.053.029 ocen**. Razvojni del vsebuje 2.289.771 ocen, testni pa 763.258.

SVD na odloženem testnem delu doseže **RMSE 0,0819**, **MAE 0,0531** in **FCP 0,9966**. Brez kontekstnega filtra sistem vrne 3.395 priporočil, s filtrom pa 250.

| Različica | Natančnost | Priklic | F1 |
|---|---:|---:|---:|
| Brez konteksta | 0,339 | 0,339 | 0,339 |
| S kontekstom časa in kraja | 0,006 | 0,006 | 0,006 |
| M3 | 0,132 | 0,129 | 0,130 |
| M4 | 0,110 | 0,063 | 0,073 |
| M5 | 0,077 | 0,051 | 0,057 |

Ocene so izračunane po pravilih, zato napake napovedovanja merijo ujemanje s temi ocenami. Rezultati ne potrjujejo zadovoljstva uporabnikov ali zdravstvenih učinkov. Na metrike top-5 vpliva tudi leksikografsko razreševanje izenačenih referenčnih ocen. M3–M5 uporabljajo reference znotraj skupin in neutežena povprečja skupinskih metrik, zato niso neposredno primerljivi z globalno referenco prvih dveh različic.

Izvozi so v `outputs/heldout_evaluation/`: primerjave modelov v `evaluation/`, tabele in slike v `latex/`, priporočila in skupinske metrike v `recommendations/`. Imena vsebujejo časovni žig zagona.

[Zapis končnega zagona](outputs/heldout_evaluation/RUN_20261015_150006.md) vsebuje povezave na izvoze in dnevnik `full_run_v7_20261015_150004.log`, hashe ter omejitve sledljivosti. Izvor pripravljenega `D_lst` ni potrjen; pozneje zajeti hashi ne dokazujejo vsebine datotek ob zgodovinskem zagonu.
