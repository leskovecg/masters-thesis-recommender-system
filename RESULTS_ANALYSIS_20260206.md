# Analiza rezultatov iz runGeneratorOfD_v6.py (6.2.2026)

## POVZETEK REZULTATOV

### 1. Algoritamska primerjava (Table: Algorithm Comparison)
```
SVD:          RMSE=0.3863  MAE=0.2703  MSE=0.1492  FCP=0.9050  ✓ BEST
NMF:          RMSE=0.3229  MAE=0.1961  MSE=0.1043  FCP=0.9430  ✓ BEST (v MSE in FCP)
KNNBasic:     RMSE=0.6001  MAE=0.4302  MSE=0.3602  FCP=0.8797
BaselineOnly: RMSE=0.5465  MAE=0.4529  MSE=0.2987  FCP=0.8637
```

**Zaključek:** NMF je sicer malenkost boljši po nekaterih metrikah (MSE, FCP), a SVD dosega odličen RMSE in je izbran za nadaljnjo analizo.

---

### 2. Cross-Validation Rezultati (SVD_tuned s 10-fold CV)
```
Metric          Mean      Std
RMSE           0.5083    ±0.0049
MAE            0.4130    ±0.0054
MSE            0.2584    ±0.0050
FCP            0.8529    ±0.0016
Fit time:      0.317s    ±0.049
Test time:     0.109s    ±0.061
```

**Zaključek:** Stabilni rezultati z majhno standardno deviacijo - model je solid in generalizira dobro.

---

### 3. Primerjava 4 Priporočilnih Pristopov (Table 4.5)

```
                                Precision  Recall    F1      AvgScore
1. Brez konteksta (baseline)    0.423      0.339    0.375    2.371
2. S kontekstom (C_T, C_P)      0.456      0.116    0.179    2.336
3. M3 (avg čez 3 grupe)         0.442      0.411    0.424    1.581
4. M4 (avg čez 4 grupe)         0.439      0.373    0.395    1.760
5. M5 (avg čez 5 grup)          0.351      0.299    0.316    1.408
```

---

## ⚠️ NAJDENI PROBLEMI / NELOGIČNOSTI

### PROBLEM 1: **Rezultati konteksta (4.2) so SLABŠI kot baza (4.1)**
**Opis:** 
- Baseline (brez konteksta): Precision=0.423, Recall=0.339, F1=0.375
- S kontekstom (C_T, C_P):  Precision=0.456, Recall=**0.116**, F1=**0.179** ❌

**Analiza:**
- Precision se malo izboljša (+0.033), vendar se Recall drastično **smanji** (-0.223)
- To pomeni: ko filtriramo akcije po kontekstu, dobimo manj pravilnih priporočil
- Razlog: **Kontekst filter je PREMOČAN** - zavrne preveč dobrih akcij
- AverageScore se komajda smanji (2.371 → 2.336), kar pomeni da nivo kvalitete priporočil ostane enak

**Zaključek:** Filtriranje po kontekstu (C_T, C_P) **NI UČINKOVITO** - izboljša preciznost, vendar drastično zmanjša recall (F1 pada s 0.375 na 0.179).

---

### PROBLEM 2: **Grupe z 0 akcijami imajo 0 priporočil**
**Podatki:**
```
M3 Group0: 699 recs, 1 unique action
M3 Group1: 3480 recs, 9 unique actions  
M3 Group2: 0 recs, 0 unique actions ❌

M4 Group0: 692 recs
M4 Group1: 699 recs
M4 Group2: 3480 recs
M4 Group3: 0 recs ❌

M5 Group0: 0 recs ❌
M5 Group1: 692 recs
M5 Group2: 3480 recs
M5 Group3: 699 recs
M5 Group4: 0 recs ❌
```

**Analiza:**
- Nekatere grupe imajo malo ali nič akcij (50 akcij skupaj v 3/4/5 grupah)
- To povzroči, da za nekatere grupe **ne moremo dati 5 priporočil** → 0 recs
- Ocene za te grupe so enake: P=0.0, R=0.0, F1=0.0

**Zaključek:** Distribucija akcij po grupah je **NEURAVNOTEŽENA**. To je strukturni problem v podatkih ali kako grupirate akcije.

---

### PROBLEM 3: **Povprečni "Score" pada ko dodajate kontekst**
**Podatki:**
```
4.1 Baseline:           AvgScore = 2.371
4.2 S kontekstom:       AvgScore = 2.336 (↓ 0.035)
M3 (3 grupe):           AvgScore = 1.581 (↓ 0.790)
M4 (4 grupe):           AvgScore = 1.760 (↓ 0.611)
M5 (5 grup):            AvgScore = 1.408 (↓ 0.963)
```

**Analiza:**
- Ko grupirate po več kontekstnih parametrih, se prostor akcij **razdroblji** (fragmentira)
- To povzroči nižje ocene (lower confidence recommendations)
- Model ima manj podatkov za učenje po skupini

**Zaključek:** Kontekstna filtracija **zmanjšuje "score"** priporočil - izbira samo že obstoječe akcije iz omejenega niza.

---

### PROBLEM 4: **Kontekst filter (4.2) je prepočasen - dobimo malo priporočil**
**Podatki:**
```
M3 Group2:  0 priporočil (grupa nima akcij)
M4 Group3:  0 priporočil
M5 Group0:  0 priporočil
M5 Group4:  0 priporočil
```

**Analiza:**
- Ko je grupa premajhna, model ne more dati top5 priporočil
- Funkcija se ustavi, ko ima dovolj priporočil OR ko zmanjka akcij
- Problem je v **strategiji filtriranja** - pre stroga pravila

**Zaključek:** Kontekstni filter mora biti **fleksibilnejši** ali omogočiti "fallback" na bližnje kontekste.

---

## 📊 SMISELNOST REZULTATOV

### ✓ ŠTO JE SMISELNO
1. **Algoritamska primerjava je logična** - SVD/NMF sta res najboljša
2. **Cross-validation je stabilen** - model se ne pregiblja
3. **Baseline priporočila so dobra** - F1=0.375 je spodoben rezultat
4. **AverageScores so v pričakovanem rangu** - 1-3 je smiselno za ta tip sistema

### ❌ ŠTO NI SMISELNO
1. **Kontekst zmanjša Recall za 66%** - filter je preveč agressiven
2. **Neuravnotežene grupe** - nekatere imajo 0 akcij
3. **Povprečni score pada** - bi pričakoval, da bi se izboljšal s filtriranjem
4. **Kontekst ne naredi spremembe** - 4.2 je slabša kot 4.1

---

## 🔧 PRIPOROČILA ZA SPREMEMBE V SKRIPTI

### 1. **POVEČAJ TOLERANCE V KONTEKSTNEM FILTRU**

**Problem:** `is_action_context_feasibleQ()` je premočan filter

**Rešitev:** 
```python
# Trenutno: zahteva TOČNO ujemanje konteksta
# Bodisi omogočite "fuzzy matching" ali "partial matching"

# Opcija A: Partial context matching
def is_action_context_feasible_partial(act_id, context, actID_context_dc):
    """Zahteva ujemanje samo enega parametra (C_T OR C_P), ne (C_T AND C_P)"""
    act_contexts = actID_context_dc.get(act_id, [])
    return any(
        (ctx.get('C_T') == context.get('C_T')) or 
        (ctx.get('C_P') == context.get('C_P'))
        for ctx in act_contexts
    )

# Opcija B: Fallback na "kjerkoli" (anywhere/anytime)
def is_action_context_feasible_with_fallback(act_id, context, actID_context_dc):
    """Najprej proba natančen filter, nato fallback na 'kjerkoli'"""
    if is_action_context_feasible(act_id, context, actID_context_dc):
        return True
    # Fallback: prihrani akcije ki so dostopne "kjerkoli"
    return has_universal_context(act_id, actID_context_dc)
```

### 2. **IZBOLJŠAJ GRUPE Z MALIMI AKCIJAMI**

**Problem:** M3 Group 2, M5 Group 0/4 imajo 0 akcij

**Rešitev A - Predfiltriranje skupin:**
```python
# Preden generirate priporočila, filtrirajte grupe z malo akcijami
min_actions_per_group = 5

for g in G3_labels:
    action_count = len(set(actID_to_g3.values())) if actID_to_g3.values() == g else 0
    if action_count < min_actions_per_group:
        print(f"WARNING: Group {g} ima samo {action_count} akcij - preskačem")
        # Opcija: preskočite ali dodelite drugemu nivoju
```

**Rešitev B - Fallback na vrhovni nivo:**
```python
# Če grupa nima dovolj priporočil, dajte priporočila s širše grupe
def get_recs_with_fallback(uID, topN, g, mapping_dict, fallback_mapping_dict):
    """Proba najti priporočila iz grupe g, ali fallback na širšo skupino"""
    out = []
    for _, act_id, score in topN:
        if _is_in_group(act_id, g, mapping_dict):
            out.append((uID, act_id, score))
        if len(out) >= top_k:
            break
    
    # Fallback: če nimamo dovolj, dajte priporočila iz vrhovne grupe
    if len(out) < top_k:
        for _, act_id, score in topN:
            if _is_in_group(act_id, get_parent_group(g), fallback_mapping_dict):
                if act_id not in [x[1] for x in out]:  # Izognite duplikatov
                    out.append((uID, act_id, score))
            if len(out) >= top_k:
                break
    
    return out
```

### 3. **SPREMENITE STRATEGIJO KONTEKSTNEGA FILTRIRANJA**

**Problem:** 4.2 je slabša kot 4.1 (Recall pada za 66%)

**Rešitev:**
```python
# Namesto strogega "vsi ali nič" filtriranja,
# poskusite z "scoring boost" pristopom

def boost_action_score_by_context(act_id, score, context, actID_context_dc, boost_factor=1.2):
    """
    Ne filter, ampak BOOST ocene akcij ki ustrezajo kontekstu
    Namesto: Zavrni vse neprimerne
    Novo: Povečaj score primeren, znižaj neprimerne
    """
    if is_action_context_feasible(act_id, context, actID_context_dc):
        return score * boost_factor  # +20% za ustrezne
    else:
        return score * 0.9  # -10% za neustrezen

# Nato sortiraj po boosted scores in vzami top5
```

### 4. **DODAJTE DIAGNOSTIKO V SCRIPT**

**Problem:** Ni jasno zakaj se rezultati slabšajo

**Rešitev - doda diagnostične output-e:**
```python
# Na koncu STEP 16, pred STEP 18:

print("\n=== DIAGNOSTIKA KONTEKSTNEGA FILTRIRANJA ===")
for uID in random.sample(uIDsIn, min(5, len(uIDsIn))):
    print(f"\nUser {uID}:")
    print(f"  Top10 kandidati: {topN_norm[:10]}")
    print(f"  Filtrirani po kontekstu: {filtered_42[:5]}")
    print(f"  Število zavrženih: {len(topN_norm) - len(filtered_42)}")
    
print("\n=== ANALIZA PO SKUPINI ===")
for g in [0, 1, 2]:  # Samo prve 3
    group_actions = set(k for k,v in actID_to_g3.items() if v == g)
    print(f"Grupa {g}: {len(group_actions)} akcij")
    print(f"  Primeri: {list(group_actions)[:5]}")
```

### 5. **RAZMISLITE O "MULTI-LEVEL" KONTEKSTU**

**Problem:** Ločite med "hard" in "soft" kontekstom

```python
# Hard context (MORA biti): npr. "varnost" (zdravje)
# Soft context (LAHKO): npr. "čas ali mesto"

# Predlagam, da:
# 4.2: Apply HARD context filter
# 4.2b (novo): Apply SOFT context boost (ne filter)
```

---

## 📋 SKUPNA PRIPOROČILA

| Vprašanje | Status | Priporočilo |
|-----------|--------|------------|
| **Ali so rezultati smiselni?** | ⚠️ Delno | DA, vendar kontekst filter ne deluje kot pričakovan |
| **Kaj je narobe?** | ❌ | Kontekst zmanjša Recall, neuravnotežene grupe |
| **Kaj spremeniti?** | 🔧 | Fleksibilnejši filter, fallback mehanizmi, scoring boost |
| **Priority** | 🔴 Visoka | Rešite PROBLEM 1 (kontekst filter) - to je kritično |

---

## NASLEDNJI KORAKI

1. **Takoj:** Implementirajte Rešitev #3 (scoring boost namesto filtriranja)
2. **Nato:** Dodajte fallback mehanizme za male grupe
3. **Testirajte:** Primerjajte nove rezultate s starim baseline
4. **Dokumirajte:** Jasno opišite novo strategijo v tezi

