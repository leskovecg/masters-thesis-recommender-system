# DATA & PIPELINE OVERVIEW

This document gives a **full, engineering-style overview of all datasets and their role in the recommender system pipeline**.
Its purpose is to clearly answer:
- what each dataset is,
- why it exists,
- where it comes from,
- and how it connects to the others.

---

## 1. Conceptual Big Picture

The system follows a **layered data pipeline**:

1. Questionnaires (raw human input)
2. Weighted & aggregated scores
3. Discretized ML-ready features
4. Activity-level scores
5. Recommender dataset (Surprise input)
6. Contextual filtering & compatibility reranking

---

## 2. Activity & Context Definition Layer

### ActivityContextGen_v11.xlsx

**Role:**  
Defines the *world of possible activities* and their constraints.

**Contains:**
- ActionLst – canonical activity list (actID)
- Context feasibility (act_C_T*, act_C_P*, act_C_A*)
- Recommendation hints (rec_C_*, currently unused)
- Compatibility groups (compGrp)
- Context groups (contextG_M3 … M7)
- Action groups (semantic grouping)

**Purpose:**
- Defines what activities exist
- Defines when they are allowed
- Defines how activities relate semantically

**Key output artifact:**
- actID_context_dc
- Compatibility rules used for reranking

---

## 3. Raw Questionnaires (Human Input Layer)

### PhysicalHealth.xlsx  
### MentalHealth.xlsx  
### SocialHealth.xlsx  
### IndependentLiving.xlsx  

**Role:**  
Raw questionnaire responses describing different aspects of a person.

**Rows:**  
- One row = one person

**Columns:**  
- Individual questionnaire questions (Likert, yes/no, frequency)

**Purpose:**
- Capture physical, mental, social and independence dimensions
- Serve as raw signals (not used directly by ML)

---

## 4. Weighted / Aggregated Score Layer

### wgt_results_annotations_3009_4users.xlsx

**Role:**  
Aggregated and weighted Likert-based scores per person.

**Key columns:**
- person_id
- likert_value_*
- *_wgt (e.g. ha_wgt)

**Purpose:**
- Reduce noise from individual questions
- Produce stable composite indicators
- Provide interpretable domain-level scores

**Usage:**
- Feature engineering
- Calibration
- Validation (not ML training)

---

## 5. Feature Consolidation Layer

### ml_data_scores_and_wgt.xlsx

**Role:**  
Unified feature table containing all numerical user signals.

**Rows:**  
- One row = one person

**Columns:**  
- Weighted scores (*_wgt)
- Intermediate scores from multiple domains

**Purpose:**
- Centralized feature view
- Sanity-check point before discretization

---

## 6. Discretization Layer

### ml_data_with_scores_and_wgt_3_values.xlsx

**Role:**  
Discretized version of ML features.

**Key idea:**
- Continuous values → discrete levels (e.g. 1 / 2 / 3)

**Why discretization is needed:**
- Context rules are discrete
- Activity feasibility is discrete
- Easier explainability

**Purpose:**
- Bridge between numeric scoring and rule-based logic
- Prepare user profiles for context matching

---

## 7. Activity-Level Scoring Layer

### Activities.xlsx

**Role:**  
User–activity score matrix.

**Rows:**  
- Users (person_id)

**Columns:**  
- Activities (actID)

**Cell value:**  
- Numeric suitability score of an activity for a user

**Purpose:**
- Translate user profiles into activity preferences
- Provide activity-level data for recommender modeling

---

## 8. Recommender Dataset Layer

### score_D_df.xlsx

**Role:**  
Final long-format recommender dataset.

**Format:**
user_id | actID | score

**Purpose:**
- Direct input for recommender models (Surprise)
- Basis for building D_lst and training datasets

---

## 9. Machine Learning Layer

### D_lst (.pkl)

**Role:**  
Python list / DataFrame containing recommender-ready data.

**Used to build:**
- Surprise Dataset
- Surprise Trainset

**Models trained:**
- SVD
- KNN
- Baseline
- NMF

---

## 10. Recommendation Generation Flow

User profile  
↓  
ML prediction (top-20 activities)  
↓  
Context feasibility filter (act_C_*)  
↓  
Compatibility reranking (compGrp + matrix)  
↓  
Final top-5 recommendations  

---

## 11. Key Design Principles

- Questionnaires are never used directly for ML
- All ML sees activity-level scores
- Context acts as a hard constraint
- Compatibility acts as a semantic coherence layer
- Discretization enables explainability and rule-based logic

---

## 12. One-Sentence System Summary

The system transforms raw questionnaire responses into weighted and discretized user profiles, maps them onto activity-level preferences, trains collaborative filtering models on user–activity scores, and finally applies contextual feasibility and activity compatibility rules to generate coherent daily activity recommendations.
