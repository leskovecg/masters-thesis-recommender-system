# SYSTEM OVERVIEW (private notes)

This document summarizes the **inputs, core artifacts, and recommendation flow**
of the Explainable Elderly Daily Activity Recommender System.
It is intended as a **technical orientation guide** for understanding the codebase
and for quickly explaining the system to a mentor.

---

## (1) Inputs

All inputs are provided as Excel files and loaded at the beginning of the pipeline.
They define users, activities, contexts, scores, and constraints.

### Main input files

- **ActivityContextGen_v11.xlsx**
  - Defines:
    - Activities (`actID`)
    - Allowed contexts per activity (time, physical condition, etc.)
  - Used to build:
    - `actID_context_dc`
    - `actID_qID_dc`

- **ml_data_with_scores_and_wgt_3_values.xlsx**
  - Contains:
    - User answers to questionnaire items
    - Normalized scores and weights
  - Used to compute:
    - User–activity score estimations

- **ContextGroups.xlsx**
  - Defines:
    - Context groupings (e.g. M3, M4, M5)
  - Used for:
    - Context-aware evaluation
    - Group-based analysis

- **Activities.xlsx**
  - Master list of activities
  - Activity metadata and identifiers

- **PhysicalHealth.xlsx**
- **MentalHealth.xlsx**
- **SocialHealth.xlsx**
- **IndependentLiving.xlsx**
  - Domain-specific questionnaires
  - Provide user answers used in score estimation

### Summary of input role

Inputs define:
- what activities exist
- which contexts are valid for each activity
- how users answered questionnaires
- how raw answers are transformed into scores
- which activity–context combinations are allowed

---

## (2) Core artifacts

These objects are the **backbone of the system** and are reused across training,
recommendation, and evaluation.

### `actID_context_dc`
- Dictionary: `actID → allowed contexts`
- Encodes hard feasibility constraints
- Used during recommendation filtering

### `D_lst`
- List of tuples:
  ```
  [userID, activity_sequence, context_sequence, rating]
  ```
- Represents the constructed user–activity interaction dataset
- Serves as:
  - training data proxy
  - evaluation ground truth proxy

### Surprise `data` and `trainset`
- `Dataset.load_from_df(...)`
- Internal Surprise representation of:
  - users
  - items (activities)
  - ratings
- Used by all CF algorithms

### `model`
- Trained collaborative filtering model (SVD, KNN, NMF, Baseline)
- Learns latent user–activity preferences
- Used to predict unseen activity scores

---

## (4) Recommending: top20 → context filter → top5

The recommendation process is executed **after model training**
and consists of two clear phases.

### Step 1: Generate top-20 recommendations (context-free)

For a given user:
- The trained Surprise model predicts scores for all activities
- Activities are ranked by predicted score
- The top 20 activities are selected

Result:
```
top20_recs = [(actID_1, score_1), ..., (actID_20, score_20)]
```

At this stage:
- Context is **not yet applied**
- Ranking is purely based on collaborative filtering

---

### Step 2: Context feasibility filtering

Each candidate activity from `top20_recs` is checked against the current context.

Filtering logic:
- Activity is kept if:
  - its allowed context matches the current context, OR
  - the activity is marked as context-independent (`kjerkoli`, `nd`)
- Context rules are defined in `actID_context_dc`

This step enforces **hard semantic constraints**.

---

### Step 3: Select final top-5 recommendations

- From the context-feasible subset:
  - activities are re-ranked by score
  - the top 5 are selected

Final output:
```
top5_recommendations = [(actID_a, score_a), ..., (actID_e, score_e)]
```

If fewer than 5 activities are context-feasible:
- fewer recommendations are returned
- this is expected behavior, not an error

---

## Mini diagram: recommendation flow (ASCII)

```
User ID + Context
        |
        v
Trained CF Model (SVD / KNN / ...)
        |
        v
Predict scores for all activities
        |
        v
Rank activities by score
        |
        v
Select TOP 20 (context-free)
        |
        v
Context feasibility filter
(actID_context_dc rules)
        |
        v
Select TOP 5 feasible activities
        |
        v
Final recommendations
```

---

## 30-second pitch for mentor

> The system is a context-aware recommender for elderly daily activities.
> First, we construct a user–activity interaction dataset from questionnaire
> responses and transform it into a collaborative filtering problem.
> We train standard CF models like SVD using Surprise.
> At recommendation time, the model proposes the top 20 activities purely
> based on learned preferences.
> Then, we apply a rule-based context filter that enforces feasibility constraints
> such as time of day or physical condition.
> The final output is a top-5 list of activities that are both personalized
> and context-appropriate, with explicit and explainable decision logic.

---
