## Daily Activity Recommender System for the Elderly

Master's thesis — Gašper Leskovec, Mentor: Andrej Košir  
Faculty of Electrical Engineering, University of Ljubljana

---

## Project Structure

The repository is organized to clearly separate **input data**, **generated artifacts**, **source code**, and **thesis materials**, ensuring reproducibility and clarity throughout the thesis development.

```
THESIS/
├── data/                          # All data used in the project
│   ├── annotations/              # Expert-annotated reference data (qualitative validation)
│   ├── context/                  # Activity context definitions and context groupings
│   │   └── legacy/               # Older or deprecated context versions
│   ├── D_lst/                    # Generated sparse user–activity datasets
│   │   └── legacy/               # Older generated dataset variants
│   ├── questionnaires/           # Original survey input data (raw questionnaires)
│   └── scores/                   # User scores, weights, and scoring configurations
│
├── literature/                   # Scientific papers and related work (PDFs)
│
├── logs/                         # Runtime logs and profiling information
│
├── notebooks/                    # Jupyter notebooks for analysis and exploration
│
├── outputs/                      # Automatically generated experimental outputs
│   ├── evaluation/               # Quantitative evaluation results and metrics
│   │   └── legacy/               # Older evaluation outputs
│   ├── latex/                    # Auto-generated LaTeX figures and tables
│   │   ├── figs/                 # Figures generated from experiments
│   │   │   └── legacy/           # Older generated figures
│   │   └── tabs/                 # Tables generated from experiments
│   │       └── legacy/           # Older generated tables
│   └── recommendations/          # Generated activity recommendations
│       └── legacy/               # Older recommendation outputs
│
├── src/                          # Source code
│   ├── legacy_versions/          # Older script versions kept for reference
│
├── .gitattributes                # Git configuration
├── .gitignore                    # Files excluded from version control
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies

```
