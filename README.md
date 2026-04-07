# Datasets Branch

This branch stores the cleaned and processed datasets used across the Nashville SOPP project. It serves as the central data repository for all other branches.

## Branch Contents

```
datasets/
├── .github/                     ← GitHub Classroom feedback configuration
├── df_housing_clean.csv         ← Cleaned Nashville Housing dataset with geocoded coordinates
├── main_df_final.csv.zip        ← Final merged dataset (policing + wealth index, ~1.3M rows)
└── README.md                    ← This file
```

## Datasets

### `df_housing_clean.csv`
The cleaned Nashville Housing dataset, produced by the `geocoding` branch. Contains property sale records with latitude/longitude coordinates assigned via the OpenStreetMap Nominatim API.

**Key columns:**
- `sale_price` — property sale price in USD
- `sale_date` — date of sale
- `lat` / `lon` — geocoded coordinates of the property address

**Source:** Originally from [Kaggle — Nashville Housing Data](https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data) (CCO: Public Domain)

---

### `main_df_final.csv.zip`
The final model-ready dataset used for all EDA and modeling. This is the merged output of the SOPP traffic stop data + the neighborhood wealth index constructed in the `cleaning-and-wrangling` branch.

**Key columns:**
- `lat` / `lon` — stop location coordinates
- `stop_date` / `stop_year` — date of traffic stop
- `subject_race_encd` — encoded race (White=1, Black=2, Hispanic=3, Asian/Pacific Islander=4, Unknown=5, Other=6)
- `subject_sex_encd` — encoded sex (Male=1, Female=2)
- `outcome_encd` — encoded stop outcome (Warning=1, Citation=2, Arrest=3)
- `subject_age` — driver age
- `WealthIndex` — log median housing sale price within 1-mile radius of stop

**Rows:** ~1.3 million traffic stop observations (2013–2016)

**Note:** The raw SOPP traffic stop data is not stored in this repo due to file size. Download it directly from the [Stanford Open Policing Project](https://openpolicing.stanford.edu/data/) (scroll to Tennessee → Nashville).

## How to Access

To use these datasets in another branch, either:

**Option 1 — Download directly from GitHub**
Navigate to this branch on GitHub and download the files manually.

**Option 2 — Pull via git**
```bash
# From inside your local repo
git checkout datasets
cp df_housing_clean.csv ../your-working-directory/
cp main_df_final.csv.zip ../your-working-directory/
git checkout your-branch
```

## Relationship to Other Branches

| Branch | Relationship |
|--------|-------------|
| `geocoding` | Produces `df_housing_clean.csv` → stored here |
| `cleaning-and-wrangling` | Produces `main_df_final.csv.zip` → stored here |
| `EDA-and-modeling` | Reads `main_df_final.csv.zip` from here |
| `main` | This branch merges into `main` when complete |
