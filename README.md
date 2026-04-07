# Cleaning and Wrangling Branch

This branch contains all data cleaning, preprocessing, and wealth index construction for the Nashville SOPP project. It transforms the raw SOPP and Nashville Housing datasets into a single model-ready dataframe.

## Branch Contents

```
cleaning-and-wrangling/
├── .github/                     ← GitHub Classroom feedback configuration
├── housing_cleaning.ipynb       ← Cleaning pipeline for Nashville Housing dataset
├── main_cleaning.ipynb          ← Cleaning pipeline for SOPP traffic stop dataset
├── wrangling.ipynb              ← Wealth index construction and dataset merging
└── README.md                    ← This file
```

## What This Branch Does

### `housing_cleaning.ipynb`
Cleans the raw Nashville Housing dataset. Key steps:
- Drops rows with missing property address or sale price values
- Removes non-essential columns
- Rounds latitude/longitude to 6 decimal places for spatial consistency
- Outputs cleaned housing data ready for wealth index construction

### `main_cleaning.ipynb`
Cleans the raw SOPP Nashville traffic stop dataset. Key steps:
- Filters observations to 2013–2016 to align temporally with housing data
- Retains only key variables: date, latitude, longitude, subject age, subject race, subject sex, stop outcome
- Drops rows with missing values in key columns
- Encodes categorical variables:
  - Race: White=1, Black=2, Hispanic=3, Asian/Pacific Islander=4, Unknown=5, Other=6
  - Sex: Male=1, Female=2
  - Outcome: Warning=1, Citation=2, Arrest=3
- Standardizes latitude/longitude to 6 decimal places

### `wrangling.ipynb`
Constructs the neighborhood wealth index and merges it into the policing dataset. Key steps:
- Builds a BallTree spatial index from geocoded housing sale coordinates
- Computes median housing sale price within three radius definitions per traffic stop:
  - 0.5-mile radius
  - 1-mile radius ← **selected as final WealthIndex**
  - KNN-30 (nearest 30 sales)
- Uses only housing sales from years ≤ the stop year to prevent data leakage
- Log-transforms the median price to reduce skew
- Merges wealth index into the main policing dataframe
- Outputs `main_df_final.csv`

The 1-mile radius was selected as the final wealth index because it provides the best balance between stability and geographic interpretability (see Figure 2 in the midterm report).

## Required Input Data

Before running these notebooks you will need:

1. **Raw SOPP Nashville data** — download from [Stanford Open Policing Project](https://openpolicing.stanford.edu/data/) (scroll to Tennessee → Nashville)
2. **`df_housing_clean.csv`** — produced by the `geocoding` branch, also available in the `datasets` branch

## How to Run

```bash
# 1. Switch to this branch
git checkout cleaning-and-wrangling

# 2. Install dependencies
pip install pandas numpy scikit-learn jupyter

# 3. Run notebooks in order
jupyter notebook housing_cleaning.ipynb
jupyter notebook main_cleaning.ipynb
jupyter notebook wrangling.ipynb
```

## Output

Running all three notebooks in order produces:

- **`main_df_final.csv`** — the final merged dataset (~1.3 million rows) with encoded demographic variables and neighborhood wealth index, ready for EDA and modeling

## Relationship to Other Branches

| Branch | Relationship |
|--------|-------------|
| `geocoding` | Provides `df_housing_clean.csv` as input |
| `datasets` | `main_df_final.csv` is stored there for access by other branches |
| `EDA-and-modeling` | Consumes the final merged dataset produced here |
| `main` | This branch merges into `main` when complete |
