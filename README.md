# Nashville SOPP: Effect of Neighborhood Wealth on Traffic Stop Outcomes

> **SDS 357: Case Studies in Data Science** | University of Texas at Austin | Spring 2026

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Data](https://img.shields.io/badge/Data-Stanford%20Open%20Policing%20Project-orange)](https://openpolicing.stanford.edu/data/)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Setup and Installation](#setup-and-installation)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Methodology Summary](#methodology-summary)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [References](#references)

---

## Project Overview

This project uses the [Stanford Open Policing Project (SOPP)](https://openpolicing.stanford.edu/data/) dataset to examine traffic stop outcomes in Nashville, Tennessee. We investigate whether **neighborhood wealth** — proxied by a constructed wealth index derived from local housing sale prices — is associated with traffic stop outcomes (warnings, citations, arrests), and whether that association differs across demographic groups (race, sex, age).

Our primary statistical model is a **Multinomial Logistic Regression (MLR)**, chosen for its interpretability and suitability for nominal categorical outcomes. The analysis is inferential in nature: rather than optimizing predictive accuracy, we aim to estimate and interpret the direction and magnitude of associations between neighborhood wealth, demographic characteristics, and stop outcomes.

**Midterm Report:** [📄 View the Midterm Report](https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1/blob/main/SDS357_%20Midterm_Report.pdf)

---

## Team Members

| Name | EID | GitHub |
|------|-----|--------|
| Viren Halaharivi | vsh353 | [vhalarivi26](https://github.com/vhalaharivi26) |
| John Reynoldson | jsr3598 | [jreynoldson21](https://github.com/jreynoldson21) |
| Vidhi Sapru | vs26487 | [vidhisapru](https://github.com/vidhisapru) |
| Rachel Nguyen | rtn398 | [racheltnguyen07](https://github.com/racheltnguyen07) |
| Caroline Langenkamp | cl48358 | [carolinelangenkamp](https://github.com/carolinelangenkamp) |

---

## Repository Structure

```
sds357-project-sp26-hyperparameter-tuners-1/
│
├── README.md                    ← You are here
│
├── data/                        ← Raw source data (see Datasets section for downloads)
│
├── df_housing_clean.csv         ← Cleaned Nashville Housing dataset with geocoded coordinates
├── main_df_final.csv.zip        ← Final merged, model-ready dataset (~1.3M traffic stop rows)
│
├── housing_cleaning.ipynb       ← Data cleaning of the Nashville housing dataset
├── main_cleaning.ipynb          ← Data cleaning of the SOPP dataset
├── wrangling.ipynb              ← Data wrangling, geocoding, and wealth index construction
├── eda.ipynb                    ← Exploratory data analysis and figures
└── modeling.ipynb               ← MLR model estimation, assumption checks, and results
```

> **Note:** The raw SOPP traffic stop data is not committed to this repository due to file size. See the [Datasets](#datasets) section for download instructions. The processed outputs (`df_housing_clean.csv` and `main_df_final.csv.zip`) are included so you can run `eda.ipynb` and `modeling.ipynb` directly without rerunning the full pipeline. However, the data cleaning notebooks (`housing_cleaning.ipynb` and `main_cleaning.ipynb`) are included for transparency and reproducibility.

---

## Datasets

### 1. Stanford Open Policing Project — Nashville, TN (Primary Dataset)

**Source:** [https://openpolicing.stanford.edu/data/](https://openpolicing.stanford.edu/data/)

**Download instructions:**
1. Navigate to the link above.
2. Scroll down to the **Tennessee** section.
3. Click **Nashville** and download the raw `.csv` file.
4. Place the file inside the `data/` folder in this repository.

**Description:** Official Nashville Police Department traffic stop records. Includes stop date, location (latitude/longitude), driver demographics (race, age, sex), and stop outcome (warning, citation, arrest). Raw dataset: ~3.8 million rows. After filtering to 2013–2016 and dropping rows with missing key variables, the working dataset contains ~1.3 million rows.

---

### 2. Nashville Housing Dataset (Supplemental Dataset)

**Source:** [https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data](https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data)

**License:** CCO: Public Domain — free to use, modify, and redistribute without permission.

**Download instructions:**
1. Navigate to the Kaggle link above (a free Kaggle account is required).
2. Click **Download** to get the dataset.
3. Place the file inside the `data/` folder in this repository.

> **Note:** A cleaned version with geocoded coordinates is already available in this repo as [`df_housing_clean.csv`](https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1/blob/main/df_housing_clean.csv). You only need to re-download and re-geocode if you want to reproduce the geocoding step from scratch.

**Description:** Nashville property transaction records including sale prices, addresses, and sale dates (~56,000 rows raw, ~24,000 after cleaning). Used to construct a neighborhood-level wealth index by geocoding addresses to latitude/longitude and computing the median sale price within a 1-mile radius of each traffic stop.

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the repository

```bash
git clone https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1.git
cd sds357-project-sp26-hyperparameter-tuners-1
```

### Step 2 — Install dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn geopy tqdm certifi patsy jupyter
```

The key packages used in this project are:

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and cleaning |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `statsmodels` | MLR inference (coefficients, cluster-robust SEs, p-values) |
| `scikit-learn` | MLR out-of-sample validation |
| `geopy` | Geocoding via OpenStreetMap Nominatim API |
| `tqdm` | Progress bars for geocoding loop |
| `certifi` | SSL certificate handling for API calls |
| `patsy` | Spline basis construction for age term |

### Step 3 — Download the raw SOPP data

Follow the instructions in [Datasets → Dataset 1](#1-stanford-open-policing-project--nashville-tn-primary-dataset) and place the file in the `data/` folder. The cleaned housing data and merged main dataframe are already in the repo, so you can skip straight to running the notebooks if you don't need to reproduce the geocoding step.

---

## Reproducing the Analysis

Pipeline overview: Data Cleaning → Data Wrangling → EDA → Modeling

The included notebooks should be run in the following order. If you are starting from the pre-processed files already in the repo (`df_housing_clean.csv` and `main_df_final.csv.zip`), you can skip steps 1-2 and begin with Step 3.

### Step 1 - Data Cleaning

**Notebooks:** [`housing_cleaning.ipynb`](https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1/blob/main/housing_cleaning.ipynb)
[`main_cleaning.ipynb`](https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1/blob/main/main_cleaning.ipynb)

```bash
jupyter notebook housing_cleaning.ipynb
jupyter notebook main_cleaning.ipynb
```

These notebooks:
- Drops unnecessary columns of the Nashville housing dataset
- Rounds latitude and longitude values to six decimals in the Nashville housing dataset
- Removes NA values from key variable columns of the Nashville housing dataset
- Filters the raw SOPP dataset to 2013–2016 and drops rows with missing key variables
- Drops unnecessary columns of the raw SOPP dataset
- Rounds latitude and longitude values to six decimals in the raw SOPP dataset
- Changes datatypes of certain columns of the raw SOPP dataset so it is ready for analysis
- Encodes categorical variable of outcome into numeric format

---

### Step 2 — Data Wrangling, Geocoding, and Wealth Index Construction

**Notebook:** [`wrangling.ipynb`](https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1/blob/main/wrangling.ipynb)

```bash
jupyter notebook wrangling.ipynb
```

This notebook:
- Encodes categorical variables (race, sex) into numeric format
- Geocodes Nashville housing addresses via the OpenStreetMap Nominatim API (see ⚠️ below)
- Constructs the wealth index as the median sale price within a 1-mile radius of each traffic stop
- Merges the wealth index into the policing dataset and outputs `main_df_final.csv.zip`

> ⚠️ **Geocoding note:** Converting ~24,000 addresses takes significant time due to Nominatim API rate limits. The output `df_housing_clean.csv` is already included in the repo — you can skip this step entirely.

---

### Step 3 — Exploratory Data Analysis

**Notebook:** [`eda.ipynb`](https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1/blob/main/eda.ipynb)

```bash
jupyter notebook eda.ipynb
```

This notebook generates the following figures from the midterm report:
- **Figure 1** — Distribution of traffic stop outcomes (warning, citation, arrest)
- **Figure 2** — Comparison of wealth index distributions across neighborhood definitions (0.5mi, 1mi, KNN-30)
- **Figure 3** — Spatial map of log-scaled median housing prices across Nashville
- **Figure 4** — Boxplots of wealth index by stop outcome

**Input:** `main_df_final.csv.zip` (already in repo)

---

### Step 4 — Model Estimation and Results

**Notebook:** [`modeling.ipynb`](https://github.com/utexas-SDS-357/sds357-project-sp26-hyperparameter-tuners-1/blob/main/modeling.ipynb)

```bash
jupyter notebook modeling.ipynb
```

This notebook:
- Runs assumption checks: VIF analysis, Box–Tidwell linearity tests, IIA sensitivity analysis
- Fits the MLR on a stratified sample from 2013–2015 (training set) using `statsmodels`
- Uses cluster-robust standard errors (clustered by year × geographic grid cell)
- Models age with a natural cubic spline (df=4); wealth retained as log-linear
- Includes wealth × race and wealth × sex interaction terms
- Generates coefficient tables, odds ratios, p-values, and z-values (Figures 5–8)
- Generates predicted probability plots by wealth index, race, sex, and age (Figures 9–11)
- Validates on 2016 test set using `scikit-learn`

**Input:** `main_df_final.csv.zip` (already in repo)

---

## Methodology Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Model | Multinomial Logistic Regression | 3-class nominal outcome; inferential goal |
| Baseline outcome | Warning | Most common outcome; compares citation and arrest against it |
| Wealth proxy | Median sale price within 1-mile radius | Balance of stability and geographic interpretability |
| Wealth transformation | Log-transformed, mean-centered | Reduces skew; centers main effects for interactions |
| Age specification | Natural cubic spline (df=4) | Box–Tidwell test flagged nonlinearity |
| Standard errors | Cluster-robust (year × lat/lon grid) | Accounts for spatial and temporal dependence |
| Train/test split | Temporal (2013–2015 train, 2016 test) | Avoids spatial leakage from a random split |
| Interactions | Wealth × race, Wealth × sex | Tests whether wealth effect is heterogeneous across groups |

---

## Key Findings

- **Demographic factors dominate.** Race and sex are the strongest predictors of stop outcome. Hispanic individuals show over 2× higher odds of citation and nearly 3× higher odds of arrest compared to White individuals. Black individuals face more than 2× higher odds of arrest. Female subjects have significantly lower odds of both citations (~13% lower) and arrests (~59% lower) than males.

- **Neighborhood wealth alone is not statistically significant** as a main effect on stop outcomes (p = 0.068 for citation, p = 0.46 for arrest).

- **The wealth effect is heterogeneous by race.** For White and Hispanic drivers, higher neighborhood wealth is associated with a *lower* probability of citation. For Black drivers, the relationship is reversed: a 10% increase in neighborhood housing prices is associated with approximately a 1–2% *increase* in the odds of citation — a statistically significant interaction (p = 0.011).

- **Younger drivers face higher citation probabilities** across all wealth levels, though the direction of the wealth–citation relationship is consistent across age groups.

These findings suggest a potential systemic pattern in Nashville policing that disproportionately impacts Black residents in wealthier neighborhoods.

---

## Limitations

1. **IIA assumption** — The MLR model assumes independence of irrelevant alternatives. Our sensitivity analysis suggests this may not hold perfectly, as stop outcomes likely arise from a sequential escalation process rather than independent choices.

2. **Simultaneity / endogeneity** — The wealth index (housing prices) and policing intensity may influence each other. Future work could use an Instrumental Variable approach (Two-Stage Least Squares) to assess endogeneity.

3. **Sample size for inference** — The inferential MLR was fit on a stratified sample of ~75,000 observations (out of ~1.3 million) due to computational constraints. Expanding to the full dataset is planned.

4. **Unknown race category** — A sensitivity analysis to assess the impact of excluding the "Unknown" race category is planned for future work.

5. **Observational data** — This study is associational, not causal. Unmeasured confounders may explain part of the observed patterns.

---

## References

- Stanford Open Policing Project. (n.d.). *Traffic stop data: Nashville, TN.* Stanford University. [https://openpolicing.stanford.edu/data/](https://openpolicing.stanford.edu/data/)

- tmthyjames. (n.d.). *Nashville housing data.* Kaggle. [https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data](https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data)

---

*This project was completed as part of SDS 357: Case Studies in Data Science at the University of Texas at Austin, Spring 2026.*
