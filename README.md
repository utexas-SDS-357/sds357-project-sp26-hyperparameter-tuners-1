# Geocoding Branch

This branch contains the geocoding pipeline that converts Nashville housing property addresses into latitude/longitude coordinates, enabling spatial alignment with the SOPP traffic stop data.

## Branch Contents

```
geocoding/
├── .github/                     ← GitHub Classroom feedback configuration
├── geocoding.py                 ← Main geocoding script (addresses → lat/lon)
├── geocoding_complete.txt       ← Log file confirming geocoding completion
└── README.md                    ← This file
```

## What This Branch Does

### `geocoding.py`
Reads the raw Nashville Housing dataset and uses the **OpenStreetMap Nominatim API** to convert each property address into latitude and longitude coordinates. The geocoded output feeds directly into the wealth index construction in the `cleaning-and-wrangling` branch.

Key steps performed:
- Loads raw Nashville housing CSV
- Iterates through each property address and calls the Nominatim API
- Assigns latitude and longitude to each record
- Outputs the geocoded dataset as `df_housing_clean.csv`

### `geocoding_complete.txt`
A log file confirming that the geocoding process ran to completion. Since geocoding ~24,000 addresses takes several hours due to API rate limits, this file serves as a checkpoint so the process doesn't need to be re-run unnecessarily.

## Required Input Data

You will need the raw Nashville Housing dataset before running this script:

**Source:** [https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data](https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data)

Download it and place it in the same directory as `geocoding.py`.

## How to Run

```bash
# 1. Switch to this branch
git checkout geocoding

# 2. Install dependencies
pip install pandas geopy tqdm certifi

# 3. Run the geocoding script
python geocoding.py
```

> ⚠️ **Important:** Geocoding ~24,000 addresses takes several hours due to Nominatim API rate limits. The output `df_housing_clean.csv` is already available in the `datasets` branch — only re-run this script if you need to reproduce geocoding from scratch.

## Output

Running `geocoding.py` produces:

- **`df_housing_clean.csv`** — the cleaned Nashville Housing dataset with geocoded latitude/longitude coordinates, ready for wealth index construction

## Relationship to Other Branches

| Branch | Relationship |
|--------|-------------|
| `datasets` | Provides the raw Nashville Housing input data |
| `cleaning-and-wrangling` | Consumes `df_housing_clean.csv` to build the wealth index |
| `main` | This branch merges into `main` when complete |
