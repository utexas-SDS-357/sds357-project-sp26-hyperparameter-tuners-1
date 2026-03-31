# adding latitude and longitude

import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ==============================
# SETTINGS
# ==============================

INPUT_FILE = "df_housing_clean.csv"
OUTPUT_FILE = "df_housing_geocoded.csv"
CACHE_FILE = "geocode_cache.pkl"

ADDRESS_COLUMN = "property_address"

USER_AGENT = "caroline_nashville_housing_geocoder_v1"

# ==============================
# LOAD DATA
# ==============================

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

if ADDRESS_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{ADDRESS_COLUMN}' not found. Columns are: {df.columns.tolist()}"
    )

# Clean addresses
df[ADDRESS_COLUMN] = df[ADDRESS_COLUMN].astype(str).str.strip()

# Add city/state to improve geocoding accuracy
df[ADDRESS_COLUMN] = df[ADDRESS_COLUMN] + ", Nashville, TN"

# ==============================
# DEDUPLICATE ADDRESSES
# ==============================

unique_addresses = (
    df[ADDRESS_COLUMN]
    .dropna()
    .drop_duplicates()
)

print(f"Total rows: {len(df)}")
print(f"Unique addresses to geocode: {len(unique_addresses)}")

# ==============================
# LOAD CACHE (resume capability)
# ==============================

cache_path = Path(CACHE_FILE)

if cache_path.exists():
    print("Loading existing cache...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}

print(f"Cached addresses: {len(cache)}")

# ==============================
# SETUP GEOCODER + RATE LIMITER
# ==============================

geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1,     # Required by OSM policy
    max_retries=3,
    error_wait_seconds=5,
    swallow_exceptions=True
)

# ==============================
# GEOCODING FUNCTION
# ==============================

def geocode_address(address):

    if address in cache:
        return cache[address]

    location = geocode(address)

    if location:
        result = (location.latitude, location.longitude)
    else:
        result = (None, None)

    cache[address] = result
    return result

# ==============================
# RUN GEOCODING
# ==============================

print("\nStarting geocoding (this may take a while)...")

results = {}

for i, addr in enumerate(tqdm(unique_addresses)):
    results[addr] = geocode_address(addr)

    # Save cache every 100 queries
    if i % 100 == 0:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)

# Final cache save
with open(CACHE_FILE, "wb") as f:
    pickle.dump(cache, f)

print("Geocoding finished.")

# ==============================
# CREATE COORDINATE TABLE
# ==============================

coords = pd.DataFrame.from_dict(
    results,
    orient="index",
    columns=["latitude", "longitude"]
)

coords.index.name = ADDRESS_COLUMN
coords.reset_index(inplace=True)

# ==============================
# MERGE BACK INTO DF_HOUSING
# ==============================

print("Merging coordinates back into df_housing_clean...")

df_final = df.merge(coords, on=ADDRESS_COLUMN, how="left")

# ==============================
# SAVE OUTPUT
# ==============================

df_final.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Done! Saved to: {OUTPUT_FILE}")

import os
os.chdir("/Users/caroline/Desktop/SDS 357/Data")
print(os.listdir())

import pandas as pd
df_housing_geocoded = pd.read_csv("df_housing_geocoded.csv")

non_blank_count = df_housing_geocoded['latitude'].count()
print(non_blank_count)

df_final.to_csv("df_housing_geocoded.csv", index=False)
print("\n✅ Done! Saved to: df_housing_geocoded.csv")

