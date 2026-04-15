"""
SolarHive — Synthetic Training Data Generator
==============================================
Calls all four API services (Open-Meteo, NREL PVWatts, OWM, EIA),
validates in pandas DataFrames, and generates ~520 data-grounded
training examples for fine-tuning.

No GPU required — pure data pipeline (CPU-only Colab).

Gemma is a trademark of Google LLC.

Cell structure:
  0: Dependencies
  1: API Keys + Constants
  2: Open-Meteo Historical (free, no key)
  3: NREL PVWatts v8 (uses NREL_API_KEY)
  4: OWM Current Weather (uses OWM_API_KEY)
  5: EIA Grid Data (uses EIA_API_KEY)
  6: DataFrame EDA, Validation & Cross-Validation
  6b: Visualizations (12 charts: distributions, heatmaps, time series)
  7: Synthetic Example Generation (~520 examples)
  8: Export
"""

"""## 0: Dependencies"""

# === CELL 0: Dependencies ====================================================
# All pre-installed on Colab — no pip installs needed.

import pandas as pd
import numpy as np
import requests
import json
import os
import time
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

print("✅ Cell 0 complete — all imports available")

"""## 1: API Keys + Constants"""

# === CELL 1: API Keys + Constants ============================================
# Same Kaggle/Colab detection pattern as finetune.py / inference.py.

_on_kaggle = os.path.exists("/kaggle/working")
if _on_kaggle:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    OWM_API_KEY  = secrets.get_secret("OWM_API_KEY")
    NREL_API_KEY = secrets.get_secret("NREL_API_KEY")
    EIA_API_KEY  = secrets.get_secret("EIA_API_KEY")
    print("   Keys loaded from Kaggle Secrets")
else:
    from google.colab import userdata
    OWM_API_KEY  = userdata.get("OWM_API_KEY")
    NREL_API_KEY = userdata.get("NREL_API_KEY")
    EIA_API_KEY  = userdata.get("EIA_API_KEY")
    print("   Keys loaded from Colab Secrets")

# Two community locations for geographic diversity
LOCATIONS = {
    "Ann Arbor, MI": {
        "lat": 42.2808, "lon": -83.7430,
        "capacity_kw": 72, "battery_kwh": 100,
        "tilt": 30, "azimuth": 180, "losses": 14,
        "grid_region": "MISO",
        "timezone": "America/New_York",
    },
    "San Mateo, CA": {
        "lat": 37.5630, "lon": -122.3255,
        "capacity_kw": 48, "battery_kwh": 60,
        "tilt": 25, "azimuth": 180, "losses": 14,
        "grid_region": "CAISO",
        "timezone": "America/Los_Angeles",
    },
}

SYSTEM_EFF = 0.85  # inverter * wiring * soiling * mismatch

# Mount Google Drive for export
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/models/solarhive_datasets"
    os.makedirs(DRIVE_DIR, exist_ok=True)
except ImportError:
    DRIVE_DIR = None

print("✅ Cell 1 complete — API keys loaded, 2 locations configured")

"""## 2: Open-Meteo Historical (free, no key)"""

# === CELL 2: Open-Meteo Historical (free, no key) ============================
# Pull 1 year of hourly data for both locations.
# Variables: GHI, temperature, cloud cover, wind speed, relative humidity.

def fetch_open_meteo(lat, lon, timezone="America/New_York", start_date="2025-04-01", end_date="2026-03-31"):
    """Fetch hourly historical weather from Open-Meteo archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation,temperature_2m,cloudcover,windspeed_10m,relative_humidity_2m",
        "timezone": timezone,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df

_meteo_frames = []
for loc_name, loc in LOCATIONS.items():
    print(f"  Fetching Open-Meteo for {loc_name}...")
    df = fetch_open_meteo(loc["lat"], loc["lon"], timezone=loc["timezone"])
    df["location"] = loc_name
    df["capacity_kw"] = loc["capacity_kw"]
    _meteo_frames.append(df)
    time.sleep(1)  # rate-limit courtesy

df_meteo = pd.concat(_meteo_frames, ignore_index=True)

# Derived columns
df_meteo["temp_f"] = df_meteo["temperature_2m"] * 9 / 5 + 32
df_meteo["wind_mph"] = df_meteo["windspeed_10m"] * 0.621371
df_meteo["temp_derate"] = df_meteo["temp_f"].apply(
    lambda t: max(0.75, 1.0 - 0.004 * max(0, t - 77))
)
df_meteo["ghi"] = df_meteo["shortwave_radiation"].fillna(0).clip(0, 1400)
df_meteo["prod_kw"] = (
    df_meteo["capacity_kw"] * (df_meteo["ghi"] / 1000) * SYSTEM_EFF * df_meteo["temp_derate"]
)
df_meteo["hour"] = df_meteo["time"].dt.hour
df_meteo["month"] = df_meteo["time"].dt.month
df_meteo["season"] = df_meteo["month"].map(
    {12: "winter", 1: "winter", 2: "winter",
     3: "spring", 4: "spring", 5: "spring",
     6: "summer", 7: "summer", 8: "summer",
     9: "fall", 10: "fall", 11: "fall"}
)

# Validate
_nulls = df_meteo[["ghi", "temperature_2m", "cloudcover"]].isnull().sum()
print(f"  Open-Meteo: {len(df_meteo)} rows, nulls: {_nulls.to_dict()}")
print(f"  GHI range: {df_meteo['ghi'].min():.0f}–{df_meteo['ghi'].max():.0f} W/m²")
print(f"  Temp range: {df_meteo['temp_f'].min():.1f}–{df_meteo['temp_f'].max():.1f} °F")
print(f"  Cloud range: {df_meteo['cloudcover'].min():.0f}–{df_meteo['cloudcover'].max():.0f}%")
print("✅ Cell 2 complete — Open-Meteo historical loaded")

"""## 3: NREL PVWatts v8"""

# === CELL 3: NREL PVWatts v8 (uses NREL_API_KEY) =============================
# Monthly + hourly TMY data for both locations.
# Cross-validates against Open-Meteo GHI-derived production.

def fetch_pvwatts(lat, lon, capacity_kw, tilt, azimuth, losses, api_key):
    """Fetch PVWatts v8 monthly + hourly data from NREL."""
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    params = {
        "api_key": api_key,
        "lat": lat,
        "lon": lon,
        "system_capacity": capacity_kw,
        "module_type": 1,  # premium
        "losses": losses,
        "array_type": 1,   # fixed roof mount
        "tilt": tilt,
        "azimuth": azimuth,
        "timeframe": "hourly",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    outputs = data["outputs"]

    # Monthly summary
    df_monthly = pd.DataFrame({
        "month": list(range(1, 13)),
        "ac_monthly_kwh": outputs["ac_monthly"],
        "solrad_monthly": outputs["solrad_monthly"],
        "dc_monthly_kwh": outputs["dc_monthly"],
    })
    df_monthly["ac_annual_kwh"] = outputs["ac_annual"]

    # Hourly TMY
    df_hourly = pd.DataFrame({
        "ac_w": outputs["ac"],
        "dc_w": outputs["dc"],
        "poa_w": outputs["poa"],
        "tamb_c": outputs["tamb"],
        "tcell_c": outputs["tcell"],
    })
    df_hourly["hour_of_year"] = range(len(df_hourly))
    df_hourly["month"] = [(h // 730 % 12) + 1 for h in range(len(df_hourly))]

    return df_monthly, df_hourly

_pvwatts_monthly_frames = []
_pvwatts_hourly_frames = []
for loc_name, loc in LOCATIONS.items():
    print(f"  Fetching PVWatts for {loc_name}...")
    try:
        df_m, df_h = fetch_pvwatts(
            loc["lat"], loc["lon"], loc["capacity_kw"],
            loc["tilt"], loc["azimuth"], loc["losses"], NREL_API_KEY
        )
        df_m["location"] = loc_name
        df_h["location"] = loc_name
        _pvwatts_monthly_frames.append(df_m)
        _pvwatts_hourly_frames.append(df_h)
    except Exception as e:
        print(f"  ⚠️  PVWatts failed for {loc_name}: {e}")
    time.sleep(1)

if _pvwatts_monthly_frames:
    df_pvwatts_monthly = pd.concat(_pvwatts_monthly_frames, ignore_index=True)
    df_pvwatts_hourly = pd.concat(_pvwatts_hourly_frames, ignore_index=True)
    print(f"  PVWatts monthly: {len(df_pvwatts_monthly)} rows")
    print(f"  PVWatts hourly: {len(df_pvwatts_hourly)} rows")
    print(f"  Annual AC by location:")
    for loc_name in LOCATIONS:
        _loc_m = df_pvwatts_monthly[df_pvwatts_monthly["location"] == loc_name]
        if len(_loc_m) > 0:
            print(f"    {loc_name}: {_loc_m['ac_annual_kwh'].iloc[0]:,.0f} kWh/year")
else:
    df_pvwatts_monthly = pd.DataFrame()
    df_pvwatts_hourly = pd.DataFrame()
    print("  ⚠️  PVWatts data unavailable — will use Open-Meteo only for NREL examples")

print("✅ Cell 3 complete — NREL PVWatts loaded")

"""## 4: OWM Current Weather"""

# === CELL 4: OWM Current Weather (uses OWM_API_KEY) ==========================
# Current conditions for both locations.

def fetch_owm(lat, lon, api_key, tz_name="America/New_York"):
    """Fetch current weather from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _tz = ZoneInfo(tz_name)
    return {
        "temp_f": data["main"]["temp"],
        "clouds_pct": data["clouds"]["all"],
        "description": data["weather"][0]["description"],
        "wind_mph": data["wind"]["speed"],
        "humidity_pct": data["main"]["humidity"],
        "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"], tz=_tz).strftime("%H:%M"),
        "sunset": datetime.fromtimestamp(data["sys"]["sunset"], tz=_tz).strftime("%H:%M"),
    }

_owm_rows = []
for loc_name, loc in LOCATIONS.items():
    print(f"  Fetching OWM for {loc_name}...")
    try:
        row = fetch_owm(loc["lat"], loc["lon"], OWM_API_KEY, tz_name=loc["timezone"])
        row["location"] = loc_name
        _owm_rows.append(row)
    except Exception as e:
        print(f"  ⚠️  OWM failed for {loc_name}: {e}")
    time.sleep(1)

df_owm = pd.DataFrame(_owm_rows)
if len(df_owm) > 0:
    print(f"  OWM: {len(df_owm)} rows")
    for _, r in df_owm.iterrows():
        print(f"    {r['location']}: {r['temp_f']:.0f}°F, {r['clouds_pct']}% clouds, {r['description']}")
else:
    print("  ⚠️  OWM data unavailable — weather impact examples will use Open-Meteo only")

print("✅ Cell 4 complete — OWM current weather loaded")

"""## 5: EIA Grid Data"""

# === CELL 5: EIA Grid Data (uses EIA_API_KEY) ================================
# Hourly generation by fuel type for MISO + CAISO (recent 1-week sample).
# Derives renewable percentage and CO2 intensity.

# CO2 emission factors (kg CO2 / MWh) by fuel type
_CO2_FACTORS = {
    "COL": 1000, "NG": 450, "NUC": 0, "PET": 900,
    "SUN": 0, "WND": 0, "WAT": 0, "OTH": 500, "GEO": 0,
}
_RENEWABLE_TYPES = {"SUN", "WND", "WAT", "GEO"}

def fetch_eia_generation(region, api_key, days=7):
    """Fetch hourly generation by fuel type from EIA API v2.

    EIA hourly data typically lags 1-2 days behind real-time.
    We query `days` worth of data ending 1 day ago to avoid the lag window.
    Record count: 7 days × 24 hours × ~8 fuel types = ~1,344 rows (well under 5,000 limit).
    """
    # End 1 day ago to avoid the EIA data lag window
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=days)
    url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region,
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    _total = data.get("response", {}).get("total", 0)
    rows = data.get("response", {}).get("data", [])
    if not rows:
        # Log diagnostic info for debugging
        print(f"    EIA debug: region={region}, start={start.strftime('%Y-%m-%dT%H')}, "
              f"end={end.strftime('%Y-%m-%dT%H')}, total={_total}, "
              f"warnings={data.get('response', {}).get('warnings', [])}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

def compute_grid_mix(df_eia_raw):
    """Pivot EIA generation data to compute renewable % and CO2 intensity."""
    if df_eia_raw.empty:
        return pd.DataFrame()
    df = df_eia_raw.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    # Pivot: period × fueltype → value
    pivot = df.pivot_table(
        index=["period", "respondent"],
        columns="fueltype",
        values="value",
        aggfunc="sum",
    ).fillna(0).reset_index()
    # Compute totals
    fuel_cols = [c for c in pivot.columns if c not in ("period", "respondent")]
    pivot["total_mw"] = pivot[fuel_cols].sum(axis=1)
    # Renewable %
    renew_cols = [c for c in fuel_cols if c in _RENEWABLE_TYPES]
    pivot["renewable_mw"] = pivot[renew_cols].sum(axis=1) if renew_cols else 0
    pivot["renewable_pct"] = (pivot["renewable_mw"] / pivot["total_mw"].replace(0, 1) * 100).round(1).clip(0, 100)
    # CO2 intensity
    pivot["co2_total"] = sum(
        pivot.get(ft, 0) * factor for ft, factor in _CO2_FACTORS.items()
    )
    pivot["co2_intensity"] = (pivot["co2_total"] / pivot["total_mw"].replace(0, 1)).round(1).clip(lower=0)
    return pivot

# Hardcoded fallback if API fails
_FALLBACK_GRID = {
    "MISO": {"renewable_pct": 12.5, "co2_intensity": 520, "top_fuel": "coal + natural gas",
             "coal_pct": 35, "ng_pct": 40, "nuclear_pct": 12, "wind_pct": 10, "solar_pct": 2.5},
    "CAISO": {"renewable_pct": 38.0, "co2_intensity": 280, "top_fuel": "natural gas + solar",
              "coal_pct": 0, "ng_pct": 42, "nuclear_pct": 9, "wind_pct": 7, "solar_pct": 25, "hydro_pct": 12},
}

# EIA uses "CISO" as respondent code for CAISO
_EIA_RESPONDENT = {"MISO": "MISO", "CAISO": "CISO"}

_eia_frames = []
for loc_name, loc in LOCATIONS.items():
    region = loc["grid_region"]
    eia_code = _EIA_RESPONDENT.get(region, region)
    print(f"  Fetching EIA for {eia_code} ({loc_name}, grid_region={region})...")
    try:
        df_raw = fetch_eia_generation(eia_code, EIA_API_KEY)
        if not df_raw.empty:
            df_raw["location"] = loc_name
            _eia_frames.append(df_raw)
            print(f"    {len(df_raw)} raw records")
        else:
            print(f"    ⚠️  No data returned — will use fallback")
    except Exception as e:
        print(f"    ⚠️  EIA failed for {eia_code}: {e} — will use fallback")
    time.sleep(1)

if _eia_frames:
    df_eia_gen = pd.concat(_eia_frames, ignore_index=True)
    df_eia_mix = compute_grid_mix(df_eia_gen)
    _eia_available = True
    print(f"  EIA generation: {len(df_eia_gen)} raw rows")
    print(f"  EIA grid mix: {len(df_eia_mix)} hourly snapshots")
    # Summarize by region
    for region in df_eia_mix["respondent"].unique():
        _rmix = df_eia_mix[df_eia_mix["respondent"] == region]
        print(f"    {region}: avg renewable {_rmix['renewable_pct'].mean():.1f}%, "
              f"CO2 {_rmix['co2_intensity'].mean():.0f} kg/MWh")
else:
    df_eia_gen = pd.DataFrame()
    df_eia_mix = pd.DataFrame()
    _eia_available = False
    print("  ⚠️  EIA data unavailable — using hardcoded fallback grid mix")

def _get_grid_stats(region):
    """Return (renewable_pct, co2_intensity) from live EIA or fallback."""
    eia_code = _EIA_RESPONDENT.get(region, region)
    if _eia_available and not df_eia_mix.empty:
        _rmix = df_eia_mix[df_eia_mix["respondent"] == eia_code]
        if len(_rmix) > 0:
            return _rmix["renewable_pct"].mean(), _rmix["co2_intensity"].mean()
    fb = _FALLBACK_GRID[region]
    return fb["renewable_pct"], fb["co2_intensity"]

print("✅ Cell 5 complete — EIA grid data loaded")

"""## 6: DataFrame EDA, Validation & Cross-Validation"""

# === CELL 6: DataFrame EDA, Validation & Cross-Validation ====================
# Full exploratory data analysis: schema, head(10), describe(), distributions,
# range checks, temporal consistency, cross-validation.

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ── 1. OPEN-METEO (df_meteo) ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1. OPEN-METEO HISTORICAL — df_meteo")
print("─" * 70)
print(f"\nShape: {df_meteo.shape[0]:,} rows × {df_meteo.shape[1]} columns")
print(f"\nSchema (dtypes):")
for col in df_meteo.columns:
    _null = df_meteo[col].isnull().sum()
    _null_str = f"  ({_null} nulls)" if _null > 0 else ""
    print(f"  {col:<25s} {str(df_meteo[col].dtype):<15s}{_null_str}")

print(f"\nFirst 10 rows:")
print(df_meteo.head(10).to_string(index=False))

print(f"\nStatistical summary (numeric columns):")
_meteo_desc = df_meteo[["ghi", "temperature_2m", "cloudcover", "windspeed_10m",
                         "relative_humidity_2m", "temp_f", "wind_mph",
                         "temp_derate", "prod_kw"]].describe()
print(_meteo_desc.round(2).to_string())

print(f"\nDistribution by location:")
print(df_meteo["location"].value_counts().to_string())

print(f"\nDistribution by season:")
print(df_meteo.groupby(["location", "season"]).size().unstack(fill_value=0).to_string())

print(f"\nHourly production profile (avg kW by hour, per location):")
_hourly_profile = df_meteo.groupby(["location", "hour"])["prod_kw"].mean().unstack(level=0)
print(_hourly_profile.round(1).to_string())

print(f"\nGHI percentiles by location:")
for loc_name in LOCATIONS:
    _loc = df_meteo[df_meteo["location"] == loc_name]
    _pcts = _loc["ghi"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"  {loc_name}: " + ", ".join(f"p{int(p*100)}={v:.0f}" for p, v in _pcts.items()))

# Correlation matrix for key numeric columns
print(f"\nCorrelation matrix (key columns):")
_corr_cols = ["ghi", "cloudcover", "temperature_2m", "relative_humidity_2m", "windspeed_10m", "prod_kw"]
print(df_meteo[_corr_cols].corr().round(3).to_string())

# Range checks
_ghi_oob = (df_meteo["ghi"] > 1400).sum()
_cloud_oob = ((df_meteo["cloudcover"] < 0) | (df_meteo["cloudcover"] > 100)).sum()
_night_ghi = df_meteo[(df_meteo["hour"] < 5) | (df_meteo["hour"] > 21)]
_night_bright = (_night_ghi["ghi"] > 50).sum()
print(f"\nRange checks:")
print(f"  GHI out-of-range (>1400):     {_ghi_oob}")
print(f"  Cloud out-of-range:            {_cloud_oob}")
print(f"  Nighttime GHI>50 (anomalies): {_night_bright}")

# Clip outliers
df_meteo["cloudcover"] = df_meteo["cloudcover"].clip(0, 100)

# Seasonal GHI patterns
print(f"\nSeasonal average GHI (W/m²) by location:")
_seasonal = df_meteo.groupby(["location", "season"])["ghi"].mean().unstack()
for loc_name in LOCATIONS:
    if loc_name in _seasonal.index:
        row = _seasonal.loc[loc_name]
        print(f"  {loc_name}: " + ", ".join(f"{s}={row.get(s, 0):.0f}" for s in ["spring", "summer", "fall", "winter"]))

# ── 2. NREL PVWATTS (df_pvwatts_monthly, df_pvwatts_hourly) ──────────────────
print("\n" + "─" * 70)
print("2. NREL PVWATTS v8 — df_pvwatts_monthly / df_pvwatts_hourly")
print("─" * 70)

if not df_pvwatts_monthly.empty:
    print(f"\nMonthly — Shape: {df_pvwatts_monthly.shape[0]} rows × {df_pvwatts_monthly.shape[1]} columns")
    print(f"\nSchema (dtypes):")
    for col in df_pvwatts_monthly.columns:
        print(f"  {col:<20s} {str(df_pvwatts_monthly[col].dtype)}")
    print(f"\nAll monthly rows:")
    print(df_pvwatts_monthly.to_string(index=False))

    print(f"\nHourly TMY — Shape: {df_pvwatts_hourly.shape[0]:,} rows × {df_pvwatts_hourly.shape[1]} columns")
    print(f"\nFirst 10 rows:")
    print(df_pvwatts_hourly.head(10).to_string(index=False))
    print(f"\nStatistical summary:")
    print(df_pvwatts_hourly[["ac_w", "dc_w", "poa_w", "tamb_c", "tcell_c"]].describe().round(2).to_string())

    # Cross-validate Open-Meteo vs PVWatts
    print(f"\nCross-validation: Open-Meteo production vs PVWatts monthly AC (kWh):")
    for loc_name, loc in LOCATIONS.items():
        _loc_meteo = df_meteo[df_meteo["location"] == loc_name]
        _meteo_monthly = _loc_meteo.groupby("month")["prod_kw"].sum()
        _loc_pvw = df_pvwatts_monthly[df_pvwatts_monthly["location"] == loc_name]
        if len(_loc_pvw) == 0:
            continue
        print(f"  {loc_name}:")
        _diffs = []
        for m in range(1, 13):
            _om = _meteo_monthly.get(m, 0)
            _pv = _loc_pvw[_loc_pvw["month"] == m]["ac_monthly_kwh"].values
            _pv_val = _pv[0] if len(_pv) > 0 else 0
            _pct_diff = abs(_om - _pv_val) / max(_pv_val, 1) * 100
            _diffs.append(_pct_diff)
            _flag = " ⚠️" if _pct_diff > 30 else ""
            print(f"    Month {m:2d}: OM={_om:7,.0f}  PV={_pv_val:7,.0f}  diff={_pct_diff:5.1f}%{_flag}")
        print(f"    Avg absolute diff: {np.mean(_diffs):.1f}%")
else:
    print("\n  PVWatts data unavailable — skipping")

# ── 3. OWM CURRENT WEATHER (df_owm) ──────────────────────────────────────────
print("\n" + "─" * 70)
print("3. OWM CURRENT WEATHER — df_owm")
print("─" * 70)

if len(df_owm) > 0:
    print(f"\nShape: {df_owm.shape[0]} rows × {df_owm.shape[1]} columns")
    print(f"\nSchema (dtypes):")
    for col in df_owm.columns:
        print(f"  {col:<20s} {str(df_owm[col].dtype)}")
    print(f"\nAll rows (current conditions):")
    print(df_owm.to_string(index=False))
else:
    print("\n  OWM data unavailable — skipping")

# ── 4. EIA GRID DATA (df_eia_gen, df_eia_mix) ────────────────────────────────
print("\n" + "─" * 70)
print("4. EIA GRID DATA — df_eia_gen / df_eia_mix")
print("─" * 70)

if _eia_available and not df_eia_gen.empty:
    print(f"\nRaw generation — Shape: {df_eia_gen.shape[0]:,} rows × {df_eia_gen.shape[1]} columns")
    print(f"\nSchema (dtypes):")
    for col in df_eia_gen.columns:
        print(f"  {col:<20s} {str(df_eia_gen[col].dtype)}")
    print(f"\nFirst 10 rows:")
    print(df_eia_gen.head(10).to_string(index=False))

    print(f"\nFuel types present: {sorted(df_eia_gen['fueltype'].unique().tolist()) if 'fueltype' in df_eia_gen.columns else 'N/A'}")
    print(f"Respondents: {sorted(df_eia_gen['respondent'].unique().tolist()) if 'respondent' in df_eia_gen.columns else 'N/A'}")

    if not df_eia_mix.empty:
        print(f"\nGrid mix (pivoted) — Shape: {df_eia_mix.shape[0]:,} rows × {df_eia_mix.shape[1]} columns")
        print(f"\nFirst 10 rows:")
        print(df_eia_mix.head(10).to_string(index=False))
        print(f"\nGrid mix summary by region:")
        for region in df_eia_mix["respondent"].unique():
            _rmix = df_eia_mix[df_eia_mix["respondent"] == region]
            print(f"  {region}:")
            print(f"    Renewable %: min={_rmix['renewable_pct'].min():.1f}, "
                  f"avg={_rmix['renewable_pct'].mean():.1f}, max={_rmix['renewable_pct'].max():.1f}")
            print(f"    CO2 (kg/MWh): min={_rmix['co2_intensity'].min():.0f}, "
                  f"avg={_rmix['co2_intensity'].mean():.0f}, max={_rmix['co2_intensity'].max():.0f}")
            print(f"    Total MW: min={_rmix['total_mw'].min():.0f}, "
                  f"avg={_rmix['total_mw'].mean():.0f}, max={_rmix['total_mw'].max():.0f}")
else:
    print(f"\n  EIA data unavailable — using hardcoded fallback:")
    for region, fb in _FALLBACK_GRID.items():
        print(f"  {region}: renewable {fb['renewable_pct']}%, CO2 {fb['co2_intensity']} kg/MWh, "
              f"top fuel: {fb['top_fuel']}")

# ── 5. SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5. DATASET SUMMARY")
print("─" * 70)
print(f"\n{'DataFrame':<25s} {'Rows':>10s}  {'Cols':>5s}  {'Source':<30s}")
print(f"{'─'*25} {'─'*10}  {'─'*5}  {'─'*30}")
print(f"{'df_meteo':<25s} {len(df_meteo):>10,}  {df_meteo.shape[1]:>5}  {'Open-Meteo hourly (1 year)'}")
print(f"{'df_pvwatts_monthly':<25s} {len(df_pvwatts_monthly):>10,}  {df_pvwatts_monthly.shape[1] if not df_pvwatts_monthly.empty else 0:>5}  {'NREL PVWatts monthly'}")
print(f"{'df_pvwatts_hourly':<25s} {len(df_pvwatts_hourly):>10,}  {df_pvwatts_hourly.shape[1] if not df_pvwatts_hourly.empty else 0:>5}  {'NREL PVWatts hourly TMY'}")
print(f"{'df_owm':<25s} {len(df_owm):>10,}  {df_owm.shape[1] if len(df_owm) > 0 else 0:>5}  {'OWM current (2 locations)'}")
if _eia_available:
    print(f"{'df_eia_gen':<25s} {len(df_eia_gen):>10,}  {df_eia_gen.shape[1]:>5}  {'EIA hourly generation'}")
    print(f"{'df_eia_mix':<25s} {len(df_eia_mix):>10,}  {df_eia_mix.shape[1]:>5}  {'EIA grid mix (pivoted)'}")
else:
    print(f"{'df_eia (fallback)':<25s} {'N/A':>10s}  {'N/A':>5s}  {'Hardcoded grid mix'}")

# ── 6. DATA FORMAT DOCUMENTATION ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("6. OUTPUT FORMAT (for solarhive_finetune.py)")
print("─" * 70)
print("""
Generated examples are 3-tuples: (system_prompt, user_question, assistant_answer)

  system_prompt:  "You are SolarHive, an AI energy advisor for a community solar microgrid."
  user_question:  Natural language question with data-grounded specifics
  assistant_answer: Detailed response referencing actual API values (50-200 words)

Export JSON schema:
  {
    "metadata": {
      "generator": "solarhive_datagen.py",
      "generated_at": "ISO timestamp",
      "total_examples": int,
      "estimated_tokens": int,
      "categories": { "A_hourly_production": int, ... },
      "locations": ["Ann Arbor, MI", "San Mateo, CA"],
      "data_sources": ["Open-Meteo", "NREL PVWatts v8", "OpenWeatherMap", "EIA v2"]
    },
    "qa_data": [ ["system", "question", "answer"], ... ]
  }

Consumed by solarhive_finetune.py Cell 4b:
  DATA.extend([tuple(ex) for ex in dg["qa_data"]])
""")

print("✅ Cell 6 complete — EDA and validation passed")

"""## 6b: Visualizations"""

# === CELL 6b: Visualizations =================================================
# Full matplotlib/seaborn chart suite — pre-installed on Colab.
# 12 charts covering all DataFrames: distributions, time series, correlations.

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
_FIG_W, _FIG_H = 14, 5  # default figure size

# ── Chart 1: GHI Distribution by Location (histogram + KDE) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))
for ax, loc_name in zip(axes, LOCATIONS.keys()):
    _loc = df_meteo[(df_meteo["location"] == loc_name) & (df_meteo["ghi"] > 0)]
    sns.histplot(_loc["ghi"], bins=50, kde=True, ax=ax, color="goldenrod")
    ax.set_title(f"GHI Distribution — {loc_name}")
    ax.set_xlabel("GHI (W/m²)")
    ax.set_ylabel("Count")
    ax.axvline(_loc["ghi"].median(), color="red", ls="--", label=f"Median: {_loc['ghi'].median():.0f}")
    ax.legend()
plt.tight_layout()
plt.savefig("chart_01_ghi_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 1: GHI distribution by location")

# ── Chart 2: Hourly Production Profile (mean ± std by hour) ──────────────────
fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for loc_name in LOCATIONS:
    _loc = df_meteo[df_meteo["location"] == loc_name]
    _hourly = _loc.groupby("hour")["prod_kw"].agg(["mean", "std"])
    ax.plot(_hourly.index, _hourly["mean"], marker="o", label=loc_name, linewidth=2)
    ax.fill_between(_hourly.index, _hourly["mean"] - _hourly["std"],
                     _hourly["mean"] + _hourly["std"], alpha=0.2)
ax.set_title("Average Hourly Solar Production (mean ± 1 std)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Production (kW)")
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig("chart_02_hourly_production.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 2: Hourly production profile")

# ── Chart 3: Seasonal Heatmap (month × hour, avg production) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, 6))
for ax, loc_name in zip(axes, LOCATIONS.keys()):
    _loc = df_meteo[df_meteo["location"] == loc_name]
    _pivot = _loc.pivot_table(index="month", columns="hour", values="prod_kw", aggfunc="mean")
    sns.heatmap(_pivot, ax=ax, cmap="YlOrRd", cbar_kws={"label": "kW"})
    ax.set_title(f"Avg Production — {loc_name}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig("chart_03_seasonal_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 3: Seasonal production heatmap")

# ── Chart 4: Temperature Derating Curve ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, _FIG_H))
_temps = np.linspace(50, 120, 100)
_derates = [max(0.75, 1.0 - 0.004 * max(0, t - 77)) for t in _temps]
ax.plot(_temps, _derates, color="crimson", linewidth=2.5)
ax.axvline(77, color="gray", ls=":", label="Reference temp (77°F)")
ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
# Scatter actual data (sample)
_sample = df_meteo[df_meteo["ghi"] > 100].sample(n=min(500, len(df_meteo)), random_state=42)
ax.scatter(_sample["temp_f"], _sample["temp_derate"], alpha=0.3, s=10, color="steelblue", label="Actual data")
ax.set_title("Temperature Derating Factor vs Temperature")
ax.set_xlabel("Temperature (°F)")
ax.set_ylabel("Derating Factor")
ax.legend()
plt.tight_layout()
plt.savefig("chart_04_temp_derating.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 4: Temperature derating curve")

# ── Chart 5: Correlation Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
_corr_cols = ["ghi", "cloudcover", "temperature_2m", "relative_humidity_2m", "windspeed_10m", "prod_kw", "temp_derate"]
_corr = df_meteo[_corr_cols].corr()
mask = np.triu(np.ones_like(_corr, dtype=bool))
sns.heatmap(_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("chart_05_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 5: Correlation heatmap")

# ── Chart 6: Cloud Cover Distribution by Season ─────────────────────────────
fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
_season_order = ["spring", "summer", "fall", "winter"]
sns.boxplot(data=df_meteo, x="season", y="cloudcover", hue="location",
            order=_season_order, ax=ax)
ax.set_title("Cloud Cover Distribution by Season and Location")
ax.set_xlabel("Season")
ax.set_ylabel("Cloud Cover (%)")
plt.tight_layout()
plt.savefig("chart_06_cloud_by_season.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 6: Cloud cover by season")

# ── Chart 7: Production Box Plots by Season ──────────────────────────────────
fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
_daytime = df_meteo[(df_meteo["hour"] >= 7) & (df_meteo["hour"] <= 18) & (df_meteo["ghi"] > 0)]
sns.boxplot(data=_daytime, x="season", y="prod_kw", hue="location",
            order=_season_order, ax=ax)
ax.set_title("Daytime (7am-6pm) Solar Production by Season")
ax.set_xlabel("Season")
ax.set_ylabel("Production (kW)")
plt.tight_layout()
plt.savefig("chart_07_production_by_season.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 7: Production by season")

# ── Chart 8: GHI vs Production Scatter (colored by cloud cover) ──────────────
fig, ax = plt.subplots(figsize=(10, 6))
_day_sample = df_meteo[df_meteo["ghi"] > 0].sample(n=min(2000, len(df_meteo)), random_state=42)
scatter = ax.scatter(_day_sample["ghi"], _day_sample["prod_kw"],
                     c=_day_sample["cloudcover"], cmap="coolwarm_r", alpha=0.5, s=15)
plt.colorbar(scatter, label="Cloud Cover (%)")
ax.set_title("GHI vs Solar Production (colored by cloud cover)")
ax.set_xlabel("GHI (W/m²)")
ax.set_ylabel("Production (kW)")
plt.tight_layout()
plt.savefig("chart_08_ghi_vs_production.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 8: GHI vs production scatter")

# ── Chart 9: Monthly Production Comparison (Open-Meteo vs PVWatts) ───────────
if not df_pvwatts_monthly.empty:
    fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))
    for ax, loc_name in zip(axes, LOCATIONS.keys()):
        _loc_meteo = df_meteo[df_meteo["location"] == loc_name]
        _om_monthly = _loc_meteo.groupby("month")["prod_kw"].sum()
        _loc_pvw = df_pvwatts_monthly[df_pvwatts_monthly["location"] == loc_name]
        months = range(1, 13)
        _om_vals = [_om_monthly.get(m, 0) for m in months]
        _pv_vals = [_loc_pvw[_loc_pvw["month"] == m]["ac_monthly_kwh"].values[0]
                    if len(_loc_pvw[_loc_pvw["month"] == m]) > 0 else 0 for m in months]
        x = np.arange(12)
        ax.bar(x - 0.2, _om_vals, 0.4, label="Open-Meteo", color="steelblue")
        ax.bar(x + 0.2, _pv_vals, 0.4, label="PVWatts", color="coral")
        ax.set_title(f"Monthly Production — {loc_name}")
        ax.set_xlabel("Month")
        ax.set_ylabel("kWh")
        ax.set_xticks(x)
        ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
        ax.legend()
    plt.tight_layout()
    plt.savefig("chart_09_monthly_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Chart 9: Monthly Open-Meteo vs PVWatts comparison")

# ── Chart 10: OWM Current Conditions Snapshot ─────────────────────────────────
if len(df_owm) > 0:
    fig, axes = plt.subplots(1, 4, figsize=(_FIG_W, 4))
    _metrics = [("temp_f", "Temperature (°F)", "tomato"),
                ("clouds_pct", "Clouds (%)", "steelblue"),
                ("wind_mph", "Wind (mph)", "seagreen"),
                ("humidity_pct", "Humidity (%)", "mediumpurple")]
    for ax, (col, title, color) in zip(axes, _metrics):
        ax.bar(df_owm["location"], df_owm[col], color=color, width=0.5)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(df_owm[col]):
            ax.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=10)
    plt.suptitle("OWM Current Conditions", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("chart_10_owm_current.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Chart 10: OWM current conditions")

# ── Chart 11: EIA Grid Mix (stacked bar by region) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))
_fuel_colors = {"COL": "#4a4a4a", "NG": "#ff8c00", "NUC": "#9b59b6", "PET": "#8b4513",
                "SUN": "#f1c40f", "WND": "#3498db", "WAT": "#1abc9c", "OTH": "#95a5a6", "GEO": "#e74c3c"}
def _render_fallback_grid(ax, region):
    """Render fallback grid mix chart for a region."""
    fb = _FALLBACK_GRID[region]
    _fuels = {k: v for k, v in fb.items() if k.endswith("_pct")}
    labels = [k.replace("_pct", "").upper() for k in _fuels]
    values = list(_fuels.values())
    colors = [_fuel_colors.get(l[:3], "#cccccc") for l in labels]
    ax.barh(labels, values, color=colors)
    ax.set_title(f"{region} — Grid Mix (%) [Fallback Data]")
    ax.set_xlabel("Percentage")
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=9)

for ax, region in zip(axes, ["MISO", "CAISO"]):
    if _eia_available and not df_eia_mix.empty:
        _rmix = df_eia_mix[df_eia_mix["respondent"] == region]
        if len(_rmix) == 0:
            # Try alternate respondent code (CISO for CAISO)
            _alt = {"CAISO": "CISO", "CISO": "CAISO"}
            _rmix = df_eia_mix[df_eia_mix["respondent"] == _alt.get(region, "")]
        if len(_rmix) > 0:
            fuel_cols = [c for c in _rmix.columns if c in _CO2_FACTORS]
            _avg = _rmix[fuel_cols].mean()
            _total = _avg.sum()
            _pcts = (_avg / _total * 100).sort_values(ascending=True)
            colors = [_fuel_colors.get(f, "#cccccc") for f in _pcts.index]
            ax.barh(_pcts.index, _pcts.values, color=colors)
            ax.set_title(f"{region} — Average Fuel Mix (%)")
            ax.set_xlabel("Percentage of Generation")
            for i, (fuel, pct) in enumerate(_pcts.items()):
                if pct > 2:
                    ax.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=9)
        else:
            _render_fallback_grid(ax, region)
    else:
        _render_fallback_grid(ax, region)
plt.tight_layout()
plt.savefig("chart_11_grid_mix.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart 11: EIA grid mix by region")

# ── Chart 12: CO2 Intensity & Renewable % Time Series ────────────────────────
if _eia_available and not df_eia_mix.empty and "period" in df_eia_mix.columns:
    fig, axes = plt.subplots(2, 1, figsize=(_FIG_W, 8), sharex=True)
    for region in df_eia_mix["respondent"].unique():
        _rmix = df_eia_mix[df_eia_mix["respondent"] == region].copy()
        _rmix["period_dt"] = pd.to_datetime(_rmix["period"], errors="coerce")
        _rmix = _rmix.dropna(subset=["period_dt"]).sort_values("period_dt")
        if len(_rmix) == 0:
            continue
        axes[0].plot(_rmix["period_dt"], _rmix["renewable_pct"], label=region, linewidth=1.5)
        axes[1].plot(_rmix["period_dt"], _rmix["co2_intensity"], label=region, linewidth=1.5)
    axes[0].set_title("Renewable Energy Percentage Over Time")
    axes[0].set_ylabel("Renewable %")
    axes[0].legend()
    axes[1].set_title("CO2 Intensity Over Time")
    axes[1].set_ylabel("kg CO2 / MWh")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("chart_12_co2_renewable_ts.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Chart 12: CO2 intensity & renewable % time series")

print("\n✅ Cell 6b complete — all visualizations rendered")
print("   Charts saved as chart_01–12_*.png in working directory")

"""## 7: Synthetic Example Generation"""

# === CELL 7: Synthetic Example Generation (~520 examples) ====================
# Stratified sampling from DataFrames → data-grounded Q&A pairs.
# random.seed(42) for reproducibility.

random.seed(42)
np.random.seed(42)

# System prompts repeated twice — see Leviathan et al. (2024), "Repeat to Improve
# Non-Reasoning LLMs", Google Research. https://arxiv.org/abs/2512.14982
SYS = (
    "You are SolarHive, an AI energy advisor for a community solar microgrid.\n\n"
    "You are SolarHive, an AI energy advisor for a community solar microgrid."
)

# Rich system prompt for tool-calling examples (matches inference SYSTEM_PROMPT)
SYS_TOOLS = (
    "You are SolarHive, an AI energy advisor for a community of 12 homes "
    "with rooftop solar and shared battery storage in Ann Arbor, Michigan. "
    "Use the available tools to get real-time data before answering. "
    "Be specific, reference actual data, and keep responses concise (3-5 sentences).\n\n"
    "You are SolarHive, an AI energy advisor for a community of 12 homes "
    "with rooftop solar and shared battery storage in Ann Arbor, Michigan. "
    "Use the available tools to get real-time data before answering. "
    "Be specific, reference actual data, and keep responses concise (3-5 sentences)."
)

# Helper: grid period/rate from hour
def _grid_period(hour):
    if 14 <= hour < 19:
        return "peak", 0.28
    elif (7 <= hour < 14) or (19 <= hour < 23):
        return "mid-peak", 0.18
    else:
        return "off-peak", 0.10

# Helper: sample rows by condition
def _sample_meteo(condition_mask, n, replace=False):
    pool = df_meteo[condition_mask]
    if len(pool) == 0:
        return pd.DataFrame()
    return pool.sample(n=min(n, len(pool)), replace=replace, random_state=42)

# Question phrasing variants (15+ per category to avoid duplication)
_STATUS_PHRASES = [
    "What's our solar production status right now?",
    "How much power are we generating currently?",
    "Give me a quick solar status update.",
    "What's the current output from our panels?",
    "How are the solar panels performing at the moment?",
    "Can you check our current solar generation?",
    "Pull up the real-time solar dashboard.",
    "What does our generation look like right now?",
    "Are the panels producing at capacity?",
    "Quick check — how's our solar array doing?",
    "What's the live production reading?",
    "How many kilowatts are we pushing right now?",
    "Solar status report, please.",
    "Is the array running at full power?",
    "What's our current kilowatt output?",
    "Show me today's solar performance so far.",
]

_APPLIANCE_PHRASES = [
    "Should I run the dishwasher now or wait?",
    "Is it a good time to do laundry?",
    "Can I charge my EV now without wasting solar?",
    "Should I hold off on running the dryer?",
    "Is now a good time to run high-power appliances?",
    "When should I schedule my pool pump today?",
    "I want to run the washing machine — is solar available?",
    "Can I bake without pulling from the grid?",
    "Is there enough solar to run the AC at full blast?",
    "Good time to charge the electric mower?",
    "Should I start the hot water heater now?",
    "Can the community handle my EV charger right now?",
    "When's the cheapest time to run appliances today?",
    "I need to vacuum — should I wait for more sun?",
    "Is the oven okay to use or will we draw from grid?",
    "Best window today for high-power appliance use?",
]

_COMPARE_PHRASES = [
    "How does today compare to yesterday's solar output?",
    "Are we doing better or worse than usual for this time of year?",
    "How does this month's production stack up against last month?",
    "Is our solar generation above or below the seasonal average?",
    "Compare current output to our historical average.",
    "What's our production trend this week?",
    "Are we tracking above or below our monthly target?",
    "How does today's GHI compare to the seasonal norm?",
    "Is this a high-production or low-production day?",
    "Where do we stand relative to last year's numbers?",
    "Are conditions better or worse than the 30-day average?",
    "Rate today's solar performance on a scale.",
    "How are we trending compared to typical spring/summer/fall/winter output?",
    "Is this week above or below our capacity factor benchmark?",
    "How does current efficiency compare to our system's rated performance?",
]

_BATTERY_PHRASES = [
    "What's the best battery strategy right now?",
    "Should we charge or discharge the battery?",
    "How should we manage the battery given current conditions?",
    "What's the optimal battery action for this situation?",
    "Give me a battery management recommendation.",
    "Battery decision time — what do you suggest?",
    "Should the battery be absorbing or releasing energy?",
    "What's the smart move for our battery right now?",
    "How full should we keep the battery at this hour?",
    "Is it worth charging the battery from the grid right now?",
    "Battery strategy check — hold, charge, or export?",
    "What SOC target should we aim for by tonight?",
    "Should we reserve battery for the evening peak?",
    "Is the battery better used now or saved for later?",
    "What's the best use of our stored energy right now?",
]

_GRID_PHRASES = [
    "What's the current grid situation?",
    "Should we export to the grid or store energy?",
    "How does the grid rate affect our strategy right now?",
    "What's the economic case for grid export vs battery storage?",
    "Are we in peak, mid-peak, or off-peak pricing?",
    "What's our grid import/export balance?",
    "Is the grid rate high enough to justify selling power?",
    "How dirty is the grid right now — should we avoid importing?",
    "What rate are we paying for grid electricity this hour?",
    "Should we island from the grid or stay connected?",
    "Grid economics update — are we buying or selling?",
    "Is it cheaper to use battery or grid power right now?",
    "What's the current net metering value?",
    "How much are we saving by not pulling from the grid?",
    "Grid rate forecast — what should we prepare for?",
]

EXAMPLES = []

# ---------- Category A: Hourly Production (~200 examples) --------------------
print("  Generating Category A: Hourly Production...")

# Condition-specific question phrases to boost diversity
_CLEAR_SKY_PHRASES = [
    "How are the panels doing in this beautiful weather?",
    "Looks sunny out — what's our solar output?",
    "Clear skies! Are we maxing out production?",
    "Perfect solar day — what are the numbers?",
    "Sun's out in full force. What's the dashboard say?",
    "No clouds — how much are we generating?",
    "Blue sky conditions — give me the production stats.",
    "It's a gorgeous day. How's solar performing?",
    "Sunshine report — are we at peak output?",
    "How efficiently are the panels converting in this sun?",
    "Maximum solar conditions — what's our actual kW?",
    "We should be crushing it today. What's the output?",
    "Any reason we're not at full capacity on a clear day?",
    "How close to rated power are we right now?",
    "Sunny and warm — is that helping or hurting output?",
    "Is this a good day to charge up the battery?",
    "Clear sky performance check — how's the array?",
    "Give me the clear-sky production breakdown.",
    "What efficiency are we hitting in these conditions?",
    "Ideal weather — what's our generation ceiling today?",
]

_CLOUDY_PHRASES = [
    "It's overcast — how badly is production hit?",
    "Clouds moved in. What's the damage to our solar?",
    "Not much sun today — what are we still generating?",
    "How much are we losing to cloud cover?",
    "Gloomy day. Should we even bother with solar loads?",
    "Clouds are thick — is the battery keeping up?",
    "Overcast conditions — should I defer heavy appliances?",
    "Cloud cover report — how's our output impacted?",
    "What happens to production with this cloud layer?",
    "Gray skies — give me the solar reality check.",
    "Can we still run appliances on a cloudy day like this?",
    "How much diffuse radiation are we capturing in these clouds?",
    "Are the panels still worth anything under this overcast?",
    "Cloud impact assessment — what's the production cut?",
    "Partly cloudy turning overcast — what should I expect?",
    "Solar viability check for these cloudy conditions?",
    "Is it better to use grid or battery right now?",
    "Heavy clouds — should we switch to grid power?",
    "What percentage of capacity are we hitting under these clouds?",
    "Cloudy day strategy — where do we stand?",
]

_NIGHT_PHRASES = [
    "It's dark out. What's happening with our power?",
    "No sun — where's our electricity coming from?",
    "Nighttime power status?",
    "After-hours energy report — what are we drawing from?",
    "Sun's down. What's our grid and battery situation?",
    "Late night — how's the energy balance?",
    "Midnight check — are we on battery or grid?",
    "No solar at this hour. What's the cost of grid power?",
    "How much battery do we have left tonight?",
    "Pre-dawn energy status update?",
]

# A1: Clear sky status reports (~40)
_clear = _sample_meteo((df_meteo["ghi"] > 600) & (df_meteo["cloudcover"] < 20), 30)
_a1_templates = [
    lambda r, eff: (f"Clear skies in {r['location']} — panels are performing well. Current GHI is {r['ghi']:.0f} W/m² "
         f"with {r['cloudcover']:.0f}% cloud cover. At {r['temp_f']:.0f}°F (temperature derating: "
         f"{r['temp_derate']:.0%}), we're producing {r['prod_kw']:.1f} kW out of {r['capacity_kw']} kW capacity ({eff:.0f}% efficiency). "
         f"This is excellent output — a great time to run heavy loads or charge the battery."),
    lambda r, eff: (f"Solar is strong in {r['location']}. GHI reading: {r['ghi']:.0f} W/m², clouds at just "
         f"{r['cloudcover']:.0f}%. We're generating {r['prod_kw']:.1f} kW ({eff:.0f}% of {r['capacity_kw']} kW capacity). "
         f"Temperature is {r['temp_f']:.0f}°F with a derating factor of {r['temp_derate']:.3f}. "
         f"Ideal conditions — consider charging the battery or running energy-intensive appliances."),
    lambda r, eff: (f"Excellent solar conditions in {r['location']}! With {r['ghi']:.0f} W/m² irradiance and minimal "
         f"cloud cover ({r['cloudcover']:.0f}%), the array is pushing {r['prod_kw']:.1f} kW — that's "
         f"{eff:.0f}% efficiency. At {r['temp_f']:.0f}°F, thermal derating is {(1-r['temp_derate'])*100:.1f}%. "
         f"Take advantage of this window for heavy loads."),
    lambda r, eff: (f"Status for {r['location']}: {r['prod_kw']:.1f} kW output under clear skies ({r['cloudcover']:.0f}% clouds). "
         f"GHI is {r['ghi']:.0f} W/m² and we're hitting {eff:.0f}% of our {r['capacity_kw']} kW rated capacity. "
         f"Panel temp effect: {r['temp_derate']:.0%} efficiency at {r['temp_f']:.0f}°F. Battery charging and "
         f"appliance use are both recommended now."),
    lambda r, eff: (f"Peak performance in {r['location']}: {r['prod_kw']:.1f} kW from {r['capacity_kw']} kW array ({eff:.0f}% efficiency). "
         f"GHI at {r['ghi']:.0f} W/m², cloud cover just {r['cloudcover']:.0f}%. "
         f"At {r['temp_f']:.0f}°F, thermal factor is {r['temp_derate']:.3f}. "
         f"This is one of our best production windows — queue up laundry, EV charging, and battery top-up."),
    lambda r, eff: (f"Array report for {r['location']}: producing {r['prod_kw']:.1f}/{r['capacity_kw']} kW under clear conditions. "
         f"Irradiance: {r['ghi']:.0f} W/m². Clouds: {r['cloudcover']:.0f}%. Temp: {r['temp_f']:.0f}°F (derating: {(1-r['temp_derate'])*100:.1f}%). "
         f"Efficiency: {eff:.0f}%. Strong output — maximize self-consumption and charge the community battery."),
]
for i, (_, row) in enumerate(_clear.iterrows()):
    q = random.choice(_CLEAR_SKY_PHRASES) if random.random() < 0.6 else random.choice(_STATUS_PHRASES)
    eff = row["prod_kw"] / row["capacity_kw"] * 100 if row["capacity_kw"] > 0 else 0
    a = _a1_templates[i % len(_a1_templates)](row, eff)
    EXAMPLES.append((SYS, q, a))

# A2: Cloudy day status (~40)
_cloudy = _sample_meteo((df_meteo["cloudcover"] > 70) & (df_meteo["ghi"] > 50), 30)
_a2_templates = [
    lambda r: (f"Heavy cloud cover in {r['location']} at {r['cloudcover']:.0f}%. GHI is only {r['ghi']:.0f} W/m², "
         f"so production is {r['prod_kw']:.1f} kW out of {r['capacity_kw']} kW capacity. "
         f"Consider deferring large loads until conditions improve. "
         f"The battery should be conserved for essential evening needs."),
    lambda r: (f"Overcast in {r['location']} — {r['cloudcover']:.0f}% cloud cover is limiting our output. "
         f"Solar irradiance at {r['ghi']:.0f} W/m² yields just {r['prod_kw']:.1f} kW from our "
         f"{r['capacity_kw']} kW system. Prioritize essential loads and keep the battery in reserve."),
    lambda r: (f"Cloud-impacted production in {r['location']}: {r['prod_kw']:.1f} kW with {r['cloudcover']:.0f}% "
         f"overcast skies. GHI is {r['ghi']:.0f} W/m² — well below clear-sky potential. "
         f"Hold off on heavy appliances. Battery reserves should be preserved for peak-rate hours."),
    lambda r: (f"Low output conditions in {r['location']}. At {r['cloudcover']:.0f}% clouds, GHI drops to "
         f"{r['ghi']:.0f} W/m² and we're only getting {r['prod_kw']:.1f} kW ({r['prod_kw']/r['capacity_kw']*100:.0f}% "
         f"of capacity). Defer non-urgent loads and let the battery coast until solar improves."),
]
for i, (_, row) in enumerate(_cloudy.iterrows()):
    q = random.choice(_CLOUDY_PHRASES) if random.random() < 0.6 else random.choice(_STATUS_PHRASES)
    a = _a2_templates[i % len(_a2_templates)](row)
    EXAMPLES.append((SYS, q, a))

# A3: Nighttime status (~20)
_night = _sample_meteo((df_meteo["hour"] < 6) | (df_meteo["hour"] > 20), 20)
_a3_templates = [
    lambda h, loc, rate, period: (f"It's {h}:00 in {loc} — no solar production at this hour. The community "
         f"is running on battery reserves and grid power. Current grid rate is ${rate:.2f}/kWh "
         f"({period}). Minimize non-essential usage to preserve battery capacity for the "
         f"morning ramp period."),
    lambda h, loc, rate, period: (f"Nighttime in {loc} ({h}:00). Panels are idle — all power comes from "
         f"battery and grid ({period} at ${rate:.2f}/kWh). Keep consumption low and "
         f"save battery for the pre-dawn hours when grid rates may shift."),
    lambda h, loc, rate, period: (f"Zero solar at {h}:00, {loc}. Community draws from stored energy "
         f"and the grid at {period} rates (${rate:.2f}/kWh). "
         f"Non-essential loads should wait until morning solar ramp begins around 7-8am."),
]
for i, (_, row) in enumerate(_night.iterrows()):
    q = random.choice(_NIGHT_PHRASES) if random.random() < 0.6 else random.choice(_STATUS_PHRASES)
    hour = row["hour"]
    loc = row["location"]
    period, rate = _grid_period(hour)
    a = _a3_templates[i % len(_a3_templates)](hour, loc, rate, period)
    EXAMPLES.append((SYS, q, a))

# A4: Appliance decisions (~30)
_mixed = _sample_meteo(df_meteo["ghi"] > 0, 30)
for i, (_, row) in enumerate(_mixed.iterrows()):
    q = random.choice(_APPLIANCE_PHRASES)
    prod = row["prod_kw"]
    ghi = row["ghi"]
    hour = row["hour"]
    loc = row["location"]
    period, rate = _grid_period(hour)
    if prod > 30:
        _a4_advices = [
            f"Yes — we have strong solar surplus at {prod:.1f} kW. Running appliances now uses free solar energy instead of grid power.",
            f"Go for it! At {prod:.1f} kW we're well above household demand. {loc} is producing enough to run heavy loads and still charge the battery.",
            f"Absolutely. Solar output is {prod:.1f} kW — that's surplus territory. Run whatever you need; we're generating more than the community can consume.",
        ]
    elif prod > 10:
        _a4_advices = [
            f"Moderate production at {prod:.1f} kW. Running smaller appliances is fine, but save heavy loads for the peak solar window (10am-2pm).",
            f"We're at {prod:.1f} kW — enough for light-to-medium loads. Hold the heavy stuff (EV, dryer) for peak solar hours if possible.",
            f"Partial solar at {prod:.1f} kW in {loc}. A dishwasher or washing machine is fine, but the dryer or EV charger would push us to grid at ${rate:.2f}/kWh ({period}).",
        ]
    else:
        _a4_advices = [
            f"Production is low ({prod:.1f} kW). It's {period} pricing at ${rate:.2f}/kWh. Wait for better solar conditions if you can defer the load.",
            f"Only {prod:.1f} kW right now — most appliances would draw from the grid at ${rate:.2f}/kWh ({period}). Defer if you can, or use battery reserves.",
            f"Low solar ({prod:.1f} kW) in {loc}. Running heavy loads now costs ${rate:.2f}/kWh from the grid. Best to wait until conditions improve.",
        ]
    advice = _a4_advices[i % len(_a4_advices)]
    _a4_templates = [
        f"Current status: {prod:.1f} kW generation, GHI {ghi:.0f} W/m² in {loc}. {advice}",
        f"Right now in {loc}: {prod:.1f} kW output from {ghi:.0f} W/m² GHI ({period} rates). {advice}",
        f"Solar check for {loc} at {hour}:00: {prod:.1f} kW, {ghi:.0f} W/m² irradiance. {advice}",
    ]
    a = _a4_templates[i % len(_a4_templates)]
    EXAMPLES.append((SYS, q, a))

# A5: Seasonal comparisons (~30)
_seasonal_data = df_meteo.groupby(["location", "season"]).agg(
    avg_ghi=("ghi", "mean"), avg_prod=("prod_kw", "mean"), avg_temp=("temp_f", "mean")
).reset_index()
_seasonal_samples = _sample_meteo(df_meteo["ghi"] > 0, 30)
for i, (_, row) in enumerate(_seasonal_samples.iterrows()):
    q = random.choice(_COMPARE_PHRASES)
    loc = row["location"]
    season = row["season"]
    _avg = _seasonal_data[
        (_seasonal_data["location"] == loc) & (_seasonal_data["season"] == season)
    ]
    if len(_avg) > 0:
        avg_ghi = _avg.iloc[0]["avg_ghi"]
        avg_prod = _avg.iloc[0]["avg_prod"]
        diff_pct = (row["ghi"] - avg_ghi) / max(avg_ghi, 1) * 100
        comparison = "above" if diff_pct > 5 else "below" if diff_pct < -5 else "right at"
        _verdict = 'Great performance!' if diff_pct > 5 else 'Conditions will improve.' if diff_pct < -5 else 'Typical conditions.'
        _a5_templates = [
            (f"In {loc}, the {season} average GHI is {avg_ghi:.0f} W/m² with typical "
             f"production of {avg_prod:.1f} kW. Right now we're at {row['ghi']:.0f} W/m² "
             f"({row['prod_kw']:.1f} kW) — that's {comparison} the seasonal norm "
             f"({diff_pct:+.0f}%). {_verdict}"),
            (f"Seasonal comparison for {loc} ({season}): current {row['ghi']:.0f} W/m² vs "
             f"average {avg_ghi:.0f} W/m² ({diff_pct:+.0f}%). Output: {row['prod_kw']:.1f} kW "
             f"vs typical {avg_prod:.1f} kW. {_verdict}"),
            (f"{loc} {season} benchmark: we expect {avg_ghi:.0f} W/m² / {avg_prod:.1f} kW on average. "
             f"Today's reading is {row['ghi']:.0f} W/m² / {row['prod_kw']:.1f} kW — "
             f"{comparison} normal ({diff_pct:+.0f}%). {_verdict}"),
        ]
        a = _a5_templates[i % len(_a5_templates)]
    else:
        a = f"Current production in {loc} is {row['prod_kw']:.1f} kW at {row['ghi']:.0f} W/m² GHI."
    EXAMPLES.append((SYS, q, a))

# A6: Temperature derating focus (~20)
_hot = _sample_meteo(df_meteo["temp_f"] > 90, 20, replace=True)
_a6_q_templates = [
    "It's {temp}°F outside. How is the heat affecting our panels?",
    "Temperature hit {temp}°F in {loc}. What's the thermal impact on solar?",
    "Heat wave at {temp}°F — how much production are we losing?",
    "{temp}°F today in {loc}. Are the panels overheating?",
    "How bad is thermal derating at {temp}°F?",
]
for i, (_, row) in enumerate(_hot.iterrows()):
    q = _a6_q_templates[i % len(_a6_q_templates)].format(temp=f"{row['temp_f']:.0f}", loc=row['location'])
    derate = row["temp_derate"]
    prod = row["prod_kw"]
    cap = row["capacity_kw"]
    loss_pct = (1 - derate) * 100
    _theoretical = cap * row["ghi"] / 1000 * SYSTEM_EFF
    _a6_answers = [
        (f"High temperature alert in {row['location']}. At {row['temp_f']:.0f}°F, panels lose "
         f"approximately {loss_pct:.1f}% efficiency due to thermal derating (factor: {derate:.3f}). "
         f"Current production is {prod:.1f} kW vs theoretical {_theoretical:.1f} kW "
         f"at 77°F. This is normal — silicon panels lose ~0.4% efficiency per degree above 77°F. "
         f"Production will recover as temperatures cool in late afternoon."),
        (f"Thermal derating active in {row['location']}: {row['temp_f']:.0f}°F means a "
         f"{loss_pct:.1f}% efficiency loss (derating factor {derate:.3f}). Output is {prod:.1f} kW — "
         f"at the reference temperature of 77°F, we'd be hitting {_theoretical:.1f} kW. "
         f"Silicon panels shed ~0.4%/°F above 77°F. Expect improvement toward evening as temps drop."),
        (f"At {row['temp_f']:.0f}°F in {row['location']}, thermal derating cuts output by {loss_pct:.1f}%. "
         f"Producing {prod:.1f} kW instead of the {_theoretical:.1f} kW potential at 77°F. "
         f"Derating factor: {derate:.3f}. Wind helps cool panels slightly. "
         f"Late afternoon temps will ease the penalty — production should recover 5-10% by 5pm."),
    ]
    a = _a6_answers[i % len(_a6_answers)]
    EXAMPLES.append((SYS, q, a))

# A7: Dawn/dusk ramp profiles (~20)
_dawn_phrases = [
    "Panels just started producing. What should we expect this morning?",
    "It's early morning — what's the solar ramp-up looking like?",
    "Morning update: when will we hit meaningful production today?",
    "The sun is just coming up. What's the forecast for today's solar?",
    "How's the dawn ramp progressing?",
    "Are the panels waking up? What should we expect?",
    "Early morning solar check — are we generating yet?",
    "What's the sunrise production outlook?",
    "How long until we're at peak solar output?",
    "Morning ramp report, please.",
    "Sun's up — give me the early production status.",
    "What does the morning ramp look like for today?",
    "Should I wait for more solar before running appliances?",
    "How quickly will production ramp up this morning?",
    "First light check — how are the panels doing?",
]
_dawn = _sample_meteo((df_meteo["hour"] >= 6) & (df_meteo["hour"] <= 8) & (df_meteo["ghi"] > 0), 20)
for _, row in _dawn.iterrows():
    q = random.choice(_dawn_phrases)
    prod = row["prod_kw"]
    ghi = row["ghi"]
    clouds = row["cloudcover"]
    loc = row["location"]
    cap = row["capacity_kw"]
    if clouds < 30:
        _dawn_templates = [
            (f"Early morning ramp — currently {prod:.1f} kW ({ghi:.0f} W/m² GHI). "
             f"Clear morning in {loc} with {clouds:.0f}% clouds. "
             f"GHI is ramping at {ghi:.0f} W/m² — expect peak production of "
             f"{cap * 0.7:.0f}–{cap * 0.85:.0f} kW by midday. "
             f"A great day for scheduling heavy loads between 10am and 2pm."),
            (f"Sunrise in {loc}: {prod:.1f} kW and climbing under clear skies ({clouds:.0f}% clouds). "
             f"GHI at {ghi:.0f} W/m², ramping toward a projected peak of "
             f"{cap * 0.7:.0f}–{cap * 0.85:.0f} kW around noon. "
             f"Start charging the battery now; schedule heavy loads for 10am–2pm."),
            (f"Good morning — {loc} is looking clear ({clouds:.0f}% clouds). "
             f"Production at {prod:.1f} kW with {ghi:.0f} W/m² GHI and rising steadily. "
             f"By peak hours we should reach {cap * 0.7:.0f}+ kW. "
             f"Ideal conditions for running appliances on pure solar."),
        ]
    else:
        _dawn_templates = [
            (f"Early morning ramp — currently {prod:.1f} kW ({ghi:.0f} W/m² GHI). "
             f"Partly cloudy start in {loc} ({clouds:.0f}% coverage). "
             f"GHI at {ghi:.0f} W/m² and rising. Production may peak around "
             f"{cap * 0.3:.0f}–{cap * 0.5:.0f} kW with intermittent cloud dips. "
             f"Keep the battery charging to buffer variable output."),
            (f"Cloudy dawn in {loc} — {prod:.1f} kW with {clouds:.0f}% overcast. "
             f"GHI is {ghi:.0f} W/m², expect gradual ramp to "
             f"{cap * 0.3:.0f}–{cap * 0.5:.0f} kW peak. "
             f"Variable conditions ahead — battery will help smooth production dips."),
            (f"Morning ramp under clouds in {loc}: {prod:.1f} kW at {ghi:.0f} W/m² GHI. "
             f"Cloud cover at {clouds:.0f}% means output will be inconsistent. "
             f"Estimated peak: {cap * 0.3:.0f}–{cap * 0.5:.0f} kW. "
             f"Defer heavy loads until conditions improve or use battery as buffer."),
        ]
    a = random.choice(_dawn_templates)
    EXAMPLES.append((SYS, q, a))

print(f"    Category A: {len(EXAMPLES)} examples")
_cat_a_count = len(EXAMPLES)

# ---------- Category B: PVWatts Ground Truth (~80 examples) ------------------
print("  Generating Category B: PVWatts Ground Truth...")

if not df_pvwatts_monthly.empty:
    # B1: Monthly production reports (~24)
    for _, row in df_pvwatts_monthly.iterrows():
        loc = row["location"]
        month_name = datetime(2025, int(row["month"]), 1).strftime("%B")
        ac = row["ac_monthly_kwh"]
        solrad = row["solrad_monthly"]
        q = f"What's the expected solar production for {month_name} in {loc}?"
        a = (f"Based on NREL PVWatts data for {loc}, {month_name} typically produces "
             f"{ac:,.0f} kWh with average solar radiation of {solrad:.2f} kWh/m²/day. "
             f"This accounts for system losses, panel degradation, and local weather patterns. "
             f"The annual total for this system is {row['ac_annual_kwh']:,.0f} kWh.")
        EXAMPLES.append((SYS, q, a))

    # B2: System sizing questions (~16)
    for loc_name, loc in LOCATIONS.items():
        _loc_pvw = df_pvwatts_monthly[df_pvwatts_monthly["location"] == loc_name]
        if len(_loc_pvw) == 0:
            continue
        annual = _loc_pvw["ac_annual_kwh"].iloc[0]
        cap = loc["capacity_kw"]
        best_month = _loc_pvw.loc[_loc_pvw["ac_monthly_kwh"].idxmax()]
        worst_month = _loc_pvw.loc[_loc_pvw["ac_monthly_kwh"].idxmin()]
        best_name = datetime(2025, int(best_month["month"]), 1).strftime("%B")
        worst_name = datetime(2025, int(worst_month["month"]), 1).strftime("%B")

        cf = annual / (cap * 8760) * 100
        _homes = annual / 12 / 900
        _b2_questions = [
            f"Is our {cap}kW system in {loc_name} properly sized for the community?",
            f"What's the production range across seasons in {loc_name}?",
            f"How much power does our {loc_name} installation generate annually?",
            f"What's the best and worst month for solar in {loc_name}?",
            f"How does {loc_name}'s solar potential compare month-to-month?",
            f"Give me the annual production forecast for {loc_name}.",
            f"What capacity factor does our {loc_name} system achieve?",
            f"How many homes can our {loc_name} system support?",
        ]
        _b2_answers = [
            (f"The {cap}kW system in {loc_name} produces approximately {annual:,.0f} kWh/year "
             f"(capacity factor: {cf:.1f}%). Best month is {best_name} at "
             f"{best_month['ac_monthly_kwh']:,.0f} kWh, worst is {worst_name} at "
             f"{worst_month['ac_monthly_kwh']:,.0f} kWh. At the {loc['tilt']}° tilt and "
             f"{loc['azimuth']}° azimuth, we're within {loc['losses']}% system losses. "
             f"This covers an average household consumption of ~900 kWh/month for roughly "
             f"{_homes:.0f} homes."),
            (f"{loc_name} system overview: {cap} kW array generating {annual:,.0f} kWh annually. "
             f"Capacity factor: {cf:.1f}%. Seasonal range: {worst_month['ac_monthly_kwh']:,.0f} kWh "
             f"({worst_name}) to {best_month['ac_monthly_kwh']:,.0f} kWh ({best_name}). "
             f"Configuration: {loc['tilt']}° tilt, {loc['azimuth']}° azimuth, {loc['losses']}% losses. "
             f"Supports roughly {_homes:.0f} homes at average consumption."),
            (f"Annual forecast for {loc_name}: {annual:,.0f} kWh from {cap} kW panels. "
             f"CF={cf:.1f}%. Peak production in {best_name} ({best_month['ac_monthly_kwh']:,.0f} kWh), "
             f"lowest in {worst_name} ({worst_month['ac_monthly_kwh']:,.0f} kWh). "
             f"System sized for ~{_homes:.0f} homes at 900 kWh/month each. "
             f"Tilt/azimuth optimized at {loc['tilt']}°/{loc['azimuth']}°."),
        ]
        for idx, q_template in enumerate(_b2_questions):
            a = _b2_answers[idx % len(_b2_answers)]
            EXAMPLES.append((SYS, q_template, a))

    # B3: Tilt optimization (~8)
    for loc_name, loc in LOCATIONS.items():
        _loc_pvw = df_pvwatts_monthly[df_pvwatts_monthly["location"] == loc_name]
        if len(_loc_pvw) == 0:
            continue
        annual = _loc_pvw["ac_annual_kwh"].iloc[0]
        summer_ac = _loc_pvw[_loc_pvw["month"].isin([6, 7, 8])]["ac_monthly_kwh"].sum()
        winter_ac = _loc_pvw[_loc_pvw["month"].isin([12, 1, 2])]["ac_monthly_kwh"].sum()
        for q in [
            f"Should we adjust the panel tilt angle in {loc_name}?",
            f"Is {loc['tilt']}° the optimal tilt for {loc_name}?",
            f"What tilt angle maximizes annual production in {loc_name}?",
            f"How does tilt affect seasonal balance in {loc_name}?",
        ]:
            a = (f"The current {loc['tilt']}° tilt in {loc_name} (latitude {loc['lat']}°) produces "
                 f"{annual:,.0f} kWh/year. Summer output is {summer_ac:,.0f} kWh vs winter "
                 f"{winter_ac:,.0f} kWh — a {summer_ac/max(winter_ac,1):.1f}:1 ratio. "
                 f"A steeper tilt (latitude + 10-15°) would boost winter production at the cost "
                 f"of summer peak. For year-round community use, the current angle is a good "
                 f"compromise. Fixed-mount adjustments aren't cost-effective — seasonal trackers "
                 f"add ~15% but cost significantly more.")
            EXAMPLES.append((SYS, q, a))

    # B4: Performance vs expected (~16)
    _perf_samples = _sample_meteo(df_meteo["ghi"] > 100, 16)
    for _, row in _perf_samples.iterrows():
        loc = row["location"]
        month = row["month"]
        _loc_pvw = df_pvwatts_monthly[
            (df_pvwatts_monthly["location"] == loc) & (df_pvwatts_monthly["month"] == month)
        ]
        if len(_loc_pvw) == 0:
            continue
        expected_hourly_avg = _loc_pvw["ac_monthly_kwh"].iloc[0] / 730  # rough hourly avg
        actual = row["prod_kw"]
        ratio = actual / max(expected_hourly_avg, 0.1)
        month_name = datetime(2025, int(month), 1).strftime("%B")
        q = f"Are our panels in {loc} performing as expected for {month_name}?"
        if ratio > 1.2:
            assessment = "above expectations — conditions are favorable"
        elif ratio > 0.8:
            assessment = "within normal range for this time of year"
        else:
            assessment = "below expected — likely due to cloud cover or haze"
        a = (f"For {month_name} in {loc}, PVWatts expects an average hourly output of "
             f"~{expected_hourly_avg:.1f} kW. Current actual production is {actual:.1f} kW — "
             f"that's {assessment} (ratio: {ratio:.2f}x). GHI is {row['ghi']:.0f} W/m² with "
             f"{row['cloudcover']:.0f}% cloud cover.")
        EXAMPLES.append((SYS, q, a))
else:
    # PVWatts unavailable — generate from Open-Meteo monthly aggregates
    for loc_name in LOCATIONS:
        _loc_meteo = df_meteo[df_meteo["location"] == loc_name]
        _monthly_agg = _loc_meteo.groupby("month").agg(
            total_kwh=("prod_kw", "sum"), avg_ghi=("ghi", "mean")
        ).reset_index()
        for _, row in _monthly_agg.iterrows():
            month_name = datetime(2025, int(row["month"]), 1).strftime("%B")
            q = f"What's the expected solar production for {month_name} in {loc_name}?"
            a = (f"Based on historical weather data for {loc_name}, {month_name} "
                 f"typically produces approximately {row['total_kwh']:,.0f} kWh with "
                 f"average GHI of {row['avg_ghi']:.0f} W/m².")
            EXAMPLES.append((SYS, q, a))

_cat_b_count = len(EXAMPLES) - _cat_a_count
print(f"    Category B: {_cat_b_count} examples")

# ---------- Category C: Grid Mix & Carbon (~80 examples) ---------------------
print("  Generating Category C: Grid Mix & Carbon...")

_cat_c_start = len(EXAMPLES)

for loc_name, loc in LOCATIONS.items():
    region = loc["grid_region"]
    cap = loc["capacity_kw"]
    eia_code = _EIA_RESPONDENT.get(region, region)

    # Get grid data (live or fallback)
    _use_live_mix = False
    if _eia_available and not df_eia_mix.empty:
        _region_mix = df_eia_mix[df_eia_mix["respondent"] == eia_code]
        if len(_region_mix) > 0:
            avg_renew = _region_mix["renewable_pct"].mean()
            avg_co2 = _region_mix["co2_intensity"].mean()
            peak_renew = _region_mix["renewable_pct"].max()
            # Compute live fuel mix percentages from pivoted data
            _fuel_cols = [c for c in _region_mix.columns if c in _CO2_FACTORS]
            _fuel_avg = _region_mix[_fuel_cols].mean()
            _fuel_total = _fuel_avg.sum()
            if _fuel_total > 0:
                _fuel_pcts = (_fuel_avg / _fuel_total * 100).to_dict()
                _use_live_mix = True
        else:
            fb = _FALLBACK_GRID[region]
            avg_renew, avg_co2 = fb["renewable_pct"], fb["co2_intensity"]
            peak_renew = avg_renew * 1.5
    else:
        fb = _FALLBACK_GRID[region]
        avg_renew, avg_co2 = fb["renewable_pct"], fb["co2_intensity"]
        peak_renew = avg_renew * 1.5

    # Build fuel mix string from live data or fallback
    if _use_live_mix:
        _coal = _fuel_pcts.get("COL", 0)
        _ng = _fuel_pcts.get("NG", 0)
        _nuc = _fuel_pcts.get("NUC", 0)
        _wind = _fuel_pcts.get("WND", 0)
        _solar = _fuel_pcts.get("SUN", 0)
        _mix_source = "live EIA data"
    else:
        fb = _FALLBACK_GRID[region]
        _coal = fb.get("coal_pct", 0)
        _ng = fb["ng_pct"]
        _nuc = fb["nuclear_pct"]
        _wind = fb["wind_pct"]
        _solar = fb["solar_pct"]
        _mix_source = "reference data"

    # C1: Grid composition (~10 per location) — question-responsive + temporal
    # Pre-compute temporal EIA slices for varied answers
    _has_temporal = _use_live_mix and "period" in _region_mix.columns
    _night_stats = _solar_stats = None
    if _has_temporal:
        _rmh = _region_mix.copy()
        _rmh["_hour"] = pd.to_datetime(_rmh["period"]).dt.hour

        def _slice_stats(hour_mask):
            """Compute fuel mix from a temporal slice of EIA hourly data."""
            s = _rmh[hour_mask]
            if len(s) == 0:
                return None
            fc = [c for c in s.columns if c in _CO2_FACTORS]
            fa = s[fc].mean()
            ft = fa.sum()
            if ft > 0:
                fa = fa / ft * 100
            return {"coal": fa.get("COL", 0), "ng": fa.get("NG", 0),
                    "nuc": fa.get("NUC", 0), "wind": fa.get("WND", 0),
                    "solar": fa.get("SUN", 0),
                    "renew": s["renewable_pct"].mean(),
                    "co2": s["co2_intensity"].mean()}

        _night_stats = _slice_stats(_rmh["_hour"].isin([0, 1, 2, 3, 4, 5]))
        _solar_stats = _slice_stats(_rmh["_hour"].isin([11, 12, 13, 14]))

    # Pre-compute derived strings to avoid nested f-strings
    _fossil_pct = 100 - avg_renew
    _coal_note = f" and coal ({_coal:.0f}%)" if _coal > 1 else ""
    _coal_detail = f"Coal: {_coal:.0f}% — baseload fossil" if _coal > 1 else "Coal: <1% — effectively phased out"
    _ranked = sorted(
        [("natural gas", _ng), ("coal", _coal), ("nuclear", _nuc),
         ("wind", _wind), ("solar", _solar)],
        key=lambda x: -x[1])
    _top_fuels = ", ".join(f"{n} ({p:.0f}%)" for n, p in _ranked if p > 0.5)
    _us_compare = ("significantly above" if avg_renew > 35 else
                   "above" if avg_renew > 21 else "below")
    _co2_label = ("relatively clean" if avg_co2 < 200 else
                  "moderate" if avg_co2 < 400 else "carbon-heavy")
    _primary_renew = (f"Strong solar ({_solar:.0f}%) drives the advantage."
                      if _solar > 15 else
                      f"Wind ({_wind:.0f}%) is the primary renewable contributor.")

    # 10 question-answer pairs, each with a unique angle
    _c1_pairs = [
        (f"What's the current grid mix for {region}?",
         f"The {region} grid averages: coal {_coal:.0f}%, natural gas {_ng:.0f}%, "
         f"nuclear {_nuc:.0f}%, wind {_wind:.0f}%, solar {_solar:.0f}%. "
         f"Renewables total {avg_renew:.1f}%, with CO2 intensity at {avg_co2:.0f} kg/MWh."),

        (f"How clean is the grid in {loc_name} right now?",
         f"{region} runs at {avg_renew:.1f}% renewable — "
         f"{_us_compare} the US average of ~21%. "
         f"Wind contributes {_wind:.0f}% and solar {_solar:.0f}%. "
         f"Each kWh our panels generate displaces {avg_co2/1000:.3f} kg of grid CO2."),

        (f"What percentage of {region} power is renewable?",
         f"Currently {avg_renew:.1f}% of {region} generation is renewable: "
         f"solar {_solar:.0f}%, wind {_wind:.0f}%. Nuclear adds {_nuc:.0f}% zero-carbon baseload. "
         f"The remaining {_fossil_pct:.0f}% is fossil — primarily natural gas ({_ng:.0f}%)"
         f"{_coal_note}."),

        (f"Break down the electricity sources for {loc_name}.",
         f"Fuel-by-fuel for {region}:\n"
         f"• Natural gas: {_ng:.0f}% (primary dispatchable)\n"
         f"• {_coal_detail}\n"
         f"• Nuclear: {_nuc:.0f}% (zero-carbon baseload)\n"
         f"• Wind: {_wind:.0f}% (variable)\n"
         f"• Solar: {_solar:.0f}% (peaks midday)\n"
         f"CO2 intensity: {avg_co2:.0f} kg/MWh overall."),

        (f"How much fossil fuel is in {loc_name}'s grid?",
         f"Fossil fuels account for {_fossil_pct:.0f}% of {region} generation. "
         f"Natural gas leads at {_ng:.0f}%"
         f"{_coal_note}. "
         f"This drives CO2 to {avg_co2:.0f} kg/MWh. Community solar directly "
         f"displaces this fossil generation — each kWh avoids {avg_co2/1000:.3f} kg CO2."),

        (f"What fuels power the {region} grid?",
         f"Top {region} sources by share: {_top_fuels}. "
         f"Overall: {avg_renew:.1f}% renewable, {_fossil_pct:.0f}% fossil, "
         f"{_nuc:.0f}% nuclear. CO2 intensity: {avg_co2:.0f} kg/MWh."),
    ]

    # Q7: Trend — compare peak solar hours vs average (temporal)
    if _solar_stats:
        _c1_pairs.append((
            f"Is the {region} grid getting cleaner?",
            f"During peak solar hours (11am–2pm), {region} hits {_solar_stats['renew']:.1f}% "
            f"renewable with CO2 dropping to {_solar_stats['co2']:.0f} kg/MWh. "
            f"The daily average is {avg_renew:.1f}% / {avg_co2:.0f} kg/MWh. "
            f"This intraday swing shows solar penetration — our panels amplify this trend."))
    else:
        _c1_pairs.append((
            f"Is the {region} grid getting cleaner?",
            f"The {region} grid averages {avg_renew:.1f}% renewable with {avg_co2:.0f} kg/MWh CO2. "
            f"{_primary_renew} "
            f"Community solar accelerates grid transition by displacing fossil generation."))

    # Q8: National comparison
    _c1_pairs.append((
        f"How does {region} compare to the national average for renewables?",
        f"The US average is ~21% renewable. {region} at {avg_renew:.1f}% is "
        f"{_us_compare} the national benchmark. {_primary_renew} "
        f"Community solar adds {cap} kW of distributed clean energy on top."))

    # Q9: Carbon intensity focus
    _c1_pairs.append((
        f"What's the carbon intensity of {region} electricity?",
        f"{region} averages {avg_co2:.0f} kg CO2 per MWh — {_co2_label} "
        f"for a US grid region. Each kWh our {cap} kW array generates avoids "
        f"{avg_co2/1000:.3f} kg CO2. At full output, that's "
        f"{cap * avg_co2 / 1000:.1f} kg/hour of avoided emissions."))

    # Q10: Nighttime dirtiness — temporal snapshot
    if _night_stats:
        _n_coal_str = f", coal rises to {_night_stats['coal']:.0f}%" if _night_stats["coal"] > 1 else ""
        _c1_pairs.append((
            f"How dirty is the grid when our solar isn't producing?",
            f"After dark (midnight–5am), {region} drops to {_night_stats['renew']:.1f}% "
            f"renewable with CO2 climbing to {_night_stats['co2']:.0f} kg/MWh. "
            f"Natural gas rises to {_night_stats['ng']:.0f}%{_n_coal_str}. "
            f"Compare to daytime average ({avg_renew:.1f}% renewable). "
            f"Battery storage helps avoid this dirtier nighttime grid."))
    else:
        _c1_pairs.append((
            f"How dirty is the grid when our solar isn't producing?",
            f"Without solar, {region} leans on fossil fuels — mostly natural gas ({_ng:.0f}%)"
            f"{_coal_note}. "
            f"CO2 intensity is {avg_co2:.0f} kg/MWh on average, likely higher at night. "
            f"Our battery helps the community avoid drawing from this dirtier grid."))

    for q, a in _c1_pairs:
        EXAMPLES.append((SYS, q, a))

    # C2: Carbon savings (~10 per location)
    _loc_meteo = df_meteo[df_meteo["location"] == loc_name]
    _daily_prod = _loc_meteo.groupby(_loc_meteo["time"].dt.date)["prod_kw"].sum()
    _sample_days = _daily_prod.sample(n=min(10, len(_daily_prod)), random_state=42)
    for date, daily_kwh in _sample_days.items():
        co2_saved = daily_kwh * avg_co2 / 1000  # kg
        q = random.choice([
            f"How much CO2 did our {loc_name} panels save on {date}?",
            f"What's the carbon impact of our {loc_name} solar on {date}?",
            f"How many kg of CO2 did we avoid on {date} in {loc_name}?",
        ])
        a = (f"On {date}, the {loc_name} community generated {daily_kwh:.0f} kWh of solar power. "
             f"At {region}'s average CO2 intensity of {avg_co2:.0f} kg/MWh, that displaced "
             f"approximately {co2_saved:.1f} kg of CO2 emissions. That's equivalent to "
             f"{co2_saved / 8.887:.1f} gallons of gasoline not burned, or "
             f"{co2_saved / 0.42:.0f} miles not driven in an average car.")
        EXAMPLES.append((SYS, q, a))

    # C3: Peak demand economics (~10 per location)
    for hour in [7, 10, 14, 16, 18, 20, 22, 0, 3, 12]:
        period, rate = _grid_period(hour)
        _hour_rows = _loc_meteo[_loc_meteo["hour"] == hour]
        avg_prod = _hour_rows["prod_kw"].mean() if len(_hour_rows) > 0 else 0
        savings_hr = avg_prod * rate
        q = f"What's the economic value of our solar at {hour}:00 in {loc_name}?"
        if period == "peak":
            _econ_note = "Peak hours (2-7pm) have the highest economic value for solar."
        elif period == "mid-peak":
            _econ_note = "Mid-peak production is solid value — keep running loads on solar."
        elif avg_prod < 1:
            _econ_note = "No solar at this hour — rely on battery or grid."
        else:
            _econ_note = ""
        a = (f"At {hour}:00 in {loc_name}, average solar production is {avg_prod:.1f} kW. "
             f"Grid rate is ${rate:.2f}/kWh ({period}). Solar displaces "
             f"${savings_hr:.2f}/hour in grid costs. {_econ_note}")
        EXAMPLES.append((SYS, q, a))

    # C4: Temporal grid variation (~10 per location) — intraday snapshots
    if _has_temporal:
        _c4_hours = [0, 3, 6, 9, 11, 13, 15, 18, 20, 22]
        _c4_q_pool = [
            "What does the {region} grid look like at {hour}:00?",
            "Grid snapshot for {loc} at {hour}:00 — how renewable is it?",
            "How does the fuel mix shift at {hour}:00 in {region}?",
            "What's the grid composition at {hour}:00 for {loc}?",
            "How clean is {region} electricity at {hour}:00?",
        ]
        for _c4i, _c4h in enumerate(_c4_hours):
            _hr_stats = _slice_stats(_rmh["_hour"] == _c4h)
            if _hr_stats is None:
                continue
            if _c4h < 6:
                _tod = "Late night"
            elif _c4h < 10:
                _tod = "Morning"
            elif _c4h < 15:
                _tod = "Midday"
            elif _c4h < 19:
                _tod = "Afternoon"
            else:
                _tod = "Evening"
            _c4_coal = f", coal {_hr_stats['coal']:.0f}%" if _hr_stats["coal"] > 1 else ""
            if _hr_stats["solar"] > 20:
                _c4_insight = "Solar is driving renewables to peak levels."
            elif _hr_stats["solar"] < 5:
                _c4_insight = "Solar is minimal — wind and nuclear carry the clean load."
            else:
                _c4_insight = "Moderate solar contribution at this hour."
            q = _c4_q_pool[_c4i % len(_c4_q_pool)].format(
                region=region, loc=loc_name, hour=_c4h)
            a = (f"{_tod} grid snapshot for {region} at {_c4h}:00: "
                 f"{_hr_stats['renew']:.1f}% renewable, CO2 at "
                 f"{_hr_stats['co2']:.0f} kg/MWh. "
                 f"Fuel mix: natural gas {_hr_stats['ng']:.0f}%, "
                 f"solar {_hr_stats['solar']:.0f}%, "
                 f"wind {_hr_stats['wind']:.0f}%, "
                 f"nuclear {_hr_stats['nuc']:.0f}%{_c4_coal}. "
                 f"{_c4_insight}")
            EXAMPLES.append((SYS, q, a))

_cat_c_count = len(EXAMPLES) - _cat_c_start
print(f"    Category C: {_cat_c_count} examples")

# ---------- Category D: Weather Impact (~40 examples) ------------------------
print("  Generating Category D: Weather Impact...")

_cat_d_start = len(EXAMPLES)

# D1: Current OWM-based weather advisory (~10)
if len(df_owm) > 0:
    for _, row in df_owm.iterrows():
        loc = row["location"]
        for q in [
            f"What's the weather doing to our solar in {loc} right now?",
            f"Current weather advisory for {loc} solar?",
            f"How's the weather affecting panels in {loc}?",
            f"What should we expect given current {loc} weather?",
            f"Weather impact report for {loc}?",
        ]:
            _loc_info = LOCATIONS[loc]
            cap = _loc_info["capacity_kw"]
            derate = max(0.75, 1.0 - 0.004 * max(0, row["temp_f"] - 77))
            cloud_factor = 1.0 - row["clouds_pct"] / 100 * 0.75
            est_prod = cap * cloud_factor * derate * 0.5  # rough midday estimate
            _wx_verdict = ('Excellent conditions for solar.' if row['clouds_pct'] < 25
                           else 'Cloud cover is limiting output — expect variable production.' if row['clouds_pct'] < 70
                           else 'Heavy overcast — minimal solar contribution expected.')
            _d1_templates = [
                (f"Current conditions in {loc}: {row['description']}, {row['temp_f']:.0f}°F, "
                 f"{row['clouds_pct']}% clouds, wind {row['wind_mph']:.1f} mph, "
                 f"humidity {row['humidity_pct']}%. "
                 f"Temperature derating factor: {derate:.3f}. "
                 f"Estimated production: ~{est_prod:.1f} kW. "
                 f"Sunrise: {row['sunrise']}, sunset: {row['sunset']}. {_wx_verdict}"),
                (f"Weather advisory for {loc}: {row['description']} with {row['clouds_pct']}% cloud cover. "
                 f"Temperature {row['temp_f']:.0f}°F (derating: {derate:.3f}), wind {row['wind_mph']:.1f} mph, "
                 f"humidity {row['humidity_pct']}%. Producing ~{est_prod:.1f} kW. "
                 f"Sun window: {row['sunrise']}-{row['sunset']}. {_wx_verdict}"),
                (f"{loc} right now: {row['description']}, {row['temp_f']:.0f}°F. "
                 f"Clouds: {row['clouds_pct']}%, wind: {row['wind_mph']:.1f} mph, humidity: {row['humidity_pct']}%. "
                 f"Solar estimate: ~{est_prod:.1f} kW (derating factor {derate:.3f} at this temp). "
                 f"Daylight: {row['sunrise']} to {row['sunset']}. {_wx_verdict}"),
            ]
            a = _d1_templates[hash(q) % len(_d1_templates)]
            EXAMPLES.append((SYS, q, a))

# D2: High humidity/haze (~10)
_humid = _sample_meteo(df_meteo["relative_humidity_2m"] > 85, 10)
_d2_q_templates = [
    "Humidity is at {h}% in {loc}. How does that affect solar output?",
    "It's {h}% humidity in {loc}. Any impact on the panels?",
    "Haze report for {loc} — humidity at {h}%. What's the solar impact?",
]
for i, (_, row) in enumerate(_humid.iterrows()):
    humidity = row["relative_humidity_2m"]
    ghi = row["ghi"]
    loc = row["location"]
    q = _d2_q_templates[i % len(_d2_q_templates)].format(h=f"{humidity:.0f}", loc=loc)
    haze_loss = min(15, (humidity - 70) * 0.3)
    _dew_note = 'Panels may also have condensation — check for dew.' if humidity > 95 else 'No condensation risk at this level.'
    _d2_a_templates = [
        (f"High humidity ({humidity:.0f}%) in {loc} creates atmospheric haze that scatters "
         f"sunlight. GHI is {ghi:.0f} W/m², which may be {haze_loss:.0f}% below clear-air "
         f"potential. Current production: {row['prod_kw']:.1f} kW. Panels still generate — "
         f"diffuse radiation from haze contributes, but direct beam is reduced. {_dew_note}"),
        (f"Humidity impact in {loc}: {humidity:.0f}% creates haze, scattering direct sunlight. "
         f"GHI reads {ghi:.0f} W/m² — potentially {haze_loss:.0f}% below what clear air would deliver. "
         f"Output: {row['prod_kw']:.1f} kW. Diffuse radiation still reaches the panels, "
         f"but expect reduced efficiency. {_dew_note}"),
        (f"At {humidity:.0f}% humidity, {loc} has significant atmospheric moisture scattering. "
         f"Solar irradiance: {ghi:.0f} W/m² (est. {haze_loss:.0f}% haze penalty). "
         f"Producing {row['prod_kw']:.1f} kW — panels work but direct beam is weakened. {_dew_note}"),
    ]
    a = _d2_a_templates[i % len(_d2_a_templates)]
    EXAMPLES.append((SYS, q, a))

# D3: Wind impact (~10)
_windy = _sample_meteo(df_meteo["wind_mph"] > 15, 10, replace=True)
_d3_q_templates = [
    "Winds are at {w} mph in {loc}. Any concerns for the panels?",
    "High winds in {loc} ({w} mph). How does that affect solar?",
    "Wind advisory for {loc}: {w} mph. Are the panels safe?",
]
for i, (_, row) in enumerate(_windy.iterrows()):
    wind = row["wind_mph"]
    temp = row["temp_f"]
    loc = row["location"]
    q = _d3_q_templates[i % len(_d3_q_templates)].format(w=f"{wind:.0f}", loc=loc)
    _wind_advice = ('Strong winds — check mounting hardware and listen for vibration. '
                    'Panels rated to 90+ mph but sustained gusts above 40 mph warrant visual inspection.'
                    if wind > 30 else
                    'Moderate wind is actually beneficial — it cools panels, reducing thermal derating.')
    _d3_a_templates = [
        (f"Wind at {wind:.0f} mph in {loc}. {_wind_advice} "
         f"At {temp:.0f}°F with {wind:.0f} mph wind, effective panel temperature is lower than "
         f"ambient, which slightly improves efficiency. Current production: {row['prod_kw']:.1f} kW."),
        (f"Wind report for {loc}: {wind:.0f} mph. {_wind_advice} "
         f"The cooling effect at {temp:.0f}°F means slightly better panel efficiency than still air. "
         f"Output: {row['prod_kw']:.1f} kW."),
        (f"{loc} at {wind:.0f} mph wind and {temp:.0f}°F. {_wind_advice} "
         f"Wind cools panels below ambient, reducing thermal losses. "
         f"Current output: {row['prod_kw']:.1f} kW — wind is a net positive for efficiency."),
    ]
    a = _d3_a_templates[i % len(_d3_a_templates)]
    EXAMPLES.append((SYS, q, a))

# D4: Fog / low GHI morning (~10)
_fog = _sample_meteo(
    (df_meteo["hour"].between(6, 10)) & (df_meteo["ghi"] > 0) & (df_meteo["ghi"] < 100) &
    (df_meteo["relative_humidity_2m"] > 80), 10, replace=True
)
_d4_q_templates = [
    "It's foggy this morning in {loc}. When will solar kick in?",
    "Fog advisory in {loc}. How long until meaningful production?",
    "Morning fog in {loc} — what's the production outlook?",
]
for i, (_, row) in enumerate(_fog.iterrows()):
    ghi = row["ghi"]
    humidity = row["relative_humidity_2m"]
    loc = row["location"]
    batt = LOCATIONS[loc]['battery_kwh']
    q = _d4_q_templates[i % len(_d4_q_templates)].format(loc=loc)
    _d4_a_templates = [
        (f"Foggy morning in {loc} — GHI is only {ghi:.0f} W/m² with {humidity:.0f}% humidity. "
         f"Production is minimal at {row['prod_kw']:.1f} kW. Fog typically burns off by "
         f"10-11am as solar heating warms the ground. Expect a rapid ramp to normal production "
         f"once fog clears. Until then, the community is drawing from battery ({batt} kWh capacity) and grid."),
        (f"Fog conditions in {loc}: {ghi:.0f} W/m² GHI, {humidity:.0f}% humidity, {row['prod_kw']:.1f} kW output. "
         f"Coastal/morning fog usually clears by 10-11am. Once it lifts, production will ramp quickly. "
         f"Battery ({batt} kWh) and off-peak grid ($0.10/kWh) cover the gap."),
        (f"Morning fog in {loc} limiting solar to {row['prod_kw']:.1f} kW ({ghi:.0f} W/m² GHI). "
         f"Humidity at {humidity:.0f}% — typical fog conditions. Expect clearance by mid-morning "
         f"with rapid ramp to normal output. Community running on battery ({batt} kWh) "
         f"and grid until then."),
    ]
    a = _d4_a_templates[i % len(_d4_a_templates)]
    EXAMPLES.append((SYS, q, a))

_cat_d_count = len(EXAMPLES) - _cat_d_start
print(f"    Category D: {_cat_d_count} examples")

# ---------- Category E: Battery + Grid Strategy (~80 examples) ---------------
print("  Generating Category E: Battery + Grid Strategy...")

_cat_e_start = len(EXAMPLES)

_soc_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

for loc_name, loc in LOCATIONS.items():
    batt_kwh = loc["battery_kwh"]
    cap = loc["capacity_kw"]
    region = loc["grid_region"]
    fb = _FALLBACK_GRID[region]

    # E1: SOC-aware load management (~20 per location)
    for soc in _soc_levels:
        kwh_stored = batt_kwh * soc / 100
        # Pick a random hour for context
        hour = random.choice([7, 10, 14, 17, 20, 23, 2])
        period, rate = _grid_period(hour)
        _hour_rows = df_meteo[(df_meteo["location"] == loc_name) & (df_meteo["hour"] == hour)]
        avg_prod = _hour_rows["prod_kw"].mean() if len(_hour_rows) > 0 else 0

        q = random.choice(_BATTERY_PHRASES) + f" Battery at {soc}%, {hour}:00, {loc_name}."

        if soc < 20:
            strategy = (f"Critical battery level at {soc}% ({kwh_stored:.0f} kWh). "
                        f"{'Charge from solar immediately.' if avg_prod > 5 else 'Grid charging recommended at off-peak rates.'} "
                        f"Shed all non-essential loads. Reserve remaining capacity for emergency backup.")
        elif soc < 50:
            strategy = (f"Battery at {soc}% ({kwh_stored:.0f} kWh) — moderate reserves. "
                        f"{'Solar is active — prioritize charging.' if avg_prod > 5 else 'Consider grid charging during off-peak.'} "
                        f"Run only essential loads until battery reaches 60%+.")
        elif soc < 80:
            _rate_str = f"${rate:.2f}"
            _export_msg = f"Export surplus at {_rate_str}/kWh." if avg_prod > cap * 0.5 else ""
            strategy = (f"Battery at {soc}% ({kwh_stored:.0f} kWh) — healthy reserves. "
                        f"{'Good solar production — battery will top up.' if avg_prod > 10 else 'Adequate reserves for evening.'} "
                        f"Normal load operations. {_export_msg}")
        else:
            _rate_str = f"${rate:.2f}"
            _export_msg = f"Export solar surplus to grid at {_rate_str}/kWh ({period})." if avg_prod > 10 else "Ready for overnight draw-down."
            strategy = (f"Battery near full at {soc}% ({kwh_stored:.0f} kWh). "
                        f"{'Maximize self-consumption — run heavy appliances now.' if period != 'off-peak' else 'Battery topped up for overnight.'} "
                        f"{_export_msg}")

        a = (f"At {hour}:00 in {loc_name}, grid is {period} (${rate:.2f}/kWh), "
             f"avg production ~{avg_prod:.1f} kW. {strategy}")
        EXAMPLES.append((SYS, q, a))

    # E2: Overnight planning (~16 per location)
    for soc in [15, 25, 30, 40, 50, 60, 70, 90]:
        kwh_stored = batt_kwh * soc / 100
        overnight_draw = random.uniform(8, 15)  # kWh overnight estimate
        hours_left = kwh_stored / max(overnight_draw / 8, 0.1)  # rough hours
        _e2_hours = random.choice([21, 22, 23])
        q = random.choice([
            f"It's {_e2_hours}:00 in {loc_name}, battery at {soc}%. Will we make it through the night?",
            f"Battery is at {soc}% heading into the night in {loc_name}. What's the outlook?",
            f"Nighttime in {loc_name}, SOC at {soc}%. Do we need grid backup overnight?",
        ])
        if soc >= 60:
            outlook = (f"Yes — {kwh_stored:.0f} kWh stored is sufficient for overnight base loads "
                       f"(~{overnight_draw:.0f} kWh). Battery should still be at {max(soc - overnight_draw/batt_kwh*100, 10):.0f}% "
                       f"by sunrise. No grid draw needed.")
        elif soc >= 30:
            outlook = (f"Marginal — {kwh_stored:.0f} kWh may not cover full overnight demand "
                       f"(~{overnight_draw:.0f} kWh). Grid supplementation likely after 3-4am at "
                       f"off-peak rates ($0.10/kWh). Reduce overnight loads to extend battery.")
        else:
            outlook = (f"Unlikely — only {kwh_stored:.0f} kWh remaining. Grid will take over by "
                       f"midnight-1am. Off-peak rate ($0.10/kWh) applies, so grid draw is economical. "
                       f"Consider grid-charging to 50% for emergency reserve.")
        a = f"Battery at {soc}% ({kwh_stored:.0f}/{batt_kwh} kWh) at 10pm. {outlook}"
        EXAMPLES.append((SYS, q, a))

    # E3: Grid arbitrage (~16 per location)
    for buy_period, sell_period, buy_rate, sell_rate in [
        ("off-peak (midnight)", "peak (3pm)", 0.10, 0.28),
        ("off-peak (2am)", "peak (5pm)", 0.10, 0.28),
        ("off-peak (4am)", "mid-peak (8am)", 0.10, 0.18),
        ("mid-peak (8am)", "peak (3pm)", 0.18, 0.28),
        ("off-peak (1am)", "peak (4pm)", 0.10, 0.28),
        ("off-peak (3am)", "peak (6pm)", 0.10, 0.28),
        ("off-peak (5am)", "mid-peak (10am)", 0.10, 0.18),
        ("mid-peak (9am)", "peak (2pm)", 0.18, 0.28),
    ]:
        q = f"Is grid arbitrage worth it in {loc_name}? Buy {buy_period}, sell {sell_period}?"
        margin = sell_rate - buy_rate
        cycles_per_day = 1
        profit_per_cycle = batt_kwh * 0.8 * margin  # 80% usable capacity
        _margin_str = f"${margin:.2f}"
        a = (f"Grid arbitrage analysis for {loc_name} ({region}): "
             f"Buy at ${buy_rate:.2f}/kWh ({buy_period}), sell at ${sell_rate:.2f}/kWh "
             f"({sell_period}). Margin: {_margin_str}/kWh. "
             f"With {batt_kwh} kWh battery (80% usable), profit per cycle: ${profit_per_cycle:.2f}. "
             f"{f'Worthwhile — {_margin_str} spread covers battery degradation costs.' if margin >= 0.10 else f'Marginal — the {_margin_str} spread barely covers cycling costs.'} "
             f"However, solar self-consumption typically provides better ROI than pure arbitrage.")
        EXAMPLES.append((SYS, q, a))

    # E4: Outage scenarios (~8 per location)
    _outage_questions = [
        "Grid outage in {loc}! Battery at {soc}%. How long can we sustain the community?",
        "Power grid is down in {loc}. SOC at {soc}% — what's our backup runway?",
        "Emergency: grid failure in {loc}. Battery has {soc}%. What's the plan?",
        "We just lost grid power in {loc}. Battery is {soc}%. How do we manage?",
        "Outage alert for {loc} — {soc}% battery. How long can we last?",
        "Grid went down in {loc}. With {soc}% battery, what should we prioritize?",
        "Power outage in {loc}! Battery reads {soc}%. Advise on energy rationing.",
        "We're islanded in {loc} — {soc}% SOC. Give me the emergency strategy.",
    ]
    _solar_avg_kw = cap * 0.3
    for idx, (soc, outage_hours) in enumerate([(90, 24), (50, 12), (30, 6), (70, 48),
                                                (80, 8), (40, 24), (20, 4), (60, 36)]):
        kwh_stored = batt_kwh * soc / 100
        essential_load = 3  # kW average essential load
        backup_hours = kwh_stored / essential_load
        q = _outage_questions[idx % len(_outage_questions)].format(loc=loc_name, soc=soc)
        _extended = "multiple days" if backup_hours > 24 else f"{backup_hours * 1.5:.0f}+ hours"
        _e4_templates = [
            (f"Emergency mode in {loc_name}: {kwh_stored:.0f} kWh stored ({soc}%). "
             f"At essential loads (~{essential_load} kW), battery alone lasts "
             f"~{backup_hours:.0f} hours. With solar (~{_solar_avg_kw:.0f} kW daytime avg), "
             f"effective backup extends to {_extended}. "
             f"Priority: refrigeration, medical devices, communication. "
             f"Shed HVAC, water heating, EV charging until grid restores."),
            (f"Outage response for {loc_name}: {soc}% SOC = {kwh_stored:.0f} kWh available. "
             f"Essential load estimate: {essential_load} kW → {backup_hours:.0f} hours battery-only. "
             f"Solar adds ~{_solar_avg_kw:.0f} kW during daylight, extending to {_extended}. "
             f"Immediate actions: cut HVAC and EV charging, preserve power for "
             f"refrigeration and medical equipment."),
            (f"Grid-down in {loc_name} with {kwh_stored:.0f} kWh ({soc}%) in reserve. "
             f"Running essentials at ~{essential_load} kW gives ~{backup_hours:.0f} hours without solar. "
             f"Daytime solar (~{_solar_avg_kw:.0f} kW avg) pushes that to {_extended}. "
             f"Load shedding priority: keep fridges and medical devices, "
             f"drop HVAC, water heater, and EV charger immediately."),
        ]
        a = _e4_templates[idx % len(_e4_templates)]
        EXAMPLES.append((SYS, q, a))

    # E5: Peak-shaving / TOU optimization (~6 per location)
    _tou_scenarios = [
        (14, 35, "Peak just started"),
        (15, 20, "Mid-afternoon peak"),
        (17, 50, "Late peak — grid rate drops at 7pm"),
        (18, 15, "End of peak window"),
        (10, 45, "Morning mid-peak"),
        (20, 70, "Evening mid-peak — overnight approaching"),
    ]
    _tou_questions = [
        f"How should we handle the grid rate transition in {loc_name} right now?",
        f"Peak pricing strategy for {loc_name} — what's the play?",
        f"Rate optimization advice for {loc_name} this hour?",
        f"Best way to minimize grid costs in {loc_name} right now?",
        f"Should we be buying or selling power in {loc_name}?",
        f"Time-of-use strategy check for {loc_name}?",
    ]
    for idx, (hour, soc, context) in enumerate(_tou_scenarios):
        period, rate = _grid_period(hour)
        kwh_stored = batt_kwh * soc / 100
        _hour_rows = df_meteo[(df_meteo["location"] == loc_name) & (df_meteo["hour"] == hour)]
        avg_prod = _hour_rows["prod_kw"].mean() if len(_hour_rows) > 0 else 0
        next_period, next_rate = _grid_period((hour + 2) % 24)
        q = _tou_questions[idx]
        if period == "peak":
            strategy = (f"Peak rate active (${rate:.2f}/kWh). Battery at {soc}% ({kwh_stored:.0f} kWh). "
                        f"{'Discharge battery to avoid grid draw — every kWh saves $0.28.' if soc > 30 else 'Low battery — grid draw unavoidable. Shed non-essential loads to minimize cost.'} "
                        f"Solar producing ~{avg_prod:.0f} kW. {context} — "
                        f"{'rate drops to ${:.2f} at 7pm, hold some reserve.'.format(next_rate) if hour >= 17 else 'peak continues until 7pm.'}")
        elif period == "mid-peak":
            strategy = (f"Mid-peak rate (${rate:.2f}/kWh). Battery at {soc}% ({kwh_stored:.0f} kWh). "
                        f"{'Solar covers most demand — charge battery with surplus for peak hours.' if avg_prod > 15 else 'Moderate solar — supplement with battery as needed.'} "
                        f"{context}. Next rate change: {next_period} at ${next_rate:.2f}/kWh.")
        else:
            strategy = (f"Off-peak rate (${rate:.2f}/kWh). Battery at {soc}% ({kwh_stored:.0f} kWh). "
                        f"{'Good time for grid-assisted battery charging if SOC is below 50%.' if soc < 50 else 'Battery healthy — ready for tomorrow.'} "
                        f"Cheapest grid power available. {context}.")
        a = f"TOU analysis for {loc_name} at {hour}:00 ({period}): {strategy}"
        EXAMPLES.append((SYS, q, a))

_cat_e_count = len(EXAMPLES) - _cat_e_start
print(f"    Category E: {_cat_e_count} examples")

# ---------- Category F: Multi-Step Reasoning (~40 examples) ------------------
print("  Generating Category F: Multi-Step Reasoning...")

_cat_f_start = len(EXAMPLES)

# F1: Full day planning (~10)
for loc_name, loc in LOCATIONS.items():
    _loc_meteo = df_meteo[df_meteo["location"] == loc_name]
    _sample_dates = _loc_meteo.groupby(_loc_meteo["time"].dt.date).first().sample(
        n=min(8, len(_loc_meteo["time"].dt.date.unique())), random_state=42
    )
    for date, day_row in _sample_dates.iterrows():
        _day_data = _loc_meteo[_loc_meteo["time"].dt.date == date]
        peak_prod = _day_data["prod_kw"].max()
        total_kwh = _day_data["prod_kw"].sum()
        avg_clouds = _day_data["cloudcover"].mean()
        avg_temp = _day_data["temp_f"].mean()
        region = loc["grid_region"]
        cap = loc["capacity_kw"]
        batt = loc["battery_kwh"]

        _renew_pct, _co2_rate = _get_grid_stats(region)
        _co2_saved = total_kwh * _co2_rate / 1000
        q = random.choice([
            f"Plan the full energy strategy for {loc_name} on {date}.",
            f"What's the optimal energy plan for {date} in {loc_name}?",
            f"Map out today's energy schedule for {loc_name} ({date}).",
            f"Give me an hour-by-hour energy strategy for {loc_name} on {date}.",
        ])
        _f1_templates = [
            (f"Day plan for {loc_name} ({date}):\n"
             f"• Forecast: avg {avg_clouds:.0f}% clouds, {avg_temp:.0f}°F, peak production {peak_prod:.1f} kW\n"
             f"• Total generation: ~{total_kwh:.0f} kWh from {cap} kW system\n"
             f"• 6-9am: Morning ramp — charge battery from overnight low, defer heavy loads\n"
             f"• 10am-2pm: Peak solar — run dishwasher, laundry, EV charging; battery will top up\n"
             f"• 2-7pm: Peak grid rates ($0.28/kWh) — maximize self-consumption, discharge battery to avoid grid draw\n"
             f"• 7-11pm: Mid-peak wind-down — battery covers evening loads\n"
             f"• Overnight: Off-peak ($0.10/kWh) — grid supplementation if needed, cheapest time for grid draw\n"
             f"• CO2 saved: ~{_co2_saved:.1f} kg"),
            (f"Energy schedule for {loc_name} on {date}:\n"
             f"• Weather: {avg_clouds:.0f}% clouds, {avg_temp:.0f}°F — expecting {total_kwh:.0f} kWh total, peak {peak_prod:.1f} kW\n"
             f"• Morning (6-10am): Solar ramps up. Let the battery charge first, postpone big loads.\n"
             f"• Midday (10am-2pm): Maximum solar — this is the window for laundry, EV, pool pump.\n"
             f"• Afternoon (2-7pm): Grid peaks at $0.28/kWh. Use battery + remaining solar to avoid grid.\n"
             f"• Evening (7pm+): Transition to battery for base loads, grid fills gaps at mid-peak.\n"
             f"• Carbon impact: {_co2_saved:.1f} kg CO2 avoided by using solar instead of {region} grid."),
            (f"Optimized plan for {date} in {loc_name} ({cap} kW system):\n"
             f"• Conditions: {avg_temp:.0f}°F, {avg_clouds:.0f}% cloud cover — projected {total_kwh:.0f} kWh output\n"
             f"• Pre-dawn: Battery at overnight level. Off-peak grid ($0.10/kWh) if needed.\n"
             f"• 7-10am: Ramp phase — {peak_prod * 0.3:.0f} kW range. Charge battery, light loads only.\n"
             f"• 10am-3pm: Peak output ({peak_prod:.1f} kW). Heavy appliances, EV, battery topping up.\n"
             f"• 3-7pm: Declining solar meets peak rates. Discharge battery to cover the gap.\n"
             f"• Night: Battery → grid handoff. Savings: ~${total_kwh * 0.18:.0f}, CO2: ~{_co2_saved:.1f} kg avoided."),
        ]
        a = _f1_templates[hash(str(date)) % len(_f1_templates)]
        EXAMPLES.append((SYS, q, a))

# F2: Weekly audit (~8)
for loc_name, loc in LOCATIONS.items():
    _loc_meteo = df_meteo[df_meteo["location"] == loc_name]
    # Sample 4 weekly periods
    _weeks = _loc_meteo.groupby(_loc_meteo["time"].dt.isocalendar().week)
    _sampled_weeks = random.sample(list(_weeks.groups.keys()), min(6, len(_weeks.groups)))
    for week_num in _sampled_weeks:
        _week_data = _weeks.get_group(week_num)
        total_kwh = _week_data["prod_kw"].sum()
        avg_eff = _week_data["prod_kw"].mean() / loc["capacity_kw"] * 100
        peak_day = _week_data.groupby(_week_data["time"].dt.date)["prod_kw"].sum().idxmax()
        worst_day = _week_data.groupby(_week_data["time"].dt.date)["prod_kw"].sum().idxmin()
        peak_kwh = _week_data.groupby(_week_data["time"].dt.date)["prod_kw"].sum().max()
        worst_kwh = _week_data.groupby(_week_data["time"].dt.date)["prod_kw"].sum().min()
        region = loc["grid_region"]

        _f2_renew, _f2_co2 = _get_grid_stats(region)
        _f2_co2_avoided = total_kwh * _f2_co2 / 1000
        _f2_savings = total_kwh * 0.18
        _f2_rec = 'Performance on track.' if avg_eff > 10 else 'Below expected — check for panel soiling or shading issues.'
        _f2_q_templates = [
            f"Weekly solar audit for {loc_name}, week {week_num}.",
            f"How did week {week_num} go for {loc_name}?",
            f"Performance summary for {loc_name}, week {week_num}?",
        ]
        q = _f2_q_templates[hash(str(week_num)) % len(_f2_q_templates)]
        _f2_a_templates = [
            (f"Week {week_num} audit for {loc_name}:\n"
             f"• Total generation: {total_kwh:,.0f} kWh\n"
             f"• Average efficiency: {avg_eff:.1f}% of {loc['capacity_kw']} kW capacity\n"
             f"• Best day: {peak_day} ({peak_kwh:.0f} kWh)\n"
             f"• Worst day: {worst_day} ({worst_kwh:.0f} kWh)\n"
             f"• Grid displacement value: ~${_f2_savings:.0f} (avg mid-peak rate)\n"
             f"• CO2 avoided: ~{_f2_co2_avoided:.0f} kg\n"
             f"• Recommendation: {_f2_rec}"),
            (f"Week {week_num} summary — {loc_name}:\n"
             f"Generated {total_kwh:,.0f} kWh at {avg_eff:.1f}% average efficiency. "
             f"Peak day: {peak_day} ({peak_kwh:.0f} kWh), weakest: {worst_day} ({worst_kwh:.0f} kWh). "
             f"Grid savings: ~${_f2_savings:.0f}. Carbon avoided: {_f2_co2_avoided:.0f} kg. {_f2_rec}"),
            (f"{loc_name} week {week_num} report: {total_kwh:,.0f} kWh total output ({avg_eff:.1f}% of {loc['capacity_kw']} kW capacity). "
             f"Range: {worst_kwh:.0f}-{peak_kwh:.0f} kWh/day. "
             f"Economic value: ${_f2_savings:.0f} in avoided grid costs. "
             f"Environmental: {_f2_co2_avoided:.0f} kg CO2 displaced from {region} grid. {_f2_rec}"),
        ]
        a = _f2_a_templates[hash(str(week_num) + loc_name) % len(_f2_a_templates)]
        EXAMPLES.append((SYS, q, a))

# F3: Maintenance scheduling (~8 per location, geographically filtered)
_maintenance_scenarios = [
    ("panel cleaning", "spring", "Schedule cleaning in early spring before peak production season. "
     "Soiling losses of 2-5% accumulate over winter. Morning cleaning when panels are cool is safest.", None),
    ("inverter check", "fall", "Pre-winter inverter inspection recommended. Check for error codes, "
     "fan operation, and connection tightness. Fall maintenance prevents winter failures.", None),
    ("wiring inspection", "summer", "Mid-summer wiring inspection during peak load season. "
     "Check for heat damage, loose connections, and critter interference. Schedule for early morning.", None),
    ("battery health check", "winter", "Winter battery assessment: check cell balance, capacity "
     "test, and thermal management. Cold weather stresses batteries — verify SOC limits are appropriate.", None),
    ("ground fault test", "spring", "Annual ground fault testing in spring. Verify all GFCI breakers "
     "trip correctly and insulation resistance is within spec. Required for NEC compliance.", None),
    ("snow guard inspection", "fall", "Pre-winter snow guard check. Ensure guards are secure and panels "
     "can shed snow safely. Inspect for ice dam formation points around panel edges.", {"MISO"}),
    ("UV degradation check", "summer", "Inspect panels for UV-induced yellowing or delamination. "
     "High-irradiance locations accelerate encapsulant aging. "
     "Check for micro-cracks with thermal imaging during peak sun.", {"CAISO"}),
    ("monitoring system calibration", "summer", "Calibrate production sensors and verify monitoring accuracy. "
     "Compare reported output to utility meter readings. Fix any data gaps in the dashboard.", None),
    ("shade analysis", "winter", "Winter shade assessment: deciduous trees now bare may cast shadows differently. "
     "Check for new construction or vegetation that could shade panels during low sun angles.", None),
]
_f3_q_variants = [
    "When should we schedule {task} for {loc}?",
    "What's the best time for {task} in {loc}?",
    "Plan a {task} window for {loc}.",
    "{task} needed in {loc} — when's optimal?",
]
for loc_name, loc in LOCATIONS.items():
    _f3_idx = 0
    for task, season, advice, climate in _maintenance_scenarios:
        if climate and loc["grid_region"] not in climate:
            continue
        q = _f3_q_variants[_f3_idx % len(_f3_q_variants)].format(task=task, loc=loc_name)
        _loc_seasonal = df_meteo[
            (df_meteo["location"] == loc_name) & (df_meteo["season"] == season)
        ]
        avg_prod = _loc_seasonal["prod_kw"].mean() if len(_loc_seasonal) > 0 else 0
        _timing = ('schedule for early morning or late evening to minimize generation loss.'
                   if avg_prod > 5 else 'low production season, so scheduling is flexible.')
        _f3_a_templates = [
            (f"Maintenance: {task} for {loc_name}. {advice} "
             f"Average {season} production is {avg_prod:.1f} kW — {_timing}"),
            (f"{task.capitalize()} — {loc_name}: best scheduled in {season}. {advice} "
             f"With {avg_prod:.1f} kW average {season} output, {_timing}"),
            (f"Recommended: {task} for {loc_name} during {season}. {advice} "
             f"Production averages {avg_prod:.1f} kW this season, so {_timing}"),
        ]
        a = _f3_a_templates[_f3_idx % len(_f3_a_templates)]
        EXAMPLES.append((SYS, q, a))
        _f3_idx += 1

# F4: Community reports (~6)
for loc_name, loc in LOCATIONS.items():
    _loc_meteo = df_meteo[df_meteo["location"] == loc_name]
    total_annual = _loc_meteo["prod_kw"].sum()
    avg_daily = total_annual / 365
    region = loc["grid_region"]
    _f4_renew, _f4_co2 = _get_grid_stats(region)
    co2_annual = total_annual * _f4_co2 / 1000
    grid_savings = total_annual * 0.18  # avg rate

    _gallons = co2_annual / 8.887
    _trees = co2_annual / 411
    _f4_questions = [
        f"Generate the annual community solar report for {loc_name}.",
        f"What's our annual impact summary for {loc_name}?",
        f"Year-end solar performance report for {loc_name}?",
        f"Summarize {loc_name}'s contribution to our community solar goals.",
        f"How did {loc_name} perform against its design targets this year?",
    ]
    _f4_a_templates = [
        (f"Annual Community Solar Report — {loc_name}:\n"
         f"• System: {loc['capacity_kw']} kW panels + {loc['battery_kwh']} kWh battery\n"
         f"• Total generation: {total_annual:,.0f} kWh\n"
         f"• Average daily: {avg_daily:,.0f} kWh\n"
         f"• Grid cost savings: ~${grid_savings:,.0f}\n"
         f"• CO2 avoided: {co2_annual:,.0f} kg ({co2_annual/1000:.1f} metric tons)\n"
         f"• Equivalent: {_gallons:,.0f} gallons gasoline, {_trees:.0f} trees planted\n"
         f"• Grid region: {region} (avg {_f4_renew:.1f}% renewable)\n"
         f"• Recommendation: Community solar is significantly cleaner than grid average."),
        (f"{loc_name} Annual Solar Summary:\n"
         f"System ({loc['capacity_kw']} kW + {loc['battery_kwh']} kWh battery) produced "
         f"{total_annual:,.0f} kWh this year (~{avg_daily:,.0f} kWh/day). "
         f"Financial impact: ${grid_savings:,.0f} in avoided grid costs. "
         f"Environmental: {co2_annual:,.0f} kg CO2 displaced — equivalent to {_gallons:,.0f} gallons of gas "
         f"or planting {_trees:.0f} trees. Grid: {region} at {_f4_renew:.1f}% renewable. "
         f"Our solar is substantially cleaner than the grid average."),
        (f"Year-end report for {loc_name} ({loc['capacity_kw']} kW array):\n"
         f"• Generation: {total_annual:,.0f} kWh total, {avg_daily:,.0f} kWh daily avg\n"
         f"• Economics: ~${grid_savings:,.0f} saved vs grid ({region}, {_f4_renew:.1f}% renewable)\n"
         f"• Carbon: {co2_annual/1000:.1f} metric tons CO2 avoided\n"
         f"• Equivalents: {_gallons:,.0f} gal gasoline / {_trees:.0f} trees\n"
         f"• Battery: {loc['battery_kwh']} kWh supports overnight and peak shaving\n"
         f"Community solar continues to outperform grid on both cost and carbon."),
    ]
    for idx, q in enumerate(_f4_questions):
        a = _f4_a_templates[idx % len(_f4_a_templates)]
        EXAMPLES.append((SYS, q, a))

# F5: Cross-location comparisons (~3, aspect-focused)
_loc_names = list(LOCATIONS.keys())
if len(_loc_names) >= 2:
    _loc_a, _loc_b = _loc_names[0], _loc_names[1]
    _a_meteo = df_meteo[df_meteo["location"] == _loc_a]
    _b_meteo = df_meteo[df_meteo["location"] == _loc_b]
    _a_total = _a_meteo["prod_kw"].sum()
    _b_total = _b_meteo["prod_kw"].sum()
    _a_cap = LOCATIONS[_loc_a]["capacity_kw"]
    _b_cap = LOCATIONS[_loc_b]["capacity_kw"]
    _a_cf = _a_total / (_a_cap * len(_a_meteo)) * 100 if len(_a_meteo) > 0 else 0
    _b_cf = _b_total / (_b_cap * len(_b_meteo)) * 100 if len(_b_meteo) > 0 else 0
    _a_renew, _a_co2 = _get_grid_stats(LOCATIONS[_loc_a]["grid_region"])
    _b_renew, _b_co2 = _get_grid_stats(LOCATIONS[_loc_b]["grid_region"])
    _a_co2_saved = _a_total * _a_co2 / 1000
    _b_co2_saved = _b_total * _b_co2 / 1000
    _better_cf = _loc_b if _b_cf > _a_cf else _loc_a
    _better_co2 = _loc_a if _a_co2_saved > _b_co2_saved else _loc_b

    # 3 questions, each with a unique answer angle
    _f5_pairs = [
        (f"Compare solar performance between {_loc_a} and {_loc_b}.",
         f"Generation comparison:\n"
         f"• {_loc_a}: {_a_total:,.0f} kWh/year from {_a_cap} kW array "
         f"(capacity factor {_a_cf:.1f}%)\n"
         f"• {_loc_b}: {_b_total:,.0f} kWh/year from {_b_cap} kW array "
         f"(capacity factor {_b_cf:.1f}%)\n"
         f"{_better_cf} has the higher capacity factor due to better solar resource."),

        (f"Which location has more carbon impact — {_loc_a} or {_loc_b}?",
         f"Carbon displacement comparison:\n"
         f"• {_loc_a} ({LOCATIONS[_loc_a]['grid_region']}, {_a_renew:.1f}% renewable): "
         f"avoids {_a_co2_saved:,.0f} kg CO2/year at {_a_co2:.0f} kg/MWh grid intensity\n"
         f"• {_loc_b} ({LOCATIONS[_loc_b]['grid_region']}, {_b_renew:.1f}% renewable): "
         f"avoids {_b_co2_saved:,.0f} kg CO2/year at {_b_co2:.0f} kg/MWh grid intensity\n"
         f"{_better_co2} delivers more total carbon value."),

        (f"Which location should we expand first — {_loc_a} or {_loc_b}?",
         f"Expansion analysis:\n"
         f"• {_loc_a}: {_a_cf:.1f}% capacity factor, "
         f"{LOCATIONS[_loc_a]['grid_region']} grid ({_a_co2:.0f} kg/MWh)\n"
         f"• {_loc_b}: {_b_cf:.1f}% capacity factor, "
         f"{LOCATIONS[_loc_b]['grid_region']} grid ({_b_co2:.0f} kg/MWh)\n"
         f"For maximum energy yield, expand {_better_cf}. "
         f"For maximum carbon impact, expand {_better_co2}."),
    ]
    for q, a in _f5_pairs:
        EXAMPLES.append((SYS, q, a))

_cat_f_count = len(EXAMPLES) - _cat_f_start
print(f"    Category F: {_cat_f_count} examples")

# ---------- Category G: Tool-Calling Examples (~50) ---------------------------
# Teaches the model WHEN and HOW to call API tools vs answer from knowledge.
# Uses real API data already fetched (OWM, Open-Meteo, EIA).
# Format: message lists matching finetune TOOL_CALL_DATA structure.
print("  Generating Category G: Tool-Calling...")

TOOL_CALL_EXAMPLES = []

def _tc(user_q, calls, results, answer):
    """Build a tool-calling conversation example."""
    msgs = [{"role": "system", "content": SYS_TOOLS},
            {"role": "user", "content": user_q},
            {"role": "assistant", "tool_calls": calls}]
    for name, content in results:
        msgs.append({"role": "tool", "name": name, "content": content})
    msgs.append({"role": "assistant", "content": answer})
    return msgs

def _fn(name, args=None):
    """Build a function call dict."""
    return {"function": {"name": name, "arguments": args or {}}}

# G1: get_weather calls (~5 per location)
_g1_questions = [
    "What's the weather like in {loc} right now?",
    "Current weather conditions?",
    "Temperature and cloud check for {loc}.",
    "Give me a weather update.",
    "Is it sunny or cloudy right now in {loc}?",
]
for loc_name in LOCATIONS:
    _wr = df_owm[df_owm["location"] == loc_name]
    if len(_wr) == 0:
        continue
    _wr = _wr.iloc[0]
    _temp = round(float(_wr["temp_f"]), 1)
    _cld = int(_wr["clouds_pct"])
    _desc = str(_wr["description"])
    _wind = round(float(_wr["wind_mph"]), 1)
    _hum = int(_wr["humidity_pct"])
    _rise = str(_wr["sunrise"])
    _sset = str(_wr["sunset"])
    _wx_json = json.dumps({"temperature_f": _temp, "clouds_pct": _cld,
                            "description": _desc, "wind_mph": _wind,
                            "humidity_pct": _hum, "sunrise": _rise,
                            "sunset": _sset})
    _solar_ok = "Good solar conditions." if _cld < 40 else "Cloud cover is reducing solar output."
    _g1_answers = [
        f"Current conditions in {loc_name}: {_temp:.0f}°F with {_desc} "
        f"({_cld}% clouds). Wind at {_wind} mph, humidity {_hum}%. "
        f"Sun window: {_rise}–{_sset}. {_solar_ok}",
        f"Weather in {loc_name}: {_temp:.0f}°F, {_desc}. "
        f"{_cld}% cloud cover, {_wind} mph wind. Sunrise {_rise}, sunset {_sset}. {_solar_ok}",
        f"It's {_temp:.0f}°F with {_desc} in {loc_name}. "
        f"Clouds: {_cld}%, wind: {_wind} mph, humidity: {_hum}%. Daylight: {_rise}–{_sset}. {_solar_ok}",
    ]
    for qi, qt in enumerate(_g1_questions):
        TOOL_CALL_EXAMPLES.append(_tc(
            qt.format(loc=loc_name),
            [_fn("get_weather", {"location": loc_name})],
            [("get_weather", _wx_json)],
            _g1_answers[qi % len(_g1_answers)]))

# G2: get_solar_production calls (~5 per location)
_g2_questions = [
    "How much solar are we generating right now?",
    "What's our current solar output in {loc}?",
    "Are the panels producing well today?",
    "Live solar production check.",
    "How much power are we getting from the panels in {loc}?",
]
for loc_name, loc in LOCATIONS.items():
    _cap = loc["capacity_kw"]
    _sol_rows = df_meteo[
        (df_meteo["ghi"] > 50) & (df_meteo["location"] == loc_name)
    ].sample(5, random_state=456)
    for qi, (_, sr) in enumerate(_sol_rows.iterrows()):
        _cld = int(sr["cloudcover"])
        _tmp = round(float(sr["temp_f"]), 1)
        _prod = round(float(sr["prod_kw"]), 1)
        _eff = round(_prod / _cap * 100, 1)
        _ghi = round(float(sr["ghi"]), 1)
        _dr = round(float(sr["temp_derate"]) * 100, 1)
        _s_json = json.dumps({"production_kw": _prod, "capacity_kw": _cap,
                               "efficiency_pct": _eff, "ghi_wm2": _ghi,
                               "temp_derate_pct": _dr, "source": "open-meteo"})
        _prod_note = ("Strong production — great time for heavy loads."
                      if _eff > 50 else "Lower output — consider deferring heavy loads.")
        _g2_answers = [
            f"Solar output in {loc_name}: {_prod} kW from {_cap} kW capacity ({_eff}%). "
            f"GHI is {_ghi} W/m², clouds {_cld}%. Thermal derating: {_dr}% at {_tmp}°F. {_prod_note}",
            f"Currently generating {_prod} kW ({_eff}% of {_cap} kW). "
            f"Irradiance at {_ghi} W/m² with {_cld}% clouds. {_prod_note}",
            f"The array in {loc_name} is producing {_prod} kW. "
            f"GHI: {_ghi} W/m², efficiency: {_eff}%, derating: {100-_dr:.1f}% at {_tmp}°F. {_prod_note}",
        ]
        TOOL_CALL_EXAMPLES.append(_tc(
            _g2_questions[qi % len(_g2_questions)].format(loc=loc_name),
            [_fn("get_solar_production", {"clouds_pct": _cld, "temp_f": _tmp})],
            [("get_solar_production", _s_json)],
            _g2_answers[qi % len(_g2_answers)]))

# G3: get_battery_state calls (~4 per location)
_g3_questions = [
    "What's the battery level?",
    "How much charge do we have stored?",
    "Battery status report.",
    "Is the battery charging or discharging?",
]
_g3_socs = [15, 40, 65, 85]
for loc_name in LOCATIONS:
    for qi, soc in enumerate(_g3_socs):
        _charging = soc < 50
        _bat_cap = LOCATIONS[loc_name]["battery_kwh"]
        _kwh_stored = round(soc / 100 * _bat_cap)
        _b_json = json.dumps({"soc_pct": soc, "kwh_stored": _kwh_stored,
                               "capacity_kwh": _bat_cap, "charging": _charging})
        if soc > 70:
            _b_note = (f"Battery is well charged at {soc}% ({_kwh_stored} kWh of {_bat_cap} kWh). "
                       f"Safe to draw from reserves or export surplus to the grid.")
        elif soc > 35:
            _b_note = (f"Battery at {soc}% ({_kwh_stored} kWh). Moderate reserves — "
                       f"good for normal operations. {'Currently charging.' if _charging else 'Holding steady.'}")
        else:
            _b_note = (f"Battery is low at {soc}% ({_kwh_stored} kWh). "
                       f"{'Charging from solar.' if _charging else 'Consider preserving for peak hours.'} "
                       f"Avoid heavy draws until reserves recover.")
        TOOL_CALL_EXAMPLES.append(_tc(
            _g3_questions[qi], [_fn("get_battery_state")],
            [("get_battery_state", _b_json)], _b_note))

# G4: get_grid_status calls (~4 per location)
_g4_questions = [
    "What are the current grid rates?",
    "Are we in peak pricing right now?",
    "What period is it for electricity pricing?",
    "How much does grid power cost right now?",
]
_g4_hours = [3, 10, 15, 21]
for loc_name in LOCATIONS:
    _g4_region = LOCATIONS[loc_name]["grid_region"]
    _g4_renew, _g4_co2 = _get_grid_stats(_g4_region)
    for qi, hr in enumerate(_g4_hours):
        _period, _rate = _grid_period(hr)
        _gr_json = json.dumps({"period": _period, "rate_per_kwh": _rate,
                                "renewable_pct": _g4_renew,
                                "co2_intensity": _g4_co2})
        if _period == "peak":
            _gr_note = (f"Peak pricing: ${_rate:.2f}/kWh (2–7pm). "
                        f"Grid is {_g4_renew:.0f}% renewable ({_g4_co2:.0f} g CO₂/kWh). "
                        f"Minimize grid draws — use solar and battery instead.")
        elif _period == "mid-peak":
            _gr_note = (f"Mid-peak pricing: ${_rate:.2f}/kWh. "
                        f"Grid is {_g4_renew:.0f}% renewable. "
                        f"Moderate rates — normal usage is fine, but heavy loads are cheaper during off-peak.")
        else:
            _gr_note = (f"Off-peak pricing: ${_rate:.2f}/kWh. "
                        f"Grid is {_g4_renew:.0f}% renewable ({_g4_co2:.0f} g CO₂/kWh). "
                        f"Cheapest rates — good time to charge the battery from grid if solar is low.")
        TOOL_CALL_EXAMPLES.append(_tc(
            _g4_questions[qi], [_fn("get_grid_status")],
            [("get_grid_status", _gr_json)], _gr_note))

# G5: Multi-tool chains (~5 per location)
_g5_scenarios = [
    ("Should I run the dishwasher and laundry now?", ["weather", "solar"]),
    ("Complete energy status report, please.", ["weather", "solar", "battery", "grid"]),
    ("Is now a good time to charge the EV?", ["solar", "grid"]),
    ("Should we sell power to the grid or store it?", ["solar", "battery", "grid"]),
    ("What's our overall energy situation right now?", ["weather", "solar", "battery"]),
]
for loc_name, loc in LOCATIONS.items():
    _cap = loc["capacity_kw"]
    _wr = df_owm[df_owm["location"] == loc_name]
    if len(_wr) == 0:
        continue
    _wr = _wr.iloc[0]
    _cld = int(_wr["clouds_pct"])
    _tmp = round(float(_wr["temp_f"]), 1)
    _desc = str(_wr["description"])
    _wind = round(float(_wr["wind_mph"]), 1)
    _hum = int(_wr["humidity_pct"])
    _g5_rise = str(_wr["sunrise"])
    _g5_sset = str(_wr["sunset"])

    # Sample a midday row for realistic production
    _mid = df_meteo[(df_meteo["location"] == loc_name) & (df_meteo["hour"] == 12)]
    _mid = _mid.sample(1, random_state=789).iloc[0] if len(_mid) > 0 else \
           df_meteo[df_meteo["location"] == loc_name].sample(1, random_state=789).iloc[0]
    _prod = round(float(_mid["prod_kw"]), 1)
    _eff = round(_prod / _cap * 100, 1)
    _ghi = round(float(_mid["ghi"]), 1)
    _dr = round(float(_mid["temp_derate"]) * 100, 1)

    _wx_j = json.dumps({"temperature_f": _tmp, "clouds_pct": _cld,
                          "description": _desc, "wind_mph": _wind, "humidity_pct": _hum,
                          "sunrise": _g5_rise, "sunset": _g5_sset})
    _sol_j = json.dumps({"production_kw": _prod, "capacity_kw": _cap,
                           "efficiency_pct": _eff, "ghi_wm2": _ghi,
                           "temp_derate_pct": _dr, "source": "open-meteo"})

    for q_text, tools_needed in _g5_scenarios:
        _calls = []
        _results = []
        if "weather" in tools_needed:
            _calls.append(_fn("get_weather", {"location": loc_name}))
            _results.append(("get_weather", _wx_j))
        if "solar" in tools_needed:
            _calls.append(_fn("get_solar_production", {"clouds_pct": _cld, "temp_f": _tmp}))
            _results.append(("get_solar_production", _sol_j))
        if "battery" in tools_needed:
            _soc = random.choice([25, 50, 75, 90])
            _g5_bat_cap = loc["battery_kwh"]
            _g5_kwh = round(_soc / 100 * _g5_bat_cap)
            _b_j = json.dumps({"soc_pct": _soc, "kwh_stored": _g5_kwh,
                                "capacity_kwh": _g5_bat_cap, "charging": _soc < 50})
            _calls.append(_fn("get_battery_state"))
            _results.append(("get_battery_state", _b_j))
        else:
            _soc = 50
        if "grid" in tools_needed:
            _period, _rate = _grid_period(12)  # midday
            _g5_renew, _g5_co2 = _get_grid_stats(loc["grid_region"])
            _gr_j = json.dumps({"period": _period, "rate_per_kwh": _rate,
                                 "renewable_pct": _g5_renew, "co2_intensity": _g5_co2})
            _calls.append(_fn("get_grid_status"))
            _results.append(("get_grid_status", _gr_j))
        else:
            _period, _rate = "mid-peak", 0.18

        # Build contextual answer based on scenario
        if "dishwasher" in q_text or "laundry" in q_text:
            a = (f"Let me check conditions in {loc_name}. It's {_tmp}°F with {_desc}, "
                 f"and we're producing {_prod} kW ({_eff}% of {_cap} kW). "
                 f"{'Yes — plenty of solar surplus to run both appliances. Go ahead!' if _prod > 20 else 'Production is limited. Run one at a time, or wait for better conditions.'}")
        elif "Complete" in q_text or "full picture" in q_text.lower():
            a = (f"Full status for {loc_name}:\n"
                 f"• Weather: {_tmp}°F, {_desc} ({_cld}% clouds)\n"
                 f"• Solar: {_prod} kW / {_cap} kW ({_eff}% efficiency)\n"
                 f"• Battery: {_soc}% ({_soc} kWh of 100 kWh)\n"
                 f"• Grid: {_period} at ${_rate:.2f}/kWh\n"
                 f"{'System is performing well. Good time to run loads on solar.' if _eff > 50 else 'Production is low — rely on battery and minimize grid draws.'}")
        elif "EV" in q_text:
            a = (f"Solar is at {_prod} kW ({_eff}%) and grid is {_period} (${_rate:.2f}/kWh). "
                 f"{'Yes — solar surplus can handle EV charging (~7 kW) without grid draw.' if _prod > 25 else 'EV charging would draw from the grid. Wait for better solar or off-peak rates ($0.10/kWh) to save money.'}")
        elif "sell" in q_text or "store" in q_text:
            a = (f"Production: {_prod} kW. Battery at {_soc}%. Grid: {_period} ${_rate:.2f}/kWh. "
                 f"{'Battery is low — store surplus to build reserves before considering export.' if _soc < 50 else 'Battery is healthy. At ' + _period + ' rates, ' + ('export surplus for $' + f'{_rate:.2f}/kWh revenue.' if _rate > 0.15 else 'store more — export value is low at off-peak rates.')}")
        else:
            a = (f"Energy situation in {loc_name}: {_tmp}°F, {_desc}. "
                 f"Producing {_prod} kW ({_eff}%). Battery at {_soc}%. "
                 f"{'Overall healthy — solar covers demand with surplus for battery.' if _prod > 20 and _soc > 40 else 'Running on reduced capacity — manage loads carefully.'}")
        TOOL_CALL_EXAMPLES.append(_tc(q_text, _calls, _results, a))

# G6: Direct answers — no tools needed (teaches when NOT to call tools)
_g6_pairs = [
    ("What's the typical winter production for Ann Arbor?",
     "Based on historical data, Ann Arbor averages about 15-25 kW during winter midday hours "
     "due to shorter days and lower sun angles. Summer production is roughly 2-3x higher."),
    ("How does MISO compare to CAISO for renewables?",
     "MISO (Midwest) runs about 30-35% renewable, heavily reliant on wind. "
     "CAISO (California) averages 65-70% renewable with strong solar penetration. "
     "Our Ann Arbor panels displace more CO2 per kWh because MISO's grid is dirtier."),
    ("What temperature causes the most panel efficiency loss?",
     "Silicon panels lose ~0.4% efficiency per °F above 77°F. At 100°F, that's about 9% loss. "
     "Below 77°F, panels actually gain efficiency. Michigan winters can boost output slightly, "
     "but lower sun angles and shorter days more than offset the thermal advantage."),
    ("How many homes does our community solar system serve?",
     "Our SolarHive community has 12 homes sharing a 72 kW panel array and 100 kWh battery "
     "in Ann Arbor. Each home averages about 6 kW of panel capacity and 8.3 kWh of battery storage."),
]
for q, a in _g6_pairs:
    TOOL_CALL_EXAMPLES.append([
        {"role": "system", "content": SYS},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ])

_cat_g_count = len(TOOL_CALL_EXAMPLES)
print(f"    Category G: {_cat_g_count} tool-calling examples")

# Summary
print(f"\n  Total synthetic examples: {len(EXAMPLES)} Q&A + {_cat_g_count} tool-calling")
print(f"    A: Hourly Production      {_cat_a_count}")
print(f"    B: PVWatts Ground Truth    {_cat_b_count}")
print(f"    C: Grid Mix & Carbon       {_cat_c_count}")
print(f"    D: Weather Impact          {_cat_d_count}")
print(f"    E: Battery + Grid Strategy {_cat_e_count}")
print(f"    F: Multi-Step Reasoning    {_cat_f_count}")
print(f"    G: Tool-Calling            {_cat_g_count}")
print("✅ Cell 7 complete — synthetic examples generated")

"""## 8: Export"""

# === CELL 8: Export ===========================================================
# Save to JSON (Google Drive + local backup). Deduplication before saving.

# Strip trailing whitespace from answers
EXAMPLES = [(s, q, a.rstrip()) for s, q, a in EXAMPLES]

# Deduplication: hash (question, answer) pairs
_seen = set()
_deduped = []
for ex in EXAMPLES:
    key = (ex[1].strip(), ex[2].strip())
    if key not in _seen:
        _seen.add(key)
        _deduped.append(ex)
_dup_count = len(EXAMPLES) - len(_deduped)
if _dup_count > 0:
    print(f"  Removed {_dup_count} duplicate examples")
EXAMPLES = _deduped

# Estimate tokens (rough: 1 token ≈ 4 chars)
_total_chars = sum(len(s) + len(q) + len(a) for s, q, a in EXAMPLES)
_tc_chars = sum(len(json.dumps(ex)) for ex in TOOL_CALL_EXAMPLES)
_est_tokens = (_total_chars + _tc_chars) // 4

# Build export payload
_export = {
    "metadata": {
        "generator": "solarhive_datagen.py",
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "total_examples": len(EXAMPLES),
        "tool_call_examples": len(TOOL_CALL_EXAMPLES),
        "total_generated": _cat_a_count + _cat_b_count + _cat_c_count + _cat_d_count + _cat_e_count + _cat_f_count,
        "duplicates_removed": _dup_count,
        "estimated_tokens": _est_tokens,
        "categories": {
            "A_hourly_production": _cat_a_count,
            "B_pvwatts_ground_truth": _cat_b_count,
            "C_grid_mix_carbon": _cat_c_count,
            "D_weather_impact": _cat_d_count,
            "E_battery_grid_strategy": _cat_e_count,
            "F_multi_step_reasoning": _cat_f_count,
            "G_tool_calling": _cat_g_count,
        },
        "locations": list(LOCATIONS.keys()),
        "data_sources": ["Open-Meteo", "NREL PVWatts v8", "OpenWeatherMap", "EIA v2"],
    },
    "qa_data": [list(ex) for ex in EXAMPLES],
    "tool_call_data": TOOL_CALL_EXAMPLES,
}

# Save to Google Drive
if DRIVE_DIR:
    _drive_path = os.path.join(DRIVE_DIR, "datagen_latest.json")
    with open(_drive_path, "w", encoding="utf-8") as f:
        json.dump(_export, f, indent=1, ensure_ascii=False)
    print(f"  Saved to Google Drive: {_drive_path}")

# Local backup
_local_path = "datagen_latest.json"
with open(_local_path, "w", encoding="utf-8") as f:
    json.dump(_export, f, indent=1, ensure_ascii=False)
print(f"  Saved locally: {_local_path}")

# Final report
print(f"\n{'=' * 60}")
print(f"EXPORT COMPLETE")
print(f"{'=' * 60}")
print(f"  Examples:  {len(EXAMPLES)}")
print(f"  Tokens:    ~{_est_tokens:,}")
print(f"  Duplicates removed: {_dup_count}")
print(f"  Format:    (system, question, answer) 3-tuples")
print(f"  Category breakdown:")
print(f"    A: Hourly Production      {_cat_a_count:>4}")
print(f"    B: PVWatts Ground Truth    {_cat_b_count:>4}")
print(f"    C: Grid Mix & Carbon       {_cat_c_count:>4}")
print(f"    D: Weather Impact          {_cat_d_count:>4}")
print(f"    E: Battery + Grid Strategy {_cat_e_count:>4}")
print(f"    F: Multi-Step Reasoning    {_cat_f_count:>4}")
print(f"    G: Tool-Calling            {_cat_g_count:>4}")
print(f"\n✅ Cell 8 complete — datagen export ready for solarhive_finetune.py")
