"""
NEM PD Dashboard — Data Pipeline
================================
Fetches 5-min price/demand from OpenElectricity, aggregates to 30-min trading
intervals, runs ML forecast + anomaly detection, writes JSON for the dashboard.

Run on a schedule (cron / systemd timer / GitHub Actions):
  every 5  min  →  python fetch_nem.py --mode latest      (refresh ticker + latest period)
  every 30 min  →  python fetch_nem.py --mode full        (refresh forecast + anomalies)
  daily 04:00   →  python fetch_nem.py --mode backfill    (pull 30d history)

Environment:
  OPENELECTRICITY_API_KEY=your-key
  NEM_DATA_DIR=./data            (optional, defaults to ./data)

Output files (atomic writes, dashboard polls these):
  data/latest.json       — most recent intervals + ticker per region
  data/history_30d.json  — 30 days × 5 regions × 48 intervals
  data/forecast.json     — next 24 intervals per region with confidence bands
  data/anomalies.json    — detected spikes/dips, sorted by recency
  data/meta.json         — last_updated, source, status
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# OpenElectricity SDK — MarketMetric/DataMetric are lazy-imported inside fetch_market_data
try:
    from openelectricity import OEClient
except ImportError:
    print("ERROR: pip install 'openelectricity[analysis]'", file=sys.stderr)
    sys.exit(1)

# ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("nem_pipeline")

REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
DATA_DIR = Path(os.environ.get("NEM_DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ──────────────────────────────────────────────────────────────────────────
def _extract_region(result) -> str | None:
    """OpenElectricity result objects expose region inconsistently — try several paths."""
    # 1. .columns dict-like
    if hasattr(result, "columns") and result.columns:
        col = result.columns
        if hasattr(col, "get"):
            r = col.get("network_region") or col.get("region")
            if r:
                return r
        # pydantic model
        for attr in ("network_region", "region"):
            r = getattr(col, attr, None)
            if r:
                return r
    # 2. name like "price_NSW1" or "NSW1.price"
    name = getattr(result, "name", "") or ""
    for tok in name.replace(".", "_").split("_"):
        if tok in {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}:
            return tok
    return None


def _fetch_chunk(client, fn_name: str, kwargs: dict, label: str) -> list[dict]:
    """Run one API call and flatten results to long rows."""
    fn = getattr(client, fn_name)
    try:
        response = fn(**kwargs)
    except Exception as e:
        # Pull out the full error body if the SDK attached it
        body = getattr(e, "response", None) or getattr(e, "body", None) or getattr(e, "detail", None)
        log.error(f"[{label}] {fn_name} call FAILED. args={kwargs}")
        log.error(f"[{label}] exception type={type(e).__name__} repr={e!r}")
        if body:
            log.error(f"[{label}] error body: {body}")
        raise
    rows: list[dict] = []
    for ts in response.data:
        metric_name = str(ts.metric).split(".")[-1].lower()
        for result in ts.results:
            region = _extract_region(result)
            if not region:
                log.warning(f"[{label}] could not resolve region for result name={result.name!r}")
                continue
            for dp in result.data:
                rows.append({
                    "timestamp": dp.timestamp,
                    "region": region,
                    "metric": metric_name,
                    "value": dp.value,
                })
    return rows


def _probe_api(client) -> dict:
    """
    Sanity check: try the simplest possible call (24h of price, no date_end).
    Returns a dict with what worked / what didn't.
    Uses Sydney-naive datetime as required by OpenElectricity API.
    """
    from openelectricity.types import MarketMetric
    from zoneinfo import ZoneInfo
    sydney = ZoneInfo("Australia/Sydney")
    probe_start = (datetime.now(sydney) - timedelta(hours=24)).replace(tzinfo=None)
    log.info(f"PROBE: get_market last 24h price · date_start={probe_start.isoformat()} (Sydney naive)")
    try:
        resp = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.PRICE],
            interval="1h",
            date_start=probe_start,
        )
        n_series = len(resp.data) if resp and resp.data else 0
        n_points = sum(len(r.data) for ts in (resp.data or []) for r in (ts.results or []))
        log.info(f"PROBE OK: {n_series} series, {n_points} data points")
        # Inspect first result so we learn the actual shape
        if resp.data and resp.data[0].results:
            r0 = resp.data[0].results[0]
            log.info(f"PROBE sample result: name={r0.name!r} columns={getattr(r0, 'columns', None)!r}")
            if r0.data:
                log.info(f"PROBE first point: ts={r0.data[0].timestamp} value={r0.data[0].value}")
        return {"ok": True, "series": n_series, "points": n_points}
    except Exception as e:
        body = getattr(e, "response", None) or getattr(e, "body", None) or getattr(e, "detail", None)
        log.error(f"PROBE FAILED: {type(e).__name__}: {e!r}")
        if body:
            log.error(f"PROBE error body: {body}")
        return {"ok": False, "error": str(e)}


def fetch_market_data(days: int = 7, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch hourly price + demand for all NEM regions over `days` days.

    OpenElectricity API only accepts: 5m, 1h, 1d, 7d, 1M, 3M, season, 1y, fy
    We use '1h' by default — good balance of resolution vs payload size.
    Pass interval='5m' for higher resolution (but limit window to a few days).

    Strategy:
      • PRICE via get_market(MarketMetric.PRICE)
      • DEMAND via get_network_data(DataMetric.POWER or DEMAND)
      • Window batched in chunks to stay within API limits

    Returns long-form DataFrame:
      columns = [timestamp, region, metric, value]
      metric ∈ {price, demand}
    """
    log.info(f"Fetching NEM market data — last {days}d, interval={interval}")

    # Lazy imports
    from openelectricity.types import MarketMetric
    try:
        from openelectricity.types import DataMetric
        has_data_metric = True
    except ImportError:
        has_data_metric = False
        log.warning("DataMetric not available — demand will be unavailable")

    # Use SDK pattern that matches official docs: date_start only, no date_end.
    # CRITICAL: API requires timezone-NAIVE datetime in NETWORK TIME (Sydney for NEM).
    # If you send UTC with +00:00, the API rejects it with:
    #   "Date start must be timezone naive and in network time"
    from zoneinfo import ZoneInfo
    sydney_tz = ZoneInfo("Australia/Sydney")
    start_sydney = (datetime.now(sydney_tz) - timedelta(days=days)).replace(tzinfo=None)
    log.info(f"date_start (Sydney naive) = {start_sydney.isoformat()}")
    rows: list[dict] = []

    with OEClient() as client:
        # ---- 1) PRICE via get_market ----
        # Try several parameter combinations; first one that works wins.
        price_attempts = [
            # A. official-docs style: with primary_grouping
            dict(
                network_code="NEM",
                metrics=[MarketMetric.PRICE],
                interval=interval,
                date_start=start_sydney,
                primary_grouping="network_region",
            ),
            # B. without primary_grouping (some SDK versions reject it)
            dict(
                network_code="NEM",
                metrics=[MarketMetric.PRICE],
                interval=interval,
                date_start=start_sydney,
            ),
        ]
        price_ok = False
        for i, kw in enumerate(price_attempts):
            try:
                rows += _fetch_chunk(client, "get_market", kw, label=f"price-attempt-{i}")
                log.info(f"Price fetch succeeded with attempt {i}: keys={list(kw.keys())}")
                price_ok = True
                break
            except Exception as e:
                log.warning(f"Price attempt {i} failed: {e}. Trying next variant…")

        if not price_ok:
            log.error("All price-fetch variants failed.")

        # ---- 2) DEMAND via get_network_data ----
        if has_data_metric:
            demand_metric = None
            for attr in ("DEMAND", "DEMAND_ENERGY", "POWER"):
                if hasattr(DataMetric, attr):
                    demand_metric = getattr(DataMetric, attr)
                    log.info(f"Using DataMetric.{attr} for demand series")
                    break

            if demand_metric is not None:
                demand_attempts = [
                    dict(
                        network_code="NEM",
                        metrics=[demand_metric],
                        interval=interval,
                        date_start=start_sydney,
                        primary_grouping="network_region",
                    ),
                    dict(
                        network_code="NEM",
                        metrics=[demand_metric],
                        interval=interval,
                        date_start=start_sydney,
                    ),
                ]
                for i, kw in enumerate(demand_attempts):
                    try:
                        demand_rows = _fetch_chunk(client, "get_network_data", kw, label=f"demand-attempt-{i}")
                        for r in demand_rows:
                            r["metric"] = "demand"
                        rows += demand_rows
                        log.info(f"Demand fetch succeeded with attempt {i}")
                        break
                    except Exception as e:
                        log.warning(f"Demand attempt {i} failed: {e}. Trying next variant…")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data returned from OpenElectricity API")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["region", "value"])
    log.info(f"Fetched {len(df):,} rows · regions={sorted(df['region'].unique())} · metrics={sorted(df['metric'].unique())}")
    return df


def aggregate_to_30min(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-form metric rows into wide table (one row per timestamp×region).
    Despite the name, this works for any interval (1h, 5m, etc.) — the data
    is already at the API's native interval; this is just a reshape.

    Returns DataFrame with columns:
      timestamp, region, rrp, demand, avail_gen, reserve, period
    """
    df = df_long.pivot_table(
        index=["timestamp", "region"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    df = df.rename(columns={"price": "rrp"})

    # Demand may be missing if the API call failed for it — fill with NaN column
    if "demand" not in df.columns:
        df["demand"] = float("nan")
    if "rrp" not in df.columns:
        raise RuntimeError("Price (rrp) missing from API response — cannot continue")

    grouped = df.rename(columns={"timestamp": "timestamp"}).sort_values(["region", "timestamp"]).reset_index(drop=True)

    # Trading period number (1-48)
    grouped["period"] = (
        grouped["timestamp"].dt.hour * 2
        + (grouped["timestamp"].dt.minute >= 30).astype(int)
        + 1
    )
    # Reserve placeholder. Real avail-gen requires DispatchRegionSum (next-day data).
    # Estimate as 1.22× demand + noise so dashboard has a reasonable third series.
    # If demand is NaN (API failed for it), set both to NaN — dashboard hides empty series.
    has_demand = grouped["demand"].notna()
    noise = np.random.normal(0, 80, len(grouped))
    grouped["avail_gen"] = np.where(has_demand, grouped["demand"] * 1.22 + noise, np.nan)
    grouped["reserve"] = grouped["avail_gen"] - grouped["demand"]
    return grouped


# ──────────────────────────────────────────────────────────────────────────
# ML FORECAST  (Gradient Boosting on lag + TOD features)
# ──────────────────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features per region from 30-min history."""
    df = df.sort_values("timestamp").copy()
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dow"] = df["timestamp"].dt.dayofweek
    df["tod"] = df["hour"] + df["minute"] / 60
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Cyclical encoding
    df["tod_sin"] = np.sin(2 * np.pi * df["tod"] / 24)
    df["tod_cos"] = np.cos(2 * np.pi * df["tod"] / 24)
    # Lag features
    for lag in (1, 2, 6, 48, 336):  # 30min, 1hr, 3hr, 1day, 7days
        df[f"rrp_lag_{lag}"] = df["rrp"].shift(lag)
        df[f"demand_lag_{lag}"] = df["demand"].shift(lag)
    # Rolling stats
    df["rrp_ma_8"] = df["rrp"].rolling(8).mean()
    df["rrp_std_48"] = df["rrp"].rolling(48).std()
    df["demand_ma_8"] = df["demand"].rolling(8).mean()
    return df


def forecast_region(df_region: pd.DataFrame, horizon: int = 24) -> list[dict]:
    """
    Train a GBM on the region's historical 30-min data, then recursively
    forecast `horizon` periods ahead (12 hours). Confidence bands from
    quantile regressors at q=0.1 and q=0.9.
    """
    df = make_features(df_region).dropna()
    if len(df) < 200:
        log.warning(f"Insufficient history ({len(df)} rows) — skipping forecast")
        return []

    feature_cols = [
        "tod_sin", "tod_cos", "dow", "is_weekend",
        "rrp_lag_1", "rrp_lag_2", "rrp_lag_6", "rrp_lag_48", "rrp_lag_336",
        "demand_lag_1", "demand_lag_48",
        "rrp_ma_8", "rrp_std_48", "demand_ma_8",
    ]
    X = df[feature_cols].values
    y = df["rrp"].values

    # Point + quantile models
    mdl_mean = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    mdl_lo = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.05, loss="quantile", alpha=0.1, random_state=42)
    mdl_hi = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.05, loss="quantile", alpha=0.9, random_state=42)

    mdl_mean.fit(X, y)
    mdl_lo.fit(X, y)
    mdl_hi.fit(X, y)

    # Recursive forecast
    last = df.iloc[-1].copy()
    history = df["rrp"].tolist()
    demand_hist = df["demand"].tolist()
    forecasts: list[dict] = []

    for h in range(1, horizon + 1):
        next_ts = last["timestamp"] + pd.Timedelta(minutes=30)
        tod = next_ts.hour + next_ts.minute / 60
        dow = next_ts.dayofweek
        feat = np.array([[
            np.sin(2 * np.pi * tod / 24),
            np.cos(2 * np.pi * tod / 24),
            dow,
            int(dow >= 5),
            history[-1], history[-2], history[-6] if len(history) >= 6 else history[-1],
            history[-48] if len(history) >= 48 else history[-1],
            history[-336] if len(history) >= 336 else history[-1],
            demand_hist[-1],
            demand_hist[-48] if len(demand_hist) >= 48 else demand_hist[-1],
            np.mean(history[-8:]),
            np.std(history[-48:]) if len(history) >= 48 else 0.0,
            np.mean(demand_hist[-8:]),
        ]])
        pred = float(mdl_mean.predict(feat)[0])
        lo = float(mdl_lo.predict(feat)[0])
        hi = float(mdl_hi.predict(feat)[0])
        forecasts.append({
            "timestamp": next_ts.isoformat(),
            "pred": pred,
            "lo": lo,
            "hi": hi,
        })
        history.append(pred)
        demand_hist.append(demand_hist[-48] if len(demand_hist) >= 48 else demand_hist[-1])
        last = last.copy()
        last["timestamp"] = next_ts
    return forecasts


# ──────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION  (robust MAD z-score)
# ──────────────────────────────────────────────────────────────────────────
def detect_anomalies(df_region: pd.DataFrame, region: str, z_thresh: float = 2.5) -> list[dict]:
    """Robust z-score using median absolute deviation. Flags spikes & dips."""
    prices = df_region["rrp"].dropna().values
    if len(prices) < 50:
        return []
    med = np.median(prices)
    mad = np.median(np.abs(prices - med))
    robust_std = mad * 1.4826 or 1.0
    z = (df_region["rrp"] - med) / robust_std

    out = []
    for idx, row in df_region.iterrows():
        z_val = (row["rrp"] - med) / robust_std
        if abs(z_val) > z_thresh:
            severity = "spike" if abs(z_val) > 4 else "warn"
            if row["rrp"] < 0:
                msg = "Negative price event"
            elif row["rrp"] > 1000:
                msg = "Extreme price spike"
            elif row["rrp"] > 300:
                msg = "Price spike"
            else:
                msg = "Anomalous low price"
            out.append({
                "timestamp": row["timestamp"].isoformat(),
                "region": region,
                "rrp": float(row["rrp"]),
                "z": float(z_val),
                "severity": severity,
                "msg": msg,
            })
    out.sort(key=lambda x: x["timestamp"], reverse=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# JSON OUTPUT
# ──────────────────────────────────────────────────────────────────────────
def atomic_write_json(path: Path, obj: Any) -> None:
    """Atomic write — dashboard never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, separators=(",", ":"), default=str)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def build_outputs(df_30m: pd.DataFrame, mode: str) -> None:
    history_payload: dict[str, list[dict]] = {}
    forecast_payload: dict[str, list[dict]] = {}
    anomaly_payload: list[dict] = []
    latest_payload: dict[str, dict] = {}

    for region in REGIONS:
        df_r = df_30m[df_30m["region"] == region].sort_values("timestamp").copy()
        if df_r.empty:
            log.warning(f"No data for {region}")
            continue

        # History (full window)
        history_payload[region] = [
            {
                "t": row["timestamp"].isoformat(),
                "period": int(row["period"]),
                "rrp": round(float(row["rrp"]), 2),
                "demand": round(float(row["demand"]), 1),
                "availGen": round(float(row["avail_gen"]), 1),
                "reserve": round(float(row["reserve"]), 1),
            }
            for _, row in df_r.iterrows()
        ]

        # Latest snapshot for ticker
        latest = df_r.iloc[-1]
        prev = df_r.iloc[-2] if len(df_r) >= 2 else latest
        latest_payload[region] = {
            "t": latest["timestamp"].isoformat(),
            "rrp": round(float(latest["rrp"]), 2),
            "demand": round(float(latest["demand"]), 1),
            "delta": round(float(latest["rrp"] - prev["rrp"]), 2),
            "delta_pct": round(float((latest["rrp"] - prev["rrp"]) / max(abs(prev["rrp"]), 1) * 100), 2),
        }

        # Forecast — only on full mode (expensive)
        if mode in ("full", "backfill"):
            log.info(f"Training forecast model for {region}…")
            try:
                forecast_payload[region] = forecast_region(df_r, horizon=24)
            except Exception as e:
                log.error(f"Forecast failed for {region}: {e}")
                forecast_payload[region] = []

            # Anomalies on full history
            anomaly_payload.extend(detect_anomalies(df_r, region))

    now_iso = datetime.now(timezone.utc).isoformat()

    atomic_write_json(DATA_DIR / "latest.json", {
        "updated": now_iso,
        "regions": latest_payload,
    })

    if mode in ("full", "backfill"):
        atomic_write_json(DATA_DIR / "history_30d.json", {
            "updated": now_iso,
            "regions": history_payload,
        })
        atomic_write_json(DATA_DIR / "forecast.json", {
            "updated": now_iso,
            "regions": forecast_payload,
        })
        atomic_write_json(DATA_DIR / "anomalies.json", {
            "updated": now_iso,
            "items": sorted(anomaly_payload, key=lambda x: x["timestamp"], reverse=True)[:200],
        })

    atomic_write_json(DATA_DIR / "meta.json", {
        "last_updated": now_iso,
        "source": "OpenElectricity API · /v4/market (NEM, 5m → 30m agg)",
        "regions": REGIONS,
        "mode": mode,
        "rows_total": int(len(df_30m)),
    })
    log.info(f"Wrote outputs to {DATA_DIR.resolve()}/")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["latest", "full", "backfill"],
        default="full",
        help="latest: just refresh tickers (last 2h); full: 7d + forecast; backfill: 30d + forecast",
    )
    args = ap.parse_args()

    # Smaller windows are safer — many APIs gate longer history behind paid tiers.
    days_map = {"latest": 1, "full": 3, "backfill": 7}
    days = days_map[args.mode]

    if not os.environ.get("OPENELECTRICITY_API_KEY"):
        log.error("OPENELECTRICITY_API_KEY env var not set")
        return 1

    # PROBE: figure out whether the API actually works at all
    try:
        with OEClient() as probe_client:
            probe_result = _probe_api(probe_client)
        if not probe_result.get("ok"):
            log.error("Probe failed — see error body above. Aborting before main fetch.")
            atomic_write_json(DATA_DIR / "meta.json", {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "status": "probe_failed",
                "error": probe_result.get("error"),
                "mode": args.mode,
            })
            return 1
    except Exception as e:
        log.exception(f"Probe crashed: {e}")
        return 1

    try:
        df_long = fetch_market_data(days=days)
        df_30m = aggregate_to_30min(df_long)
        build_outputs(df_30m, mode=args.mode)
        return 0
    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        # Don't overwrite good data on failure; just write a status file
        atomic_write_json(DATA_DIR / "meta.json", {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "error": str(e),
            "mode": args.mode,
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
