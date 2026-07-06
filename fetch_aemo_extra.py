"""
fetch_aemo_extra.py — NEM Dashboard 补充数据管道
=================================================
与 fetch_nem.py 并行运行、互不干扰。抓取三类新数据:

  1. AEMO PREDISPATCH (PD)   — 官方 30-min 预调度价格/需求预测,未来 ~40h
                                来源: nemweb PredispatchIS_Reports (每30分钟更新)
  2. AEMO ST PASA            — 未来 7 天系统充裕度: POE10/50/90 需求 + 可用容量
                                来源: nemweb Short_Term_PASA_Reports (每小时更新)
  3. 天气                     — 各州首府温度/太阳辐射/风速 (驱动需求和新能源出力)
                                来源: Open-Meteo (免费, 无需 API key)

输出 (原子写入, 前端轮询):
  data/predispatch.json  — { updated, run, regions: {NSW1: [{t, rrp, demand, availGen}]} }
  data/stpasa.json       — { updated, run, regions: {NSW1: [{t, demand10, demand50, demand90,
                                                             availCap, surplus, lor}]} }
  data/weather.json      — { updated, regions: {NSW1: {city, current, hourly: [...]}} }

调度建议 (cron / GitHub Actions):
  每 30 min  →  python fetch_aemo_extra.py                (全部三类)
  或分开     →  python fetch_aemo_extra.py --only weather (只跑某一类)

无需 API key。仅依赖 Python 标准库。
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aemo_extra")

DATA_DIR = Path(os.environ.get("NEM_DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}

# NEM 市场时间 = 固定 UTC+10 (无夏令时)
NEM_TZ = timezone(timedelta(hours=10))

NEMWEB = "https://nemweb.com.au"
PREDISPATCH_DIR = f"{NEMWEB}/Reports/Current/PredispatchIS_Reports/"
STPASA_DIR = f"{NEMWEB}/Reports/Current/Short_Term_PASA_Reports/"

# nemweb 会拒绝默认的 python UA, 必须带浏览器 UA
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nem-dashboard/1.0)"}

# 各州代表城市 (天气取样点)
CITIES = {
    "NSW1": {"city": "Sydney",    "lat": -33.87, "lon": 151.21},
    "VIC1": {"city": "Melbourne", "lat": -37.81, "lon": 144.96},
    "QLD1": {"city": "Brisbane",  "lat": -27.47, "lon": 153.03},
    "SA1":  {"city": "Adelaide",  "lat": -34.93, "lon": 138.60},
    "TAS1": {"city": "Hobart",    "lat": -42.88, "lon": 147.33},
}


# ──────────────────────────────────────────────────────────────────────────
# 通用工具
# ──────────────────────────────────────────────────────────────────────────
def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, separators=(",", ":"), default=str)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def latest_zip(dir_url: str, pattern: str) -> tuple[str, str]:
    """
    抓 nemweb 目录列表 HTML, 用正则找出最新的 zip 文件。
    pattern 第一个捕获组必须是 YYYYMMDDHHMM 时间戳。
    返回 (完整URL, run时间戳字符串)。
    """
    html = http_get(dir_url).decode("utf-8", errors="replace")
    matches = re.findall(pattern, html)
    if not matches:
        raise RuntimeError(f"目录列表里没找到匹配 {pattern} 的文件: {dir_url}")
    # 时间戳字典序 = 时间序, 取最大即最新
    latest_run = max(m if isinstance(m, str) else m[0] for m in matches)
    # 找出完整文件名 (含尾部序列号)
    fname_match = re.search(
        pattern.replace(r"(\d{12})", latest_run), html
    )
    # 更稳妥: 直接搜含该时间戳的完整文件名
    full = re.search(rf'(PUBLIC_\w+_{latest_run}_\d+\.zip)', html, re.IGNORECASE)
    if not full:
        raise RuntimeError(f"找到了时间戳 {latest_run} 但拼不出完整文件名")
    fname = full.group(1)
    log.info(f"最新文件: {fname}")
    return dir_url + fname, latest_run


def read_zip_first_csv(zip_bytes: bytes) -> str:
    """nemweb 的 zip 里只有一个 CSV, 解出来返回文本。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
        if not names:
            raise RuntimeError(f"zip 里没有 CSV: {zf.namelist()}")
        return zf.read(names[0]).decode("utf-8", errors="replace")


def parse_nem_tables(csv_text: str, wanted: set[tuple[str, str]]) -> dict[tuple, list[dict]]:
    """
    解析 AEMO NEM CSV 格式:
      C,... 注释行
      I,GROUP,TABLE,ver,COL1,COL2,...   表头行
      D,GROUP,TABLE,ver,val1,val2,...   数据行
    wanted = {("PREDISPATCH","REGION_PRICES"), ...}
    返回 {(group,table): [ {col: val}, ... ]}
    """
    out: dict[tuple, list[dict]] = {w: [] for w in wanted}
    headers: dict[tuple, list[str]] = {}
    for parts in csv.reader(io.StringIO(csv_text)):
        if len(parts) < 5:
            continue
        kind = parts[0]
        key = (parts[1].upper(), parts[2].upper())
        if key not in wanted:
            continue
        if kind == "I":
            headers[key] = [h.upper() for h in parts[4:]]
        elif kind == "D" and key in headers:
            out[key].append(dict(zip(headers[key], parts[4:])))
    for k, rows in out.items():
        log.info(f"  表 {k[0]},{k[1]}: {len(rows)} 行")
    return out


def nem_ts_to_iso(s: str) -> str | None:
    """'2026/07/07 18:30:00' (NEM 时间, UTC+10 固定) → ISO 带时区。"""
    s = (s or "").strip().strip('"')
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=NEM_TZ).isoformat()
        except ValueError:
            continue
    return None


def to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# 1) PREDISPATCH — 官方 30-min 预测 (价格 + 需求 + 可用发电)
# ──────────────────────────────────────────────────────────────────────────
def fetch_predispatch() -> None:
    log.info("── PREDISPATCH ──")
    url, run = latest_zip(
        PREDISPATCH_DIR, r"PUBLIC_PREDISPATCHIS_(\d{12})_\d+\.zip"
    )
    csv_text = read_zip_first_csv(http_get(url))
    tables = parse_nem_tables(csv_text, {
        ("PREDISPATCH", "REGION_PRICES"),
        ("PREDISPATCH", "REGION_SOLUTION"),
    })

    # 需求/可用发电: 按 (region, datetime) 索引
    solution: dict[tuple, dict] = {}
    for row in tables[("PREDISPATCH", "REGION_SOLUTION")]:
        region = row.get("REGIONID", "").strip()
        t = nem_ts_to_iso(row.get("DATETIME", ""))
        if region in REGIONS and t:
            solution[(region, t)] = {
                "demand": to_float(row.get("TOTALDEMAND")),
                "availGen": to_float(row.get("AVAILABLEGENERATION")),
            }

    regions_out: dict[str, list[dict]] = {r: [] for r in REGIONS}
    for row in tables[("PREDISPATCH", "REGION_PRICES")]:
        region = row.get("REGIONID", "").strip()
        t = nem_ts_to_iso(row.get("DATETIME", ""))
        rrp = to_float(row.get("RRP"))
        if region not in REGIONS or not t or rrp is None:
            continue
        sol = solution.get((region, t), {})
        regions_out[region].append({
            "t": t,
            "rrp": round(rrp, 2),
            "demand": round(sol["demand"], 1) if sol.get("demand") is not None else None,
            "availGen": round(sol["availGen"], 1) if sol.get("availGen") is not None else None,
        })

    for r in regions_out:
        regions_out[r].sort(key=lambda x: x["t"])
        log.info(f"  {r}: {len(regions_out[r])} 个预调度区间")

    atomic_write_json(DATA_DIR / "predispatch.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "source": "AEMO NEMWEB · PredispatchIS_Reports",
        "regions": regions_out,
    })
    log.info("已写入 data/predispatch.json")


# ──────────────────────────────────────────────────────────────────────────
# 2) ST PASA — 未来 7 天充裕度 (POE 需求 vs 可用容量)
# ──────────────────────────────────────────────────────────────────────────
def fetch_stpasa() -> None:
    log.info("── ST PASA ──")
    url, run = latest_zip(
        STPASA_DIR, r"PUBLIC_STPASA_(\d{12})_\d+\.zip"
    )
    csv_text = read_zip_first_csv(http_get(url))
    tables = parse_nem_tables(csv_text, {("STPASA", "REGIONSOLUTION")})
    rows = tables[("STPASA", "REGIONSOLUTION")]

    # STPASA 每个区间可能有多个 RUNTYPE 的结果; 记录一下, 用出现最多的那个
    runtypes = {}
    for row in rows:
        rt = row.get("RUNTYPE", "").strip() or "(none)"
        runtypes[rt] = runtypes.get(rt, 0) + 1
    log.info(f"  RUNTYPE 分布: {runtypes}")
    prefer_rt = max(runtypes, key=runtypes.get) if runtypes else None

    # 按 (region, interval) 去重, 只保留 prefer_rt 的行 (若有 RUNTYPE 字段)
    dedup: dict[tuple, dict] = {}
    for row in rows:
        rt = row.get("RUNTYPE", "").strip() or "(none)"
        if prefer_rt and rt != prefer_rt:
            continue
        region = row.get("REGIONID", "").strip()
        t = nem_ts_to_iso(row.get("INTERVAL_DATETIME", ""))
        if region not in REGIONS or not t:
            continue
        # 可用容量字段在新旧 schema 名字不同, 逐个尝试
        avail_cap = None
        for col in ("AGGREGATECAPACITYAVAILABLE", "AGGREGATEPASAAVAILABILITY",
                    "UNCONSTRAINEDCAPACITY", "CONSTRAINEDCAPACITY"):
            avail_cap = to_float(row.get(col))
            if avail_cap is not None:
                break
        d50 = to_float(row.get("DEMAND50"))
        dedup[(region, t)] = {
            "t": t,
            "demand10": to_float(row.get("DEMAND10")),
            "demand50": d50,
            "demand90": to_float(row.get("DEMAND90")),
            "availCap": avail_cap,
            "surplus": to_float(row.get("SURPLUSCAPACITY")),
            "lor": (row.get("LORCONDITION") or "").strip() or None,
        }

    regions_out: dict[str, list[dict]] = {r: [] for r in REGIONS}
    for (region, _), rec in dedup.items():
        regions_out[region].append(rec)
    for r in regions_out:
        regions_out[r].sort(key=lambda x: x["t"])
        log.info(f"  {r}: {len(regions_out[r])} 个 PASA 区间")

    atomic_write_json(DATA_DIR / "stpasa.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "runtype": prefer_rt,
        "source": "AEMO NEMWEB · Short_Term_PASA_Reports",
        "regions": regions_out,
    })
    log.info("已写入 data/stpasa.json")


# ──────────────────────────────────────────────────────────────────────────
# 3) 天气 — Open-Meteo (免费, 无 key)
# ──────────────────────────────────────────────────────────────────────────
def fetch_weather() -> None:
    log.info("── 天气 (Open-Meteo) ──")
    regions_out: dict[str, dict] = {}
    for region, cfg in CITIES.items():
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
            "&hourly=temperature_2m,shortwave_radiation,wind_speed_10m"
            "&current=temperature_2m,apparent_temperature,wind_speed_10m,weather_code"
            "&past_days=1&forecast_days=3"
            "&timezone=Australia%2FSydney"
        )
        try:
            data = json.loads(http_get(url).decode("utf-8"))
        except Exception as e:
            log.warning(f"  {region} ({cfg['city']}) 天气抓取失败: {e}")
            continue
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        rads = hourly.get("shortwave_radiation", [])
        winds = hourly.get("wind_speed_10m", [])
        cur = data.get("current", {})
        regions_out[region] = {
            "city": cfg["city"],
            "current": {
                "temp": cur.get("temperature_2m"),
                "apparent": cur.get("apparent_temperature"),
                "wind": cur.get("wind_speed_10m"),
                "code": cur.get("weather_code"),
                "t": cur.get("time"),
            },
            # Open-Meteo 返回的 time 已按请求的 timezone (Sydney), 无偏移量,
            # 前端按本地字符串直接展示即可
            "hourly": [
                {"t": times[i],
                 "temp": temps[i] if i < len(temps) else None,
                 "radiation": rads[i] if i < len(rads) else None,
                 "wind": winds[i] if i < len(winds) else None}
                for i in range(len(times))
            ],
        }
        log.info(f"  {region} ({cfg['city']}): 当前 {cur.get('temperature_2m')}°C, {len(times)} 小时数据")

    atomic_write_json(DATA_DIR / "weather.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "source": "Open-Meteo (open-meteo.com)",
        "regions": regions_out,
    })
    log.info("已写入 data/weather.json")


# ──────────────────────────────────────────────────────────────────────────
# MAIN — 三类互相独立, 一类失败不影响其他
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["predispatch", "stpasa", "weather"],
                    help="只跑某一类; 不传则全跑")
    args = ap.parse_args()

    tasks = {
        "predispatch": fetch_predispatch,
        "stpasa": fetch_stpasa,
        "weather": fetch_weather,
    }
    if args.only:
        tasks = {args.only: tasks[args.only]}

    failures = []
    for name, fn in tasks.items():
        try:
            fn()
        except Exception as e:
            log.exception(f"{name} 失败: {e}")
            failures.append(name)

    if failures:
        log.error(f"以下任务失败: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
