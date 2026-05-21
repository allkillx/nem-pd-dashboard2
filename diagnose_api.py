"""
OpenElectricity API diagnostic script.
Bypasses the SDK and hits the REST API directly so we can see the raw
error body the SDK is hiding.
"""
import os
import sys
import json
from datetime import datetime, timedelta, timezone
import urllib.request
import urllib.parse
import urllib.error

API_KEY = os.environ.get("OPENELECTRICITY_API_KEY", "")
BASE_URL = "https://api.openelectricity.org.au/v4"

if not API_KEY:
    print("ERROR: OPENELECTRICITY_API_KEY not set")
    sys.exit(1)

print(f"Key length: {len(API_KEY)} chars (first 4: {API_KEY[:4]}…)")
print(f"Base URL: {BASE_URL}")
print()


def call(path, params=None):
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    print(f"────────────────────────────────────────────────────────")
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "User-Agent": "nem-pd-dashboard/diagnose",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"OK  HTTP {resp.status}")
            body = resp.read().decode("utf-8", errors="replace")
            print(f"Body preview: {body[:800]}")
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    print(f"Top-level keys: {list(parsed.keys())}")
            except Exception:
                pass
    except urllib.error.HTTPError as e:
        print(f"FAIL  HTTP {e.code} {e.reason}")
        body = e.read().decode("utf-8", errors="replace")
        print(f"Error body: {body}")
        print(f"Response headers: {dict(e.headers)}")
    except Exception as e:
        print(f"FAIL  {type(e).__name__}: {e}")
    print()


# Test 1: /me - cheapest auth check
call("/me")

# Test 2: List networks
call("/networks")

# Test 3: Market data minimal
call("/market/NEM", params={"metrics": "price", "interval": "1h"})

# Test 4: With date_start
start = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
call("/market/NEM", params={
    "metrics": "price",
    "interval": "1h",
    "date_start": start,
})

# Test 5: With primary_grouping
call("/market/NEM", params={
    "metrics": "price",
    "interval": "1h",
    "date_start": start,
    "primary_grouping": "network_region",
})

# Test 6: /data endpoint
call("/data/network/NEM", params={
    "metrics": "power",
    "interval": "1h",
    "date_start": start,
})

print("Diagnostic complete.")
