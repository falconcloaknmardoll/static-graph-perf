"""Phase 7 coverage check (O15): run the six queries declared in phase7_meta_analysis.md.

Saves each raw Semantic Scholar response plus a deduplicated, citation-sorted candidate list.
Screening against the scope rule is done afterwards, by hand, from candidates.json.

  python scripts/phase7_coverage_search.py
"""
import datetime, json, os, time, urllib.parse, urllib.request

QUERIES = [
    "learned cost model tensor program",
    "learned performance model",
    "graph neural network performance prediction",
    "basic block throughput prediction",
    "deep learning model latency prediction",
    "high-level synthesis performance prediction graph neural network",
]
FIELDS = "title,year,venue,citationCount,externalIds,abstract"
OUT = "experiments/phase7_coverage"
os.makedirs(OUT, exist_ok=True)


def fetch(q):
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?" +
           urllib.parse.urlencode({"query": q, "limit": 100, "year": "2018-2026", "fields": FIELDS}))
    # The anonymous API pool is shared and often saturated; back off patiently (~2 h max).
    for attempt in range(60):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(min(30 * (attempt + 1), 180))
    raise RuntimeError(f"rate-limited too often: {q}")


today = datetime.date.today().isoformat()
papers = {}
for i, q in enumerate(QUERIES, 1):
    resp = fetch(q)
    json.dump({"query": q, "retrieved": today, "response": resp},
              open(f"{OUT}/raw_q{i}.json", "w"), indent=1)
    hits = resp.get("data", [])
    print(f"q{i}: {len(hits)} hits  ({q})", flush=True)
    for p in hits:
        e = papers.setdefault(p["paperId"], {**p, "queries": []})
        e["queries"].append(i)
    time.sleep(3)

cands = sorted(papers.values(), key=lambda p: -(p.get("citationCount") or 0))
json.dump({"retrieved": today, "n": len(cands), "candidates": cands},
          open(f"{OUT}/candidates.json", "w"), indent=1)
print(f"{len(cands)} unique candidates -> {OUT}/candidates.json")
