"""Build paper/refs.bib from fetched metadata, never from memory.

arXiv abstract pages give full author lists; the published venue comes from the audit's own
per-paper record (VENUE_BY_ARXIV) or OpenAlex, else the entry stays an arXiv @misc; Crossref
resolves DOI-only works. MLSys requires every reference to list ALL authors, so no entry is
truncated. Raw responses are cached in paper/bib_sources/.

  python scripts/build_bib.py
"""
import html, json, os, re, time, unicodedata, urllib.parse, urllib.request

OUT_DIR = "paper/bib_sources"
os.makedirs(OUT_DIR, exist_ok=True)

ARXIV = ["2308.13490", "2008.01040", "2403.01232", "2405.16623", "1808.07412", "2210.03894",
         "2310.13212", "2310.16792", "1805.08166", "2211.03578", "2606.09312", "2506.00424",
         "2003.10536", "2301.10999", "2407.13853", "2511.01872", "2509.26476", "2506.20886",
         "2201.06848", "1806.08804", "1706.02216", "1905.12265", "2006.05205"]
NO_ARXIV = ["TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers"]
DOIS = ["10.1111/j.2517-6161.1995.tb02031.x",   # Benjamini & Hochberg 1995
        "10.1145/3360307",                      # Jouppi et al. 2020, TPU v2/v3 (CACM)
        "10.1093/biomet/30.1-2.81",             # Kendall 1938
        "10.1145/1390156.1390306"]              # Xia et al. 2008, ListMLE (ICML)

NEURIPS_DB = "Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track"
VENUES = {"NeurIPS": "Advances in Neural Information Processing Systems (NeurIPS)",
          "ICML": "International Conference on Machine Learning (ICML)",
          "ICLR": "International Conference on Learning Representations (ICLR)",
          "MLSys": "Proceedings of Machine Learning and Systems (MLSys)",
          "IISWC": "IEEE International Symposium on Workload Characterization (IISWC)",
          "DAC": "ACM/IEEE Design Automation Conference (DAC)",
          "SC": "International Conference for High Performance Computing, Networking, Storage and Analysis (SC)"}
# Venue as recorded in the audit's own per-paper headings (phase7_meta_analysis.md §A), which were
# read from each paper; OpenAlex files these under their arXiv records.
VENUE_BY_ARXIV = {"2308.13490": (NEURIPS_DB, "2023"), "2003.10536": (VENUES["ICML"], "2021"),
                  "2008.01040": (VENUES["MLSys"], "2021"), "1805.08166": (VENUES["NeurIPS"], "2018"),
                  "2403.01232": (VENUES["ICLR"], "2024"), "2310.13212": (VENUES["IISWC"], "2023"),
                  "2506.00424": (VENUES["ICML"], "2025"), "2201.06848": (VENUES["DAC"], "2022"),
                  "2310.16792": (VENUES["SC"], "2024")}
VENUE_BY_TITLE = {"TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers": (NEURIPS_DB, "2021")}
OPENALEX_NAMES = {"Neural Information Processing Systems": VENUES["NeurIPS"]}


def sentence_case_if_shouting(t):
    return t.capitalize() if t.isupper() else t


def get(url, cache):
    path = os.path.join(OUT_DIR, cache)
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    req = urllib.request.Request(url, headers={"User-Agent": "refs-builder/1.0 (academic use)"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8")
            open(path, "w", encoding="utf-8").write(body)
            time.sleep(3)
            return body
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(url)


def norm(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", t)


def tex(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")



def published_venue(title):
    """OpenAlex (DBLP sits behind a bot challenge): the first exact-title work with a
    conference or journal location that is not arXiv. None means preprint only."""
    q = urllib.parse.quote(title)
    body = get(f"https://api.openalex.org/works?search={q}&per-page=10", f"openalex_{norm(title)[:60]}.json")
    for w in json.loads(body)["results"]:
        if norm(w.get("display_name") or "") != norm(title):
            continue
        for loc in w.get("locations", []):
            s = loc.get("source") or {}
            if s.get("type") in ("conference", "journal") and "arxiv" not in (s.get("display_name") or "").lower():
                return {"venue": s["display_name"], "year": str(w["publication_year"]),
                        "kind": "article" if s["type"] == "journal" else "inproceedings",
                        "title": w["display_name"],
                        "authors": [a["author"]["display_name"] for a in w["authorships"]]}
    return None


def key_for(authors, year, title):
    last = norm(authors[0].split()[-1]) if authors else "anon"
    word = next((norm(w) for w in title.split() if len(norm(w)) > 3), "paper")
    return f"{last}{year}{word}"


entries, report = [], []

# arXiv abstract pages: their citation_* meta tags carry the full author list.
# (The export API answered 406 to every request on 2026-09-27.)
for aid in ARXIV:
    page = get(f"https://arxiv.org/abs/{aid}", f"arxiv_{aid}.html")
    meta = lambda name: [html.unescape(m) for m in re.findall(rf'<meta name="{name}" content="([^"]*)"', page)]
    title = re.sub(r"\s+", " ", meta("citation_title")[0]).strip()
    authors = [" ".join(reversed([p.strip() for p in a.split(",", 1)])) for a in meta("citation_author")]
    year = (meta("citation_date") or meta("citation_online_date"))[0][:4]
    # Authors always from arXiv (complete, as the authors wrote them); venue from the audit
    # record, else OpenAlex.
    fields = {"title": "{" + tex(title) + "}", "author": " and ".join(tex(a) for a in authors)}
    ven = published_venue(title) if aid not in VENUE_BY_ARXIV else None
    if aid in VENUE_BY_ARXIV:
        kind = "inproceedings"
        fields["booktitle"], fields["year"] = VENUE_BY_ARXIV[aid]
        src = f"arXiv+audit ({fields['booktitle'][-12:]} {fields['year']})"
    elif ven:
        kind = ven["kind"]
        fields["journal" if kind == "article" else "booktitle"] = tex(OPENALEX_NAMES.get(ven["venue"], ven["venue"]))
        fields["year"] = ven["year"]
        src = f"arXiv+OpenAlex ({ven['venue'][:28]} {ven['year']})"
    else:
        kind = "misc"
        fields.update({"year": year, "eprint": aid, "archivePrefix": "arXiv",
                       "howpublished": f"arXiv:{aid}"})
        src = "arXiv only"
    fields["note"] = f"arXiv:{aid}" if kind != "misc" else None
    k = key_for(authors, fields["year"], title)
    entries.append((kind, k, fields))
    report.append((k, src, len(fields["author"].split(" and ")), title[:60]))

for title in NO_ARXIV:
    ven = published_venue(title)
    if not ven:
        report.append(("MISSING", "OpenAlex no match", 0, title[:60])); continue
    auth, kind = ven["authors"], ven["kind"]
    venue, year = VENUE_BY_TITLE.get(title, (OPENALEX_NAMES.get(ven["venue"], ven["venue"]), ven["year"]))
    fields = {"title": "{" + tex(ven["title"]) + "}", "author": " and ".join(tex(a) for a in auth),
              ("journal" if kind == "article" else "booktitle"): tex(venue), "year": year}
    ven["venue"], ven["year"] = venue, year
    k = key_for(auth, ven["year"], ven["title"])
    entries.append((kind, k, fields)); report.append((k, f"OpenAlex ({ven['venue'][:28]} {ven['year']})", len(auth), ven["title"][:60]))

for doi in DOIS:
    body = get(f"https://api.crossref.org/works/{doi}", f"crossref_{norm(doi)}.json")
    m = json.loads(body)["message"]
    auth = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])]
    title = sentence_case_if_shouting(m["title"][0])
    year = str(m.get("issued", {}).get("date-parts", [[None]])[0][0])
    container = (m.get("container-title") or [""])[0]
    kind = "inproceedings" if m.get("type") == "proceedings-article" else "article"
    fields = {"title": "{" + tex(title) + "}", "author": " and ".join(tex(a) for a in auth),
              ("booktitle" if kind == "inproceedings" else "journal"): tex(container), "year": year,
              "volume": m.get("volume"), "number": m.get("issue"), "pages": m.get("page"), "doi": doi}
    k = key_for(auth, year, title)
    entries.append((kind, k, fields)); report.append((k, "Crossref", len(auth), title[:60]))

with open("paper/refs.bib", "w", encoding="utf-8") as f:
    f.write("% Generated by scripts/build_bib.py from arXiv, OpenAlex, Crossref and the audit record.\n"
            "% Do not hand-edit: fix the script or its source lists and regenerate.\n\n")
    for kind, k, fields in sorted(entries, key=lambda x: x[1]):
        # title values already carry one brace pair; the second protects their capitalization.
        body = ",\n".join(f"  {fk} = {{{fv}}}" for fk, fv in fields.items() if fv)
        f.write(f"@{kind}{{{k},\n{body}\n}}\n\n")

for r in sorted(report):
    print(f"{r[0]:32s} {r[2]:3d} authors  {r[1]:40s} {r[3]}")
