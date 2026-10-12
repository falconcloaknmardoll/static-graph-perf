# Audit coverage assessment: does the recent-work gap need new evidence? (2026-10-12)

> **Read-only assessment.** No paper was scored. No existing result was edited. No paper's full
> text was opened. Sources are the files in the repo only. Every number is followed by `file:line`.
> Paths are relative to `experiments/` unless they start with `paper/` or `DEFENSIBILITY.md`.
> **Declared** = written before the data it governs existed. **Post-hoc** = derived here, today,
> from data already in the repo.
>
> Companion file (NOT frozen, NOT committed): `axis3_recent_coverage_declaration_DRAFT.md`.

## Bottom line

1. **All the registered figures check out.** Frame A covers 8/30 and Frame B covers 0/10. The frame
   retrieves 10 of the 20 sampled papers. The ambiguous → IN rule was declared before the search.
2. **The concern is partly mis-stated.** The sample does not skew old by year: **9 of the 20
   sampled papers are from 2024–2026** (post-hoc count from the §A headers, below). What 0/10
   actually shows is different. The *most-cited* recent in-scope papers in this frame come from
   another part of the area. Seven of the ten are tensor-program tuning or cost-model papers,
   mostly in IEEE/ACM systems and EDA venues. The sample has none of these 10.
3. **Even so, the reviewer objection is real.** Option A leaves it standing: "the recent papers
   you chose are not the recent papers the field cites most."
4. **Recommendation: Option B.** It costs about 5–7 hours of the first author's reading, done between
   Oct 19 and Oct 22. It must be declared before any PDF is opened. The 20-paper headline stays
   untouched. The post-hoc year split in A is kept as a no-cost fallback either way.
5. **Two risks need to be in the declaration up front:**
   - **Access:** 5 of the 10 papers are closed-access with no arXiv version.
   - **Clustering:** 3 of the 10 share 6–8 authors, so they are not independent.

---

## TASK 1 — Facts

### 1a. Registered figures and the ambiguous → IN rule

| Fact | Value | Source |
|---|---|---|
| Frame A coverage | **8 of 30** | `phase7_coverage/screening.md:11` |
| Frame B coverage | **0 of 10** | `phase7_coverage/screening.md:12` |
| In-scope after screening and de-duplication | 62 of 571 | `phase7_coverage/screening.md:13` |
| Sensitivity, *not registered* (all ambiguous calls → OUT) | 10/30 and 2/10 | `phase7_coverage/screening.md:15-17` |
| Frame retrieves only 10 of the 20 corpus papers | 10/20 | `phase7_coverage/screening.md:39-41` |
| Ambiguous → IN rule (**declared** 2026-09-27, before the search ran) | "**Ambiguous abstracts are screened IN.** This enlarges the frame … so it can only lower measured coverage." | `phase7_meta_analysis.md:119-120` (declaration header `:94-98`) |
| Frame definitions (declared) | A = 30 most-cited in-scope; B = 10 most-cited in-scope from 2024–2026 | `phase7_meta_analysis.md:113-116` |
| Match rule (declared) | same arXiv ID or same title | `phase7_meta_analysis.md:121` |
| Status FINAL; the first author agreed with all 31 flagged calls | — | `phase7_coverage/screening.md:6-7`; `DEFENSIBILITY.md:207-211` |

**Independent re-check, done today:**
- `screening.json` holds 571 entries, aligned 1:1 by title with `candidates.json`.
- Exactly 10 entries carry a `corpus_match`: A3, A4, A6, A9, A10, A11, A14, A15, A18 and A19.
- Searching `candidates.json` by arXiv ID *and* by title keyword finds none of A1, A2, A5, A7,
  A8, A12, A13, A16, A17 or A20. The 10-of-20 figure is confirmed.

**Screening steps that were *not* in the declaration** (disclosed in the log, but post-hoc relative
to the 2026-09-27 protocol):
- **Title-only exclusion of about 480 off-domain papers.** The protocol says title + abstract.
  (`phase7_coverage/screening.md:21-24`)
- **The A14 rule extended to analytical models and benchmarks** (uiCA, Crop, BHive).
  (`phase7_coverage/screening.md:26-28`)
- **The tie-break by S2 file order.** The protocol only says "sorted by citation count"
  (`phase7_meta_analysis.md:113`); ties are handled at `phase7_coverage/screening.md:31-33`.
  - The tie-break decides Frame B's 10th place: positions 9–11 all have 2 citations.
  - The paper it pushes out, Intelligent4DSE (#467), is uncovered too, so the 0/10 count does
    not change.

### 1b. The 10 Frame B papers and the 22 uncovered Frame A papers

**About the columns:**
- *Venue* is Semantic Scholar's `venue` field in `candidates.json`; the `title@` line is given
  once per paper.
- *OA* means an open-access PDF or an arXiv ID is recorded in `candidates.json`. It decides
  whether the first author can get the paper without a library.

**Frame B** (`phase7_coverage/screening.md:84-95`; none in the corpus)

| B# | S2 # | Year | Cites | Title (short) | Venue (S2) | OA? | Flags |
|---|---|---|---|---|---|---|---|
| 1 | 319 | 2024 | 16 | LOOPer | PACT (`candidates.json:14641`) | arXiv 2403.11522 | Headline is a speedup over autoschedulers; cost model is a component (`screening.json:3830` reason) |
| 2 | 384 | 2024 | 10 | Learning to Compare Hardware Designs for HLS | MLCAD workshop (`:17647`) | arXiv + gold | Clear scope (pairwise model vs HARP) |
| 3 | 387 | 2024 | 9 | Pruner | ASPLOS (`:17793`) | arXiv 2402.02361 | **Ambiguous**, spot-checked (`screening.md:114`). Search-mechanism claim; same risk as the Ansor exclusion (`phase7_meta_analysis.md:84`) |
| 4 | 423 | 2024 | 5 | Fasor | ICS (`:19441`) | gold | **Ambiguous** (`screening.md:115`). Framework claim, same risk as Pruner |
| 5 | 433 | 2024 | 4 | Crop | DAC (`:19878`) | **closed** | Analytical model, IN only by the A14 rule (`screening.md:116`) |
| 6 | 439 | 2024 | 4 | ORXESTRA (context-aware BB timing) | CC (`:20180`) | green | Clear scope |
| 7 | 443 | 2025 | 3 | NLTSP | J. Systems Architecture (`:20365`) | **closed** | **No abstract in S2**: IN on title alone (`screening.md:118`) |
| 8 | 444 | 2024 | 3 | UniCoMo | ICCD (`:20405`) | **closed** | **Same group as B9 and B10** (6 shared authors) |
| 9 | 451 | 2026 | 2 | UniCoX | IEEE TC (`:20695`) | **closed** | **Likely the journal extension of B8** (8 shared authors with B10, 6 with B8; titles near-identical). Not a duplicate under the strict title rule |
| 10 | 452 | 2026 | 2 | UniSparTa | IEEE TCAD (`:20751`) | **none recorded** | **Ambiguous** (`screening.md:120`); same group as B8 and B9 |

**What the Frame B flags mean:**
- **Exclusion list overlap.** None of the 10 is on the corpus exclusion list
  (`phase7_meta_analysis.md:80-88`). But the 11th Frame B paper, Intelligent4DSE (#467, tied at
  2 citations), **is** on that list: "found by search snippet only; not fetched or scored before
  the n=20 freeze" (`phase7_meta_analysis.md:87`). The registered tie-break keeps it out, and the
  declaration should say so explicitly.
- **Duplicates.** There are no strict duplicates, but **B8, B9 and B10 come from one group**
  (author overlap computed from `candidates.json` author lists). For B, the effective number of
  independent groups is about 8, not 10.
- **Closer-read scope risk.** Pruner, Fasor, UniSparTa, NLTSP and LOOPer may turn out, on a
  full read, to make no comparative cost-model claim. Pruner, Fasor and UniSparTa were already
  ambiguous, and NLTSP was screened on title only. A declared N/A rule is needed.
- **Access.** 5 of the 10 are open (B1, B2, B3, B4, B6). B5, B7, B8 and B9 are recorded as
  CLOSED with no arXiv ID, and B10 has no open-access record (`candidates.json` `openAccessPdf`).
  Whether the first author has institutional access **cannot be determined from the repo**.

**Frame A, 22 uncovered** (`phase7_coverage/screening.md:49-80`; the 8 covered papers are A10,
A4, A9, A18, A11, A6, A3 and A14)

| A# | Year | Cites | Title (short) | Venue (S2) | Flags |
|---|---|---|---|---|---|
| 2 | 2021 | 223 | nn-Meter | MobiSys (`:1723`) | closed; shares authors with CoDL |
| 4 | 2022 | 103 | CoDL | MobiSys (`:3537`) | **ambiguous**: system paper (`screening.md:104`); closed |
| 8 | 2023 | 70 | IronMan-Pro | TCAD (`:5477`) | Design-space exploration via RL; predictor is a sub-contribution |
| 10 | 2022 | 55 | DietCode | MLSys (`:6631`) | **ambiguous, no abstract** (`screening.md:106`); auto-scheduler, same risk as the Ansor exclusion |
| 11 | 2023 | 53 | DNNPerf (Runtime Perf. Prediction w/ GNN) | ICSE-SEIP (`:7012`) | closed |
| 12 | 2023 | 37 | WACO | ASPLOS (`:9383`) | Is the *baseline* in corpus paper A15 (`phase7_meta_analysis.md:357,362`) |
| 13 | 2021 | 37 | uiCA | ICS (`:9514`) | Analytical; A14 rule; spot-checked |
| 14 | 2022 | 36 | GNNs for HLS DSE | TODAES (`:9710`) | — |
| 15 | 2019 | 32 | BHive | IISWC (`:10153`) | **ambiguous**: benchmark paper (`screening.md:108`) |
| 17 | 2019 | 22 | Learned TPU Cost Model for XLA | (none) (`:12508`) | **Workshop precursor of corpus paper A9.** Same work family, counted uncovered only because the title differs (`screening.md:34-35`). Effectively a duplicate |
| 18 | 2022 | 19 | NAR-Former | CVPR (`:13587`) | Predicts accuracy *and* latency |
| 19 | 2024 | 16 | LOOPer | PACT | also B1 |
| 20 | 2019 | 14 | Absinthe | PACT (`:15475`) | ambiguous |
| 21 | 2022 | 13 | NoCeption | DATE (`:16227`) | **ambiguous**: NoC power/performance/area prediction, borderline scope |
| 22 | 2023 | 12 | CDMPP | EuroSys (`:16305`) | — |
| 23 | 2023 | 12 | FamilySeer (Subgraph Similarities) | ICPP (`:16358`) | — |
| 25 | 2022 | 10 | Moses | arXiv (`:17239`) | arXiv-only per S2 (#389 merged into it as a duplicate, `screening.md:30`) |
| 26 | 2023 | 10 | GNN QoR estimation for HLS | IEEE Access (`:17557`) | — |
| 27 | 2024 | 10 | Learning to Compare HW Designs | MLCAD | also B2 |
| 28 | 2024 | 9 | Pruner | ASPLOS | also B3 |
| 29 | 2023 | 5 | ATFormer | EMNLP (`:19396`) | — |
| 30 | 2024 | 5 | Fasor | ICS | also B4 |

**What the Frame A flags mean:**
- **Overlap with Frame B.** 4 of the 22 are also in Frame B (LOOPer, Learning to Compare,
  Pruner, Fasor). So Option C adds **18** new papers to B, or **17** without the A9 precursor.
- **Exclusion list overlap.** None of the 22 is on the corpus exclusion list.
- **Out-of-scope risk on a full read** (ambiguous or framework-type claims): CoDL, DietCode,
  BHive, Absinthe, NoCeption, Pruner and Fasor.

### 1c. Which 10 sampled papers does the frame miss, and why?

**The missed papers** (`phase7_coverage/screening.md:39-41`):
- A1 TPUGraphs: a dataset paper (`phase7_meta_analysis.md:129-132`).
- A2 TGraph.
- A5 ProGraML: a program-representation paper.
- A7 PerfSAGE.
- A8 NeuSight: "GPU performance forecasting".
- A12 Polynormer: a general GNN architecture paper; TPUGraphs appears only in Appendix L
  (`phase7_meta_analysis.md:301`).
- A13 "Learned Cost Model for Placement on Reconfigurable Dataflow Hardware".
- A16 Regression Language Models for Code: LLM regression.
- A17 Omniwise: an LLM-based kernel predictor.
- A20 PerfVec.

**What the repo shows:**
- Each query kept only the **top 100 of a very large, noisy result set.** Total S2 hits were
  726, 1,993,866, 118,769, 1,598, 146,418 and 10,420 (`phase7_coverage/raw_q1.json:5` …
  `raw_q6.json:5`).
- **Relevance ranking was poor.** About 480 of 571 retrieved papers were off-domain
  (`phase7_coverage/screening.md:21-24`). Query 1's top 100 includes a point-cloud compression
  paper and a medical-aid-in-dying cohort study (`phase7_coverage/raw_q1.json:1359,1457`).
- A13's title contains the phrase "learned cost model" and still was not retrieved. That points
  to truncation and ranking, not to query vocabulary alone.

**Likely causes, by paper type.** These are inferences, *not verifiable from the repo*: the
raw files only hold the top 100, so they cannot show why a paper was missed.
- **Paper type outside the query phrasing:** a dataset paper (A1), a general GNN paper (A12),
  a program-representation paper (A5), and LLM-based predictors (A16, A17).
- **Vocabulary mismatch:** "forecasting" (A8); "inference performance" rather than "latency"
  (A7).
- **Recent papers with few citations** sitting below the top 100 of huge result sets (A2, A13,
  A16, A17).

Whether these papers are indexed in S2 at all was **not checked**. Checking it would take a new
query, which is out of scope for a read-only assessment.

**Weakness of the search or of the sample?** Mainly **of the search**. A frame that misses half
of a hand-built sample, including the benchmark the project is built on, is not a complete
census of the area. Its 0/10 is therefore a statement about *this frame's* recent top-10, not
about recent work in general. But there are two things this cannot excuse:
1. The sample was assembled without a logged search (`phase7_meta_analysis.md:75-78`).
2. The frame shows a real **sub-area gap**:
   - TVM-ecosystem tensor-program tuning papers in IEEE/ACM venues: B3–B5 and B7–B10.
   - DL-model latency predictors: nn-Meter, CoDL, DNNPerf and NAR-Former, none covered
     (`phase7_coverage/screening.md:42-44`).

Both weaknesses exist. The search's weakness limits what 0/10 can *prove*. The sub-area gap is
what a reviewer can *point at*.

**The concern, restated accurately.** This is post-hoc, from the §A headers at
`phase7_meta_analysis.md:143,230,301,318,354,369,385,416,432`. The corpus has **9 papers from
2024–2026**: A2, A8, A12, A13, A15, A16, A17, A19 and A20. Two of them (A15, A19) *are* in the
frame, but rank low because they have 1 and 0 citations (`screening.json:5678,5822`). So the
problem is not that the sample is old. It is that the sample's recent papers are not the
**most-cited** recent ones. Note also that Frame B's citation counts run from 2 to 16
(`phase7_coverage/screening.md:45`), so "most-cited" is a weak criterion this close to
retrieval.

---

## TASK 2 — Blindness check

What was **seen during screening** about the evaluation practice of the 28 unique uncovered
papers (Frame B ∪ uncovered Frame A):

1. **Abstracts** (`candidates.json` `abstract`) were read for all papers screened on title +
   abstract (`phase7_coverage/screening.md:25`). The first author re-read the full abstracts of the 31
   flagged papers during the spot-check (`DEFENSIBILITY.md:208`).
2. **Today's keyword scan of those abstracts** covered: seed, signific*, statistic*, p-value,
   p <, error bar, confidence, standard deviation, std, variance, t-test, Wilcoxon, bootstrap,
   mean and ±.
   - **No abstract states a number of seeds, a hypothesis test, a p-value, error bars,
     confidence intervals or dispersion.**
   - **What *was* visible: unqualified "significantly" claims.** Examples:
     - nn-Meter: "significantly outperforms the prior state-of-the-art".
     - CDMPP: "significantly outperforms state-of-the-art baselines with 14.03% and 10.85%".
     - Learning to Compare: "achieves significant improvements in ranking metrics".
     - Fasor, UniCoMo and UniSparTa use "significantly" about tuning time or adaptability.
   - These are colloquial uses with no statistic. **The screener has therefore seen that
     several abstracts *claim* significance in words.** Under the rubric, words alone do not
     earn a P (see the declaration's operational rule). The blindness statement must still
     disclose this, because it could in principle predispose a scorer toward F.
   - The "p <" match in UniSparTa's abstract is a LaTeX artefact, not a p-value.
   - The "variance" matches (UniCoX, CDMPP, the HLS-DSE paper) are MathML or "invariant", not
     statistics.
3. **Three papers had no abstract in S2:** NLTSP (B7), DietCode, and the Learned TPU cost model
   (`candidates.json` `abstract` empty). Nothing about their evaluation was seen.
4. **Indirect exposure through corpus papers.** the first author read A7 and A15 in full during the Axis 3
   check, and they use **nn-Meter** and **WACO** as *baselines*
   (`phase7_meta_analysis.md:217,222,357,362`). This exposes how those papers' *methods*
   compare. It does not expose their *own* evaluation practice. Disclose it, but it is not a
   blindness breach for Axis 3.
5. **The Claude Doc "Coverage spot-check" sheet** (`phase7_coverage/screening.md:6-7`) is
   outside the repo. **What exactly it showed cannot be determined here.** Per
   `DEFENSIBILITY.md:208` it showed full abstracts and a Verdict per paper.
6. **Prior familiarity (not determinable).** the first author and Claude may know some of these papers from
   general reading or training (nn-Meter, uiCA, BHive, DietCode, WACO are well known). The repo
   cannot establish what either remembers about their statistics. The blindness statement
   should say "blind to full text; not necessarily naïve to the papers".
7. **No full text** of any of the 28 papers has been opened in this project, as far as the repo
   records. No §A entry or §C queue item covers them. The only one in §C is Intelligent4DSE, the
   11th Frame B paper, and it was seen "via search snippet only" (`phase7_meta_analysis.md:489-490`).

---

## TASK 3 — Options

**Calendar** (memory session note, 2026-09-27; outline):
- Full draft to the second author: Oct 16.
- Claim audit: Oct 17–18.
- Internal abstract target: Oct 21.
- **New evidence cutoff: Oct 23.**
- Submission: Oct 29.
- The first author's next writing task is §4. The 8e, R0 and R3/R4 evaluations also land this week and next.

**Hour estimates are assumptions.** The repo logs no reading times. The 15-paper Axis 3 human
check was scheduled for, and finished within, one day (Oct 5), alongside other work
(`DEFENSIBILITY.md:195-206`; session note). That suggests roughly **20–40 min per paper for an
Axis-3-only read** of an open PDF, plus time to get closed PDFs.

### (A) Disclose only (current plan)

- **What it answers:** "How was the sample chosen, and how much of a reproducible frame does it
  cover?" It does not answer "is recent or most-cited work more rigorous?"
- **Hours:** about 0.5 h of writing. §8 ¶2 already has the slot (`paper/sections/08_audit.tex:8`).
- **Fits before Oct 23:** yes.
- **Wording:**
  > "A purposive sample of 20 … A citation-ranked frame (Semantic Scholar, six fixed queries)
  > contains 8 of its 30 most-cited in-scope papers and none of its 10 most-cited 2024–26
  > in-scope papers; the uncovered papers were not audited."
- **No-cost strengthener (post-hoc; must be labelled post-hoc):** the year split *within* the
  20, from the §A footer lines `phase7_meta_analysis.md:141`–`445`.
  - None of the **9** papers from 2024–26 runs a significance test (A2 is ∂, the other 8 are F).
  - None of the 11 earlier papers does either.
  - **7 of 9** recent papers fail ≥3 axes, versus **10 of 11** earlier ones. The direction is
    "recent slightly better" on compound failure. Report it if the split is used at all.
- **Objection that remains:**
  > "Your 9 recent papers are hand-picked. The frame's most-cited recent papers are a different
  > set, mostly TVM-style tuning papers in systems/EDA venues. You audited none of them, so
  > 'no paper runs a test' may be a property of your selection."

  A careful reviewer will find this from the paper's own disclosure. A cannot rebut it, only
  concede it.
- **Guardrails:** none touched.

### (B) Axis-3-only supplementary check on the 10 Frame B papers

- **What it answers:** "Do the 10 most-cited recent in-scope papers in a reproducible frame,
  chosen by a rule we did not control, run a significance test for their headline claim?"
  This is the exact claim the paper leads with ("no paper runs a significance test"), checked on
  a set picked by someone else's ranking.
- **Hours:**
  - About 5 × 30 min for the open-access papers.
  - About 5 × 30 min for the closed ones, plus library or interlibrary-loan time (unknown).
  - About 1 h for the scoring sheet, appendix table and sentence.
  - **Total about 5–7 h of the first author's time**, plus acquisition.
- **Fits before Oct 23:** yes, if:
  1. the declaration is frozen by Oct 17;
  2. PDFs are *downloaded, not opened* by Oct 18;
  3. scoring happens Oct 19–22 in the declared order.

  Risk: closed papers that cannot be obtained in time. The declaration handles this by
  reporting them as "not obtained", never substituting.
- **Possible outcomes and §8 wording** (n = papers scored, excluding N/A and not-obtained):

  | Outcome | §8 sentence (one sentence + appendix table) | Effect on the headline |
  |---|---|---|
  | 0 of n run a test (n ≥ 7) | "In a separate, pre-declared check of the 10 most-cited in-scope 2024–26 papers in that frame, none of the n we could obtain runs a significance test for its headline comparison (Appendix X)." | Strongest. Answers the residual objection in A directly. |
  | k ≥ 1 of n run a test | "In a separate, pre-declared check …, k of n run a significance test for their headline comparison (Appendix X)." Name the passing papers as counter-examples in P5, as A9, A20 and A18 already are (`phase7_meta_analysis.md:586-602`). | The 20-paper "no paper runs a test" stays true *for the sample* but must not be generalised. §1 and the abstract wording must stay "in a purposive sample of 20". If k ≥ 3, the meta-claim weakens and the §8 lead should say so. |
  | Many N/A or not obtained (n ≤ 5) | Report the counts as they are. The supplement is weak; say it is inconclusive. | Falls back to A. |

  **Clustering caveat for every outcome:** B8, B9 and B10 share a group, so their results are
  likely correlated. Say so in the appendix.
- **Guardrails:**
  - Failures appear as counts in prose (guardrail 5, `phase7_meta_analysis.md:24-34`;
    `paper/sections/08_audit.tex:4`).
  - Per-paper scores go in the appendix table, the same practice as §A's full release
    (`phase7_meta_analysis.md:42-43`).
  - Passes may be named (guardrail 6, `:35-38`).
  - **Never merged:** no "29/30" or "30 papers" anywhere. The 20-paper table and percentages
    are unchanged.

### (C) B plus the uncovered Frame A papers, or a declared random subset

- **What it answers:** adds the "canonical" question, i.e. does the *most-cited* work in the area
  test? This bears less on the stated concern, which is about *recent* work: only 4 of the 22
  are from 2024, and all 4 are already in B.
- **Hours:**
  - 18 additional papers (17 without the A9 precursor) at about 30–40 min each is about
    9–12 h, plus acquisition. Several are closed: nn-Meter, CoDL, DNNPerf, IronMan-Pro;
    DietCode has no DOI.
  - **Total about 15–20 h with B.** A declared seeded random subset of 6 adds about 3–4 h
    (total about 9–11 h).
- **Fits before Oct 23:** the full version does not, given the Oct 16 draft and the
  Oct 17–18 claim audit. The subset version only just fits, at the cost of writing time.
- **Outcomes:** the same three cases as B, with a separate sentence for Frame A. Note that
  "most-cited" is skewed toward 2018–2022, so a 0/k result would mostly repeat what the
  sample's older half already shows.
- **Guardrails:** same as B. More scope risk, because 7 of the 22 are ambiguous or framework
  papers.

### (D) Full 5-axis audit of additional papers

- **What it answers:** the full rigor profile, so the papers could in principle join the corpus.
- **Hours:** roughly 2–3 h per paper for a human 5-axis read (no logged figure; the Aug 25
  5-paper verification is the closest precedent, `phase7_meta_analysis.md:522-548`).
  10 papers is about 20–30 h. **Does not fit.**
- **Risks:**
  - Axes 1, 2, 4 and 5 for 15 of the existing 20 are still AI-prescreened
    (`phase7_meta_analysis.md:556-557`). Human-scored additions would be held to a different
    standard than the corpus they join.
  - It invites merging, which would re-open the frozen n = 20 (`phase7_meta_analysis.md:97-98`).
- **Not recommended for this submission.** Listed as camera-ready or journal-extension work.

---

## TASK 4 — Recommendation: **B**, with A's disclosure and the post-hoc year split as the floor

**Reasons:**
1. **It targets exactly the remaining objection.** The paper's headline claim is about
   significance testing, and the uncovered set is the one a reviewer would name. B checks that
   claim, and only that claim, on a set we did not choose.
2. **It is affordable.** About 5–7 h fits the Oct 19–22 window without touching the Oct 16
   draft. C and D do not fit.
3. **It cannot damage the registered result.** It is reported separately and never merged.
   Every outcome has a pre-written sentence, so there is no post-hoc framing decision to make.
   This fits the project's "never soften a pre-registered result" rule.
4. **It also helps when k ≥ 1.** Finding testing papers would give a better calibrated §8 and
   more named counter-examples. The audit already says it wants those (guardrail 6).

**Conditions:**
- **Freeze before reading.** the first author approves and commits the declaration **before any of the 10
  PDFs is opened**. Downloading is fine.
- **Fixed scoring order:** B1 → B10. If time runs out, the unscored papers are the
  lowest-ranked, not a chosen subset.
- **Partial or fallback outcome.** If fewer than 6 papers are scorable by Oct 22, §8 uses A's
  wording and the appendix reports the partial table as inconclusive.

**Things this assessment noticed but did not change** (each needs the first author's call):
- **Older "recently published" wording.** The draft sentence at `phase7_meta_analysis.md:581`
  says "Across 20 *recently published* … papers". With 11 of 20 from before 2024 and Frame B at
  0/10, that wording should not reach the manuscript. The `.tex` files already follow the
  "purposive sample" rule (`paper/sections/08_audit.tex:4`, `paper/sections/01_introduction.tex:10`).
- **Log entry.** Per the defensibility-gates rule, the O15 entry should get a log line once the first author
  picks an option (`DEFENSIBILITY.md:65`). Not done here, per this task's no-edit rule.
- **The A9 precursor.** #269 is the precursor of A9 but is counted as uncovered under the strict
  title rule. The registered figure stands; the appendix can footnote it.

Draft declaration: `experiments/axis3_recent_coverage_declaration_DRAFT.md`. Not frozen and
not committed.
