# Personalised AI Health Agent (PHAI) — End-to-End Project Roadmap

> **Inspiration:** Google Research's *"The Anatomy of a Personal Health Agent"* (PHA, arXiv 2508.20148).
> **Differentiator:** Fuse **gene mutation data** + **wearable sensor data** + **natural-language translations of sensor readings** to produce personalised, evidence-based health recommendations.
> **Target form factor:** A working, demonstrable AI agent (CLI + Streamlit/Gradio UI) that can answer queries like *"I want to feel more energetic"* with a personalised plan.

---

## 1. Vision & Scope

### 1.1 What PHAI does (one paragraph)
PHAI ingests a user's genetic variant profile, their wearable time-series data (steps, sleep, HR, HRV, etc.), and short natural-language journal/sensor descriptions ("felt sluggish this morning"). It runs three cooperating sub-agents — a **Data Science Agent** (analyses wearables), a **Domain Expert Agent** (looks up medical/genetic knowledge), and a **Health Coach Agent** (produces a behaviour-change plan) — coordinated by an **Orchestrator**. When asked an open question, PHAI returns a tailored multi-step plan grounded in the user's data and trustworthy sources.

### 1.2 MVP vs. Full Vision

| Capability | MVP (4–6 weeks) | Full (3–6 months) |
|---|---|---|
| Genetic data | 20–50 well-studied SNPs (sleep, energy, caffeine, fitness) | Full VCF / 23andMe import + GWAS panel |
| Wearable data | 1 public dataset (LifeSnaps Fitbit) | Live Fitbit/Apple Health/Google Fit ingestion |
| NL sensor layer | Rule-based templates | LLM-generated narratives + user journal RAG |
| Models | 2–3 simple classifiers (energy, sleep quality, stress) | Multi-task time-series transformers + per-user fine-tuning |
| Agents | 3 sub-agents + simple orchestrator (Claude/Gemini API) | LangGraph-based stateful agents with memory + tools |
| UI | Streamlit | Web app (React + FastAPI) + mobile companion |
| Evaluation | 50–100 test queries, manual rubric | Automated + clinician-in-the-loop benchmark |

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                       │
│  (Streamlit / FastAPI + React) — chat, plan view, dashboards   │
└───────────────────────────┬────────────────────────────────────┘
                            │
                ┌───────────▼────────────┐
                │     ORCHESTRATOR        │  ← intent → routing → synthesis
                │  (LLM + state machine)  │
                └─┬────────┬───────────┬──┘
                  │        │           │
       ┌──────────▼──┐  ┌──▼──────┐  ┌─▼────────────┐
       │ DATA-SCIENCE│  │ DOMAIN  │  │ HEALTH COACH │
       │   AGENT     │  │ EXPERT  │  │   AGENT      │
       │ (Python +   │  │ AGENT   │  │ (planner +   │
       │  pandas +   │  │ (RAG +  │  │  motivational│
       │  ML models) │  │ KG)     │  │  interview)  │
       └──────┬──────┘  └────┬────┘  └──────┬───────┘
              │              │              │
       ┌──────▼──────────────▼──────────────▼──────┐
       │             SHARED CONTEXT LAYER           │
       │  user profile · memory · session state     │
       └──────┬──────────────┬──────────────┬──────┘
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐
        │ Wearable  │  │  Gene     │  │ Knowledge │
        │  TS DB    │  │ Variant   │  │   Base    │
        │ (DuckDB/  │  │  DB       │  │ (Vector + │
        │  SQLite)  │  │ (SQLite)  │  │  Graph)   │
        └───────────┘  └───────────┘  └───────────┘
```

---

## 3. Data Strategy

### 3.1 Gene Mutation Data — Open Sources

| Source | What it gives you | Format | Notes |
|---|---|---|---|
| **dbSNP** (NCBI) | Catalogue of all known SNPs | API / VCF | Foundation reference |
| **ClinVar** (NCBI) | SNP → clinical significance | TSV / VCF | Critical for "is this variant meaningful?" |
| **GWAS Catalog** (EBI) | SNP → trait associations w/ p-values | TSV | Backbone of trait predictions |
| **SNPedia** | Human-readable SNP annotations | MediaWiki API | Great for narrative explanations |
| **OpenSNP** | Real anonymised user genotypes | CSV/JSON | Sample data for end-to-end testing |
| **gnomAD** | Population allele frequencies | VCF | For variant rarity scoring |
| **OMIM** (free for research) | Gene → disease | XML | Classical reference |
| **1000 Genomes** | Reference genotypes | VCF | Population baselines |

**Practical MVP picks:** ClinVar + GWAS Catalog + SNPedia + a tiny OpenSNP sample (5–10 users).

**Curated SNP starter pack** (well-studied, lifestyle-relevant):
- *CLOCK* (rs1801260) — sleep timing
- *PER3* (rs57875989) — chronotype
- *ADRB2* (rs1042713) — exercise response
- *ACTN3* (rs1815739) — power vs. endurance
- *CYP1A2* (rs762551) — caffeine metabolism
- *MTHFR* (rs1801133) — folate / energy
- *FTO* (rs9939609) — appetite / BMI
- *APOE* (rs429358 / rs7412) — lipid metabolism
- *COMT* (rs4680) — stress response
- *BDNF* (rs6265) — learning / mood

### 3.2 Wearable Datasets — Open Sources

| Dataset | Size | Modalities | Best for |
|---|---|---|---|
| **LifeSnaps (Fitbit)** | 71 users, 4 months | steps, sleep, HR, mood, stress | MVP — best fit |
| **PMData** | 16 users, 5 months | Fitbit + sport + nutrition + reports | Multi-modal training |
| **WESAD** | 15 subjects | EDA, ECG, EMG, accel | Stress detection model |
| **PPG-DaLiA** | 15 subjects | PPG + accel + ECG | HR estimation |
| **MMASH** | 22 subjects | Actigraphy + sleep + saliva | Circadian / cortisol |
| **PhysioNet (multiple)** | varies | clinical-grade signals | Robust signal processing |
| **Stanford "MyHeartCounts"** | thousands | step + survey | Population-scale baselines |
| **Synthetic (Synthea + custom)** | unlimited | configurable | Rapid prototyping |

**Practical MVP pick:** LifeSnaps (best tradeoff of size, realism, multi-modality) + small synthetic supplement.

### 3.3 Natural-Language Sensor Layer

Two complementary approaches — build both:

**A. Rule-based templates (deterministic, MVP-ready)**
```
HRV: 28 ms (P10 for user) →
"Your overnight recovery looks low — heart-rate variability
 is in the bottom 10% of your last 30 days."
```

**B. LLM-generated narratives (richer, v2)**
Feed structured stats → Claude/Gemini → 1–2 sentence natural narrative. Cache to keep cheap.

**C. User journal channel**
Free-text user entries ("felt foggy after lunch") parsed into tagged events; stored alongside sensor data so the agent can correlate.

### 3.4 Phenotype / Trait Labels (the supervised target)
For the trait predictors you'll need labels. Options, in order of effort:
1. **Self-report from datasets** (LifeSnaps has mood/stress) — cheapest.
2. **Derived labels** (sleep score, recovery score) computed from raw signals using known formulas.
3. **Weak supervision** (e.g., "energetic = next-day step count > P75 AND HRV > median").
4. **Manual labelling** of a small high-quality holdout for evaluation.

---

## 4. Database Design

### 4.1 Schema (SQLite for MVP, Postgres later)

```sql
-- Users
users(user_id, name, dob, sex, created_at)

-- Genetic variants per user
user_variants(user_id, rsid, genotype, source, imported_at)

-- Reference SNP knowledge
snp_reference(rsid, gene, chromosome, traits_json,
              clinvar_significance, gwas_traits_json,
              snpedia_summary, last_updated)

-- Wearable time-series (one row per metric per timestamp; or wide tables per metric)
wearable_metrics(user_id, ts, metric, value, device, source)

-- Daily aggregates (precomputed, fast for the agent to read)
daily_summary(user_id, date, sleep_min, sleep_score, steps,
              resting_hr, hrv_avg, stress_score, mood_self_report)

-- Natural-language narratives
nl_narratives(user_id, date_or_ts, metric_or_topic, text, source_model)

-- User journal
user_journal(user_id, ts, text, tags_json)

-- Agent memory
agent_memory(user_id, key, value_json, updated_at)
-- e.g., goals, preferences, prior plans, adherence

-- Plan history
plans(plan_id, user_id, query, plan_json, created_at, status)
```

### 4.2 Knowledge Base (vector + relational)
- **Vector store** (Chroma / FAISS): chunked PubMed abstracts, SNPedia entries, NIH guidelines, exercise-physiology references.
- **Relational metadata**: `kb_docs(doc_id, source, url, title, license, ingested_at)` + `kb_chunks(chunk_id, doc_id, text, embedding)`.

---

## 5. Models to Train (and What For)

The orchestrator and agents are LLM-driven, but the **Data Science Agent** needs real predictive models.

| Model | Input | Output | Suggested approach (MVP) |
|---|---|---|---|
| **Energy / fatigue classifier** | last 7 days of wearable features | "low / med / high" probability | Gradient-boosted trees (XGBoost / LightGBM) |
| **Sleep-quality regressor** | overnight signals | 0–100 score | Already-known formulas + small XGB residual model |
| **Stress detector** | HRV + EDA + steps | stress level | XGB on WESAD features, transferred to LifeSnaps |
| **Gene-trait risk score** | user variants × GWAS effect sizes | per-trait polygenic score | Linear additive PRS (no training needed for MVP) |
| **Recovery / readiness** | sleep + HRV + RHR | 0–100 | Weighted blend → calibrated |
| **(v2) Multi-task TS model** | raw multi-channel signals | joint trait predictions | 1D-CNN or small Transformer in PyTorch |
| **Embeddings for RAG** | text chunks | dense vectors | `sentence-transformers/all-MiniLM-L6-v2` (free) or OpenAI/Voyage |

**Training discipline:**
- Per-user train/val/test split *by time* (no leakage from future).
- Population-level baselines + per-user fine-tuning where feasible.
- Calibrated probabilities (Platt / isotonic) — agents need honest confidences.

---

## 6. Agent Design

### 6.1 Orchestrator (the manager)
- Parses user intent (classification: data-question / knowledge-question / plan-request / mixed).
- Builds a **plan graph**: which sub-agents to call, in what order, what to pass.
- Manages **shared scratchpad** (JSON blob each agent reads/writes).
- Final synthesiser: takes all sub-agent outputs and produces one coherent answer.
- Implementation: LangGraph (recommended) or a hand-rolled state machine in ~300 lines of Python.

### 6.2 Data Science Agent
- **Tools:** `query_wearables(user_id, range, metrics)`, `run_model(name, features)`, `compute_stat(...)`, `python_sandbox`.
- **Loop:** intent → analysis plan → call tools → interpret → return structured findings + NL summary.

### 6.3 Domain Expert Agent
- **Tools:** `lookup_snp(rsid)`, `kb_search(query, k)`, `pubmed_search(query)`, `gene_trait_lookup(gene)`.
- **Job:** Take findings + user's variants, ground them in literature, flag clinical caveats, output evidence-tagged statements.

### 6.4 Health Coach Agent
- **Tools:** `get_user_goals()`, `get_recent_plans()`, `propose_plan(goal, constraints)`, `schedule_check_in(...)`.
- **Style:** Motivational interviewing — open questions, affirmations, reflective listening, summaries (the OARS framework).
- **Output:** A structured plan (`{goal, why, steps[], metrics_to_track, check_in_date}`) plus a friendly natural-language version.

### 6.5 Memory
- Short-term: session scratchpad.
- Long-term: `agent_memory` table — goals, preferences, what worked, what failed.
- Retrieval: load top-k relevant memories at each turn (small RAG over user's own history).

---

## 7. Tech Stack (recommended)

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | Ecosystem dominance |
| Data | **pandas, polars, DuckDB** | Fast local analytics |
| ML | **scikit-learn, XGBoost, LightGBM** (+ PyTorch v2) | Right tool for tabular + TS |
| LLM | **Anthropic Claude** (Sonnet/Haiku) — easiest; or **Gemini**; fallback **Llama 3 / Mistral via Ollama** | Quality vs. cost |
| Agents | **LangGraph** (or **CrewAI**) | Built-in state, tools, memory |
| Vector DB | **ChromaDB** (MVP) → **Qdrant / Weaviate** (prod) | Local-first, easy upgrade |
| Bio tooling | **PyVCF / cyvcf2, biopython, myvariant.py** | VCF parsing, NCBI lookups |
| UI | **Streamlit** (MVP) → **FastAPI + React** | Fast to ship, then real |
| Storage | **SQLite** → **Postgres + TimescaleDB** | Time-series scale |
| Eval | **DeepEval / Promptfoo / Ragas** | Standardised LLM/RAG eval |
| Orchestration ops | **Prefect or Airflow** | Daily ETL of wearables |
| Privacy | **Local-first by default**, optional cloud with explicit consent | Health data is sensitive |

---

## 8. Phased Roadmap (with concrete deliverables)

### Phase 0 — Foundations (Week 0)
- Repo structure (`/data /db /models /agents /ui /eval /notebooks /tests`).
- Python env (`uv` or `poetry`), pre-commit, CI skeleton.
- Decide LLM provider, get API keys.
- **Deliverable:** repo bootstrapped, "hello world" agent that calls the LLM.

### Phase 1 — Data Acquisition (Weeks 1–2)
- Download LifeSnaps + small OpenSNP sample.
- Pull ClinVar + GWAS Catalog + SNPedia subset for the curated SNP list.
- Write loaders that normalise everything into the schema in §4.
- **Deliverable:** populated SQLite DB; one notebook that loads and visualises one user end-to-end.

### Phase 2 — Preprocessing & Feature Engineering (Weeks 2–3)
- Wearable: resampling, gap-filling, daily aggregates, rolling windows, circadian features.
- Genetic: PRS computation per trait using GWAS effect sizes.
- NL layer v1: rule-based template generator → `nl_narratives` table.
- **Deliverable:** `daily_summary` populated; narratives auto-generated nightly.

### Phase 3 — Model Training (Weeks 3–4)
- Train energy, sleep-quality, stress models with time-based CV.
- Calibrate, save with `joblib`, register in a tiny model registry table.
- Build `run_model()` tool.
- **Deliverable:** all MVP models hit reasonable baselines on a held-out test user.

### Phase 4 — Knowledge Base & RAG (Week 4)
- Ingest: SNPedia entries for curated SNPs, a curated set of NIH/PubMed abstracts, sleep/exercise textbook excerpts (license-cleared).
- Chunk, embed, store in Chroma.
- Build `kb_search()` tool with citation passthrough.
- **Deliverable:** `kb_search("energy fatigue MTHFR")` returns ranked, cited chunks.

### Phase 5 — Sub-Agents (Week 5)
- Implement DS, DE, HC agents as LangGraph nodes with their tool sets.
- Each agent: system prompt, tool schemas, output schema (Pydantic).
- Unit tests with golden traces for each.
- **Deliverable:** each sub-agent answers its domain query in isolation.

### Phase 6 — Orchestrator & Memory (Week 6)
- Intent classifier (small LLM call or fine-tuned mini-model).
- Routing graph + shared scratchpad.
- Memory read/write hooks.
- Final synthesiser prompt with citation-merging.
- **Deliverable:** end-to-end demo answering "I want to feel more energetic" with a personalised, sourced plan.

### Phase 7 — UI (Week 7)
- Streamlit app: chat, user picker, day/week dashboards (steps/sleep/HRV charts), plan view, "explain this recommendation" drill-down.
- **Deliverable:** clickable demo you can hand to a non-technical user.

### Phase 8 — Evaluation (Weeks 7–8)
- Curate ~100 test queries across categories (data-only, knowledge-only, plan, mixed).
- Rubric: factual accuracy, personalisation, safety, actionability, evidence quality.
- Automated checks (Ragas for RAG faithfulness, custom checks for plan structure).
- Optional: small human eval (3–5 reviewers).
- **Deliverable:** eval dashboard + first quality numbers.

### Phase 9 — Safety, Privacy, Ethics (continuous, gate before any external use)
- Disclaimers everywhere ("not a medical device, not a doctor").
- Hard refusals for diagnostic/dose questions.
- Local-first storage; encryption at rest; no PHI in LLM logs.
- Bias review on small/diverse cohort.
- Audit trail of every plan + sources.
- **Deliverable:** SAFETY.md, threat model, redaction layer.

### Phase 10 — Hardening & Roadmap to v2
- Replace static dataset with live device ingestion (Fitbit / Apple Health export).
- Move to Postgres + Timescale.
- Replace rule NL with LLM narratives (cached).
- LangGraph state persistence + multi-day plan tracking.
- Per-user fine-tuning experiments.

---

## 9. Repo Structure (proposed)

```
phai/
├── README.md
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/            # downloaded datasets (gitignored)
│   ├── interim/
│   └── processed/
├── db/
│   ├── schema.sql
│   ├── migrations/
│   └── seed/
├── etl/
│   ├── load_lifesnaps.py
│   ├── load_opensnp.py
│   ├── load_clinvar.py
│   └── nightly_aggregate.py
├── models/
│   ├── energy_clf.py
│   ├── sleep_quality.py
│   ├── stress_detector.py
│   ├── prs.py
│   └── registry.py
├── kb/
│   ├── ingest.py
│   ├── chunk.py
│   └── embed.py
├── agents/
│   ├── orchestrator.py
│   ├── data_science_agent.py
│   ├── domain_expert_agent.py
│   ├── health_coach_agent.py
│   ├── tools/
│   ├── prompts/
│   └── schemas.py
├── ui/
│   └── app_streamlit.py
├── eval/
│   ├── test_queries.yaml
│   ├── rubric.py
│   └── run_eval.py
├── tests/
└── notebooks/
```

---

## 10. Example End-to-End Flow

User: *"I want to feel more energetic during the week."*

1. **Orchestrator** classifies → intent = `plan_request` (mixed: needs DS + DE + HC).
2. **Data Science Agent**
   - Pulls last 30 days of `daily_summary`.
   - Runs `energy_clf`; finds energy-low days correlate with sleep < 6.5 h and HRV in P25.
   - Returns: `{patterns: [...], confidence: 0.78}`.
3. **Domain Expert Agent**
   - Looks up user's variants → CYP1A2 slow metaboliser, CLOCK late chronotype, MTHFR C677T heterozygous.
   - `kb_search("late chronotype energy management")` and `kb_search("MTHFR fatigue evidence")`.
   - Returns evidence-tagged statements with citations.
4. **Health Coach Agent**
   - Reads memory: user prefers morning workouts, dislikes meal-prep apps.
   - Builds a plan: shift caffeine to before noon (CYP1A2), 20-min morning daylight (CLOCK), B-vitamin-rich breakfast (MTHFR), weekday sleep target 7.5 h, HRV-guided training.
   - Sets a 7-day check-in.
5. **Orchestrator** synthesises → final reply: friendly summary + structured plan + "why each step" + citations.
6. **Plan saved** to `plans` table; metrics to track flagged in `agent_memory`.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Medical/legal liability | Strong disclaimers; refuse diagnosis/dosing; cite sources; log everything |
| Genetic data privacy | Local-first; encryption; no raw genotype to LLM (use derived flags only) |
| LLM hallucination | RAG with citations; require source IDs in DE agent outputs; eval gates |
| Small / biased datasets | Mix datasets; report uncertainty; avoid demographic claims you can't back |
| Scope creep | Lock MVP scope; v2 backlog file |
| Cost (LLM tokens) | Cache narratives; use Haiku/Flash for routing; Sonnet/Pro only for synthesis |

---

## 12. Open Decisions for You

Before we write code, please confirm or choose:

1. **LLM provider** — Claude API (recommended), Gemini, or local (Llama/Mistral via Ollama)?
2. **Primary wearable dataset** — LifeSnaps (recommended) or another?
3. **MVP scope** — All three sub-agents from day one, or start with Data Science Agent + simple knowledge layer and grow?
4. **UI** — Streamlit (recommended for MVP) or jump straight to FastAPI + React?
5. **Compute** — Local laptop only, or do you have a GPU / cloud account for training?
6. **Privacy posture** — Local-only (recommended), or are you comfortable with cloud LLMs seeing derived (non-raw) data?
7. **Timeline** — Are you optimising for a working demo in ~6 weeks, or a deeper system over 3–6 months?

---

## 13. Immediate Next Steps (once you confirm above)

1. Bootstrap the repo (Phase 0) — I can scaffold this for you in ~10 minutes.
2. Wire up the LLM and write a "hello, I'm your health agent" smoke test.
3. Download and load LifeSnaps for one user; visualise.
4. Build the curated-SNP reference table.
5. Ship the rule-based NL narrative generator.
6. From there, train the first model (energy classifier) and we have a real foundation.

---

*Document v1 — generated as the project blueprint. We'll iterate this as decisions land.*
