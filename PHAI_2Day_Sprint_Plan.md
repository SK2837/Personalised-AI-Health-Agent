# PHAI — 2-Day Interview Sprint Plan (Path B: real data, ~1000 users)

> Goal: a **complete, runnable Streamlit demo** of a Personal Health Agent that fuses gene + wearable + natural-language data, in **2 days**, using **free LLMs only** and **real public datasets** anchored at scale.
> Optimised for: looks finished, demos cleanly, story is grounded in real data, code is readable. Not for: production deployment.

---

## 1. The 30-second story (what you'll demo)

> "I built PHAI — a multi-agent personal health AI inspired by Google's PHA paper. It combines **real Fitbit wearable data from the LifeSnaps study (71 users)**, **real anonymised genotypes from OpenSNP (~71 users)** paired into a fused cohort, and a **natural-language sensor narrative layer**. I extended the cohort to **1000 users via stratified resampling** to support population baselines and a trained energy classifier. New users can also onboard on the spot — answer a short questionnaire and PHAI synthesises a plausible profile they can immediately query. Questions are routed through three specialised agents — Data Scientist, Domain Expert, Health Coach — coordinated by an Orchestrator."

You'll show:
1. The user's **gene + wearable dashboard** with population baselines (percentile bands across the 1000-user cohort).
2. **"Try it as yourself"** — interviewer fills a short form, gets a synthetic profile, and chats live as that user.
3. A **live chat** where you ask 3 prepared questions and get personalised, sourced plans.
4. The **reasoning trace** ("which agent did what, with which evidence"), so the architecture is visible.

---

## 2. Data architecture (Path B specifics)

| Layer | Source | Real users | After extension |
|---|---|---|---|
| Wearable backbone | **LifeSnaps** (Yfantidou et al., 2022, Zenodo) | 71 | 1000 (resampled with noise) |
| Gene layer | **OpenSNP** bulk download | ~71 sampled | 1000 (paired + extended via genotype frequencies) |
| Pairing | Synthetic 1:1 mapping, documented in README | — | — |
| NL layer | Generated from rule-based templates over daily summaries | — | — |
| Knowledge base | ~30 curated facts (SNPs + lifestyle recs) | — | — |

**Honesty in the README** — exact paragraph to ship:

> *"Wearable backbone: 71 real users from LifeSnaps (Yfantidou et al., 2022). Genetic layer: 71 real anonymised genotypes sampled from OpenSNP, paired with wearable users via a documented synthetic mapping. Demo cohort extended to 1000 via stratified resampling with Gaussian noise on per-user statistics, to support population-level baselines and ML training. All synthetic extensions clearly flagged in the data-loading code."*

---

## 3. Free-LLM stack (zero-cost, fast)

| Option | Speed | Quality | Setup | Use as |
|---|---|---|---|---|
| **Groq API** (Llama 3.1 70B) | ⚡⚡⚡ Fastest | Strong | Free key, 60s | **PRIMARY** |
| **Google Gemini 1.5 Flash** | ⚡⚡ Fast | Strong | Free key, 60s | **BACKUP** (1500 req/day free) |
| **Ollama** (Llama 3.1 8B local) | ⚡ Slower | Decent | 4 GB download | Offline fallback |

A 30-line `llm.py` provider abstraction lets you swap with one env var.

---

## 4. The 2-day timeline

### **DAY 1 — Real data, foundations, two agents (≈8 hrs)**

**Hour 1 — Bootstrap & kick off downloads (1 h)**
- Repo scaffold (`phai/` with subfolders).
- `requirements.txt`: streamlit, pandas, polars, duckdb, plotly, chromadb, sentence-transformers, groq, google-generativeai, scikit-learn, xgboost, python-dotenv, pydantic.
- `.env` for `GROQ_API_KEY`.
- `llm.py` — provider-agnostic chat function.
- **In a parallel terminal:** start downloading LifeSnaps (~10 GB) and an OpenSNP genotype subset. These run in the background while you code.
- Smoke test: "Hello, I'm your health agent." ✅

**Hours 2–3 — Real data ETL (2 h)**
- `etl/load_lifesnaps.py`: parse the LifeSnaps CSVs, extract per-user daily summaries (steps, sleep duration, sleep score, resting HR, HRV, stress, mood). Subset to the 6 metrics that matter; ignore the rest.
- `etl/load_opensnp.py`: bulk-download a sample of ~100 OpenSNP genotype files; for each, extract only our **10 curated rsids** (CLOCK, PER3, ADRB2, ACTN3, CYP1A2, MTHFR, FTO, APOE, COMT, BDNF). Drop the other 600K SNPs.
- `etl/pair_users.py`: deterministic 1:1 pairing of LifeSnaps users with OpenSNP genotype profiles (with a fixed seed so it's reproducible). Documented as synthetic in code comments.

**Hour 4 — Extend to 1000 + load DB (1 h)**
- `etl/extend_cohort.py`: stratified resampling — for each metric, fit a per-user distribution (mean + std), then sample N=929 new "users" by perturbing existing user statistics with Gaussian noise. Genotype layer extended by sampling at population allele frequencies for our 10 rsids.
- All extensions are flagged in a `synthetic` boolean column on the `users` table — the agent can disclose this.
- Load everything into **SQLite** (`phai.db`) using the schema below.

**Hour 5 — NL translation layer (1 h)**
- `nl_translator.py` — rule-based templates that turn each user-day into a one-line narrative.
  - Examples:
    - HRV in user's bottom decile → *"Recovery is below your usual range."*
    - Sleep < 6.5h two nights running → *"Two short nights in a row."*
    - Step count > 130% of personal baseline → *"You moved more than usual yesterday."*
- Generate and store a narrative for every user-day.

**Hour 6 — Mini knowledge base + RAG (1 h)**
- `kb/snippets.json` — ~30 curated text snippets:
  - 10 SNP cards (gene, variant, what it does, lifestyle implications, citation).
  - 20 evidence-tagged lifestyle recommendations (caffeine timing, light exposure, exercise type, sleep hygiene, nutrition, stress).
- `kb/ingest.py` — embed with `sentence-transformers/all-MiniLM-L6-v2` (free, local, 80 MB). Store in **ChromaDB** with citations.

**Hours 7–8 — Data Science + Domain Expert agents (2 h)**
- `agents/base.py` — shared LLM-call helper with a tool-use loop and JSON output validation (Pydantic).
- `agents/data_science.py` — tools: `get_recent_summary(user, days)`, `compare_to_baseline(user, metric)`, `compare_to_population(user, metric)` (uses the 1000-user cohort), `find_patterns(user, target)`.
- `agents/domain_expert.py` — tools: `lookup_snp(rsid)`, `kb_search(query, k)`, `gene_trait(gene)`.
- **Smoke test:** ask the DS agent *"Why have I been so tired this week?"* via CLI, get a structured analysis. ✅

**Day 1 end-state:** real data loaded for 1000 users, NL narratives generated, knowledge base searchable, 2 agents working end-to-end on the CLI.

---

### **DAY 2 — Coach + Orchestrator + ML model + UI + demo (≈8 hrs)**

**Hour 1 — Health Coach Agent + Orchestrator (1 h)**
- `agents/health_coach.py` — tools: `get_user_goals(user)`, `propose_plan(goal, constraints, evidence)`. Outputs structured JSON plan + friendly NL version. Motivational-interviewing tone (open questions, affirmations).
- `agents/orchestrator.py` — intent classifier (one LLM call) → routes to sub-agents → final synthesiser. Shared `scratchpad` dict that all agents read/write.
- **Smoke test:** end-to-end on CLI for all 3 demo queries. ✅

**Hour 2 — Train one credible ML model (1 h)**
- `models/energy_clf.py` — XGBoost classifier predicting next-day "energy level" (proxy: above/below median next-day step count, conditioned on prior 7 days of features).
- Train on 800 users, hold out 200 for eval. Report AUC (likely 0.72–0.80 range).
- Save with `joblib`. Make it a tool the DS agent can call.
- **Why this matters:** you get a real metric in the README ("Energy classifier: AUC 0.76 on N=200 holdout users"). Interviewers love that.

**Hours 3–6 — Streamlit app (4 h)**
- **Sidebar:** user picker (dropdown of 1000 users, with a quick "load demo user" button highlighting 3 curated profiles) + an **"Onboard a new user"** button that opens the form. Profile card.
- **Tab 1 — Dashboard:** Plotly charts of last 30 days (steps, sleep, HRV, stress, mood) with **percentile bands from the 1000-user cohort**. Latest NL narrative at the top.
- **Tab 2 — Genetic profile:** table of variants with plain-English explanation per SNP, expandable "what does this mean for me?" panels.
- **Tab 3 — Chat (the headline):** chat input → orchestrator → streamed response. Below the answer, expandable "How I built this answer" panel showing which agents fired, what tools they called, what evidence they cited.
- **Tab 4 — Plan history:** list of past plans with status.

**Hour 7 — "Try it as yourself" onboarding + demo prep (1 h)**
- `onboarding/synth_user.py` — takes form inputs (age, sex, chronotype, caffeine sensitivity, exercise level, typical sleep, stress level, ~5–8 fields) and:
  - Generates a 30-day wearable trajectory using the same per-user generator as the cohort-extension step (Day 1 hour 4), seeded by the questionnaire.
  - Samples a plausible genotype: weighted toward variants consistent with self-report (e.g., "late chronotype" → bias toward late-CLOCK / late-PER3 alleles).
  - Inserts as a new `users` row with `source='onboarded'`, returns `user_id`.
- **Streamlit modal** for the form: `st.form` with sliders + dropdowns, "Create my profile" button → spinner → drops user into Tab 1 already loaded with their data.
- 3 curated demo queries to keep ready as fallbacks:
  1. *"Why have I been so tired this week?"* (DS-led, uses energy classifier)
  2. *"My CYP1A2 result says I'm a slow caffeine metaboliser — what should I do?"* (DE-led)
  3. *"Build me a 7-day plan to feel more energetic."* (HC-led, uses both)

**Hour 8 — README + demo video (1 h)**
- **README.md** with: problem statement, architecture diagram, the **honest data paragraph** from §2, how-to-run, demo screenshots, design choices, energy-classifier metrics, what's next.
- 3–5 minute screen recording (Loom / OBS / built-in OS recorder).
- Script: 30s pitch → architecture diagram → dashboard with population bands → **onboard a new user live** → 3 demo queries on that fresh user → reasoning trace → wrap with "what I'd build next."

**Day 2 end-state:** Streamlit demo runs end-to-end on real-anchored data for 1000 users with a trained model, README + video ready, repo zip-able.

---

## 5. What's IN vs OUT (ruthless scope control)

**IN (must have)**
- 71 real LifeSnaps users + 71 real OpenSNP genotypes paired
- Extended to 1000 users via documented stratified resampling
- **"Try it as yourself" onboarding** — short questionnaire → synthetic profile → immediately queryable
- 3 sub-agents + orchestrator with visible reasoning trace
- NL sensor narratives (rule-based)
- Mini RAG knowledge base with citations
- One trained ML model (energy classifier) with reported AUC
- Streamlit UI with 4 tabs and population baselines
- Memory across chat turns
- README + demo video

**OUT (call out as "next steps" in README)**
- Live device ingestion (Fitbit/Apple Health APIs)
- Multimodal inputs (voice, images) — architecture is ready, MVP focuses on structured + text
- Multiple trained models (just the energy one for now)
- Real auth / multi-tenancy
- Production safety hardening
- Fine-tuning, multi-task TS models
- Comprehensive evaluation suite (just the one model metric for now)

**De-risk reminders**
- LLM rate limits on free tier — keep responses short, cache where you can.
- Cap the LLM context per agent — don't dump 30 days of raw rows; pass aggregated summaries.
- Test each agent **alone** before wiring the orchestrator.
- LifeSnaps download is large (~10 GB) — kick off **before you start coding**.
- Subset OpenSNP aggressively — only ingest the 10 curated rsids per file.

---

## 6. Database schema (SQLite, MVP)

```sql
users(user_id PK, source ENUM('lifesnaps','synthetic','onboarded'),
      synthetic BOOLEAN, age INT, sex TEXT, created_at)

user_variants(user_id, rsid, genotype, source ENUM('opensnp','synthetic'))

snp_reference(rsid PK, gene, trait_summary, lifestyle_implications,
              clinvar_significance, citation_url)

daily_summary(user_id, date, steps, sleep_min, sleep_score,
              resting_hr, hrv_avg, stress_score, mood_self_report)

nl_narratives(user_id, date, text, generator_version)

agent_memory(user_id, key, value_json, updated_at)

plans(plan_id PK, user_id, query, plan_json, created_at, status)
```

---

## 7. Repo layout

```
phai/
├── README.md
├── requirements.txt
├── .env.example
├── phai.db                # generated by ETL
├── llm.py                 # Groq/Gemini/Ollama abstraction
├── etl/
│   ├── load_lifesnaps.py
│   ├── load_opensnp.py
│   ├── pair_users.py
│   ├── extend_cohort.py   # stratified resample to 1000
│   └── nightly_aggregate.py
├── nl_translator.py
├── onboarding/
│   └── synth_user.py      # questionnaire -> synthetic profile + genotype
├── kb/
│   ├── snippets.json
│   └── ingest.py
├── models/
│   ├── energy_clf.py      # XGBoost, train + predict
│   └── registry.py
├── agents/
│   ├── base.py
│   ├── data_science.py
│   ├── domain_expert.py
│   ├── health_coach.py
│   ├── orchestrator.py
│   ├── tools.py
│   ├── prompts.py
│   └── schemas.py
├── ui/
│   └── app.py             # Streamlit
└── demo/
    ├── screenshots/
    └── walkthrough.mp4
```

---

## 8. Talking points for the interview

When you walk them through it, hit these:

1. **"Architecture follows Google's recent PHA paper"** — credibility, signals you read research.
2. **"Real data anchor — LifeSnaps + OpenSNP — extended honestly to 1000 users for population baselines and ML training."** — shows you understand what's actually open-source and how to handle the gap with integrity.
3. **"Three-layer data fusion: gene + wearable + natural-language narrative."** — the differentiator.
4. **"Three agents separated by responsibility, with a visible reasoning trace in the UI."** — modularity and explainability.
5. **"Free LLMs (Groq), local-first storage."** — privacy + cost story.
6. **"Energy classifier hit AUC X on a 200-user holdout."** — concrete, defensible metric.
7. **"New users can onboard on the spot — short questionnaire, plausible profile generated, immediately queryable."** — the "try it yourself" demo moment.
8. **"For v2: live wearable APIs, trained per-user models, multimodal inputs, clinician-in-the-loop eval."** — shows you see the gap.

---

## 9. Right now — kick-off checklist

- [ ] Get a free Groq API key (`https://console.groq.com`, 60s, no credit card).
- [ ] Confirm Python 3.11+ + a venv ready.
- [ ] **Start the LifeSnaps download in a browser tab now** (Zenodo: search "LifeSnaps Fitbit"). It'll be downloading while we scaffold.
- [ ] Say "go" — I'll start building. Plan: scaffold the repo, write `llm.py`, write the ETL skeletons, then start the data loaders so they're ready to run as soon as your download finishes.
