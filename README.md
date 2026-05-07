# PHAI — Personalised AI Health Agent

A multi-agent personal health AI that fuses **gene mutation data**, **wearable sensor data**, and **natural-language sensor narratives** to deliver personalised, evidence-based health recommendations.

Inspired by Google Research's *"The Anatomy of a Personal Health Agent"* (arXiv 2508.20148).

## Status

Active development. See `PHAI_2Day_Sprint_Plan.md` for the build plan.

## Architecture (preview)

Three specialist sub-agents coordinated by an Orchestrator:
- **Data Science Agent** — analyses wearable time-series and population baselines
- **Domain Expert Agent** — grounds findings in a curated knowledge base of SNPs and lifestyle evidence
- **Health Coach Agent** — synthesises a personalised plan using motivational-interviewing style

## Data

- **Wearable backbone:** LifeSnaps (Yfantidou et al., 2022) — 71 real Fitbit users
- **Genetic layer:** OpenSNP — real anonymised public genotypes
- **Cohort extension:** stratified resampling to 1000 users for population baselines and ML training (clearly flagged as synthetic)

## Quick start

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Configure your LLM key
copy .env.example .env
# Edit .env and paste your Groq key

# 3. Smoke-test the LLM
python smoke_test.py

# 4. Run the app (after Day 2)
streamlit run ui/app.py
```

## License & disclaimer

For interview / demo purposes. **Not a medical device. Not a substitute for professional medical advice.**
