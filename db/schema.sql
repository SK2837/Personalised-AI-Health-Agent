-- PHAI - SQLite schema
-- Source of truth for all tables. Run via db/init_db.py.

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- USERS
-- One row per person. `source` distinguishes real LifeSnaps users, the
-- synthetic-extension cohort, and users who self-onboarded via the
-- questionnaire flow.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    source      TEXT NOT NULL CHECK (source IN ('lifesnaps', 'synthetic', 'onboarded')),
    synthetic   INTEGER NOT NULL DEFAULT 0,         -- 0 = real, 1 = generated
    age         INTEGER,
    gender      TEXT,
    bmi         REAL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ----------------------------------------------------------------------------
-- DAILY_SUMMARY
-- One row per (user, date) with the wearable + EMA metrics we keep.
-- All numeric metrics are nullable - real data has gaps.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_summary (
    user_id                 TEXT NOT NULL,
    date                    TEXT NOT NULL,                    -- ISO YYYY-MM-DD

    -- Activity
    steps                   REAL,
    distance                REAL,                             -- metres
    calories                REAL,
    lightly_active_min      REAL,
    moderately_active_min   REAL,
    very_active_min         REAL,
    sedentary_min           REAL,

    -- Heart
    bpm_avg                 REAL,
    resting_hr              REAL,
    hrv_rmssd               REAL,                             -- HRV in ms
    nrem_hr                 REAL,                             -- HR during non-REM sleep
    breathing_rate          REAL,

    -- Sleep
    sleep_min               REAL,                             -- minutes asleep
    sleep_efficiency        REAL,                             -- 0..100
    minutes_to_fall_asleep  REAL,
    minutes_awake           REAL,

    -- Wellness scores (Fitbit-derived)
    stress_score            REAL,                             -- 0..100
    sleep_score_pct         REAL,                             -- 0..1
    exertion_score_pct      REAL,                             -- 0..1
    responsiveness_score_pct REAL,                            -- 0..1
    nightly_temp            REAL,

    -- Mood self-report (binary per day; NULL if user did not respond)
    mood_alert              INTEGER,
    mood_happy              INTEGER,
    mood_neutral            INTEGER,
    mood_rested             INTEGER,
    mood_sad                INTEGER,
    mood_tense              INTEGER,
    mood_tired              INTEGER,

    PRIMARY KEY (user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_daily_user_date ON daily_summary(user_id, date);

-- ----------------------------------------------------------------------------
-- USER_VARIANTS
-- Per-user genotypes for our curated rsid panel.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_variants (
    user_id   TEXT NOT NULL,
    rsid      TEXT NOT NULL,
    genotype  TEXT NOT NULL,                                  -- e.g. 'AA', 'AG', 'GG'
    source    TEXT NOT NULL CHECK (source IN ('opensnp', 'synthetic')),
    PRIMARY KEY (user_id, rsid),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_variants_user ON user_variants(user_id);
CREATE INDEX IF NOT EXISTS idx_variants_rsid ON user_variants(rsid);

-- ----------------------------------------------------------------------------
-- SNP_REFERENCE
-- Curated catalogue of our 10 lifestyle-relevant SNPs.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS snp_reference (
    rsid                     TEXT PRIMARY KEY,
    gene                     TEXT NOT NULL,
    trait_summary            TEXT NOT NULL,                   -- short plain-English summary
    lifestyle_implications   TEXT NOT NULL,                   -- what to do about it
    clinvar_significance     TEXT,                            -- e.g. 'benign', 'risk-factor'
    citation_url             TEXT
);

-- ----------------------------------------------------------------------------
-- NL_NARRATIVES
-- One natural-language sentence per user-day, generated from rule templates.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nl_narratives (
    user_id           TEXT NOT NULL,
    date              TEXT NOT NULL,
    text              TEXT NOT NULL,
    generator_version TEXT NOT NULL DEFAULT 'v1',
    PRIMARY KEY (user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- AGENT_MEMORY
-- Long-term key/value store for goals, preferences, prior insights.
-- value_json is a JSON blob to keep the schema flexible.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_memory (
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, key),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- PLANS
-- Health plans produced by the Coach agent. plan_json is a structured plan
-- (goal, steps, metrics_to_track, check_in_date).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS plans (
    plan_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    query       TEXT NOT NULL,
    plan_json   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_plans_user ON plans(user_id);
