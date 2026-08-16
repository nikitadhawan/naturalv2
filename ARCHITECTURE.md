# NATURAL-v2 Architecture

NATURAL (Natural language-based Average Treatment effect estimation using Real-world data and LLMs) is a causal inference framework that estimates Average Treatment Effects (ATEs) from observational text data (Reddit, PubMed) and validates against randomized controlled trials from ClinicalTrials.gov.

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         create_study                                │
│  ClinicalTrials.gov → validate trials → train/val/test split        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Study (trials)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        filter_curate                                │
│                                                                     │
│  Reddit Archive ──┐                                                 │
│                   ├──► ConditionFilter → Contextualize → CurateStage│
│  PubMed ──────────┘                                                 │
│                                         └──► Parquet files          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ curated documents
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        estimate_ate                                 │
│                                                                     │
│  NATURAL Pipeline:                                                  │
│    RelevanceFilter → TreatmentOutcomeFilter → Knowns → Imputations  │
│                                    ↓                                │
│               ConditionalExtraction: P(T, Y | X)                   │
│                                    ↓                                │
│           Estimator (IPW / OI / Monte Carlo)                        │
│                                    ↓                                │
│                   ATE with bootstrap CIs                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Layout

```
naturalv2/
├── conf/                        # Hydra configuration
│   ├── common.yaml              # Shared settings (conditions, paths)
│   ├── estimate_ate.yaml        # ATE pipeline config
│   ├── filter_curate.yaml       # Curation pipeline config
│   ├── model/                   # LLM provider configs (openai, gemini, anthropic, vllm)
│   ├── source/                  # Data source configs (reddit, pubmed)
│   └── estimator/               # Estimator configs (ipw, oi, mc)
│
├── naturalv2/                   # Main package
│   ├── cli/                     # Entry points
│   │   ├── create_study.py      # Initializes trial study splits
│   │   ├── filter_curate.py     # Runs data curation pipeline
│   │   └── estimate_ate.py      # Runs ATE estimation pipeline
│   │
│   ├── pipeline/                # LLM extraction pipeline
│   │   ├── natural.py           # NATURALPipeline, PipelineContext, PipelineStage base
│   │   ├── sample_extraction.py # Relevance/treatment/outcome/covariate stages
│   │   └── conditional_extraction.py  # P(T,Y|X) extraction stage
│   │
│   ├── estimators/              # Causal estimators
│   │   ├── natural_ipw.py       # Inverse Probability Weighting
│   │   ├── natural_oi.py        # Outcome Imputation
│   │   └── natural_mc.py        # Monte Carlo estimator
│   │
│   ├── models/                  # LLM wrappers
│   │   ├── lm.py                # Model, APIModel, LocalModel
│   │   └── rate_limiter/        # Token/request rate limiting
│   │
│   ├── sources/                 # Data source handlers
│   │   ├── core.py              # CurationContext, SourceStage, FilterCurateRunner
│   │   ├── reddit/              # Reddit download, filter, contextualize, curate
│   │   ├── pubmed/              # PubMed search, contextualize, curate
│   │   └── anonymizer.py        # PII removal via Presidio
│   │
│   ├── prompts/                 # LLM prompt templates
│   ├── clinical_trial.py        # ClinicalTrial data structures
│   ├── experiment.py            # Experiment (single trial run)
│   ├── study.py                 # Study (collection of trials)
│   └── utils.py                 # Shared utilities
│
├── scripts/                     # Analysis and utility scripts
└── tests/                       # Unit tests
```

## Core Abstractions

### Study & Experiment
- **`ClinicalTrial`** — metadata for one NCT trial (arms, endpoints, eligibility)
- **`Study`** — collection of trials split into train/val/test with temporal ordering
- **`Experiment`** — represents one trial's extraction run (documents → effect estimate)

### Pipeline
`NATURALPipeline` orchestrates sequential `PipelineStage` objects against a set of curated documents. Stages filter down to high-quality (T, Y, X) observations, then a final `ConditionalExtractionStage` converts those observations into probability estimates fed to the estimators. See [Pipeline Stages](#pipeline-stages) for full details.

### Estimators
`NaturalIPW`, `NaturalOI`, and `NaturalMC` each consume the pipeline output to produce an ATE estimate with bootstrap confidence intervals, validated against RCT ground truth. See [Estimators](#estimators-1) for full details.

### Models
`APIModel` wraps LiteLLM to support OpenAI, Anthropic, Gemini, and hosted/local vLLM backends uniformly. A multi-bucket `RateLimiter` enforces per-provider token and request limits.

### Data Sources
Each source (Reddit, PubMed) implements a `FilterCurateRunner` composed of `SourceStage` steps:
1. Download / search
2. Condition filter
3. Contextualize (enrich with trial metadata)
4. Curate (trial-specific filtering, anonymize PII)
5. Write Parquet output

## Configuration

All three CLI commands are configured via [Hydra](https://hydra.cc/). Base configs live in `conf/` and are composed with model/source/estimator overrides at runtime:

```bash
estimate_ate source=reddit_20k model=openai estimator=natural_ipw
```

Custom `coalesce` resolver enables fallback value selection across config layers.

## Data Sourcing

### Overview

The Reddit pipeline converts raw, multi-terabyte historical Reddit data into a per-trial Parquet dataset of relevant posts and comments. It runs as part of `filter_curate` and is composed of four sequential `SourceStage` steps.

```
  RedditConditionFilter           ← identify relevant subreddits via LLM
           │
           ▼  condition → [subreddit, ...] mapping
           │
  RedditDownloadAndClean          ← download, decompress, parse, clean
           │
           ▼  hive-partitioned Parquet (by content_type / bucket)
           │
  SynonymStage                    ← expand treatment term variants via LLM
           │
           ▼  treatment → [canonical name, brand name, abbreviation, ...]
           │
  RedditCurateStage               ← match treatment terms, filter by trial
           │
           ▼  per-trial Parquet (one file per NCT ID)
           │
  Anonymizer                      ← strip PII via Presidio
           │
           ▼  final curated dataset
```

---

### Stage 1: `RedditConditionFilter`

The Pushshift archive contains thousands of subreddits; most are irrelevant to any given medical condition. This stage runs first to identify which subreddits are worth downloading at all.

1. **Searches the Reddit API** for candidate subreddits using condition and treatment keywords, fetching a small sample of posts per candidate.
2. **Prompts an LLM** (concurrently) with each subreddit's description and post snippets, asking it to judge relevance to the condition.
3. **Builds a `condition → [subreddit]` mapping** that is saved to the study dataset and passed to the download and curation stages.

This stage runs once per condition and its output is cached — subsequent runs skip the API calls.

---

### Stage 2: `RedditDownloadAndClean`

**Source:** [The Eye Pushshift archive](https://the-eye.eu/redarcs/) — a third-party mirror of Reddit's historical bulk exports. The archive hosts one `.zst` (Zstandard-compressed NDJSON) file per subreddit per content type (submissions and comments).

1. **Downloads `.zst` archives** for each subreddit identified in Stage 1, using `wget` with exponential-backoff retry and an insecure-TLS fallback for the archive's flaky certificate.
2. **Stream-decompresses** each archive in ~256 MB chunks. A carry buffer handles NDJSON records that straddle chunk boundaries, keeping memory use bounded regardless of file size.
3. **Parses to Apache Arrow** with a fixed schema (id, author, subreddit, score, selftext/body, created_utc, permalink). A regex pass de-quotes `created_utc` and `score` fields that some Pushshift archive vintages stored as strings instead of numbers.
4. **Filters low-quality rows** — drops records with no usable text, removed/deleted content, and bot posts via rule-based filters.
5. **Labels content type** — rows are tagged `submissions` or `comments` based on which text field (`selftext` vs `body`) survives filtering.
6. **Assigns a bucket** — subreddits are hashed into numbered buckets so downstream Parquet writes stay partition-contiguous.
7. **Writes hive-partitioned Parquet** under `content_type=.../bucket=...` directories for efficient predicate-pushdown reads.
8. **Contextualizes** — joins submissions with their comment threads to produce a `report_text` field containing the full conversation, plus metadata (title, date, permalink, author replies).

Processed archives are marked with a `.done` marker file so re-runs skip already-completed subreddits.

---

### Stage 3: `SynonymStage`

Before curation, treatment names are expanded to cover the full range of terms users might write. A drug like "fluoxetine" may appear as "Prozac", "fluox", or various misspellings. An LLM generates canonical variations (brand names, abbreviations, common misspellings) for each treatment. These expanded term lists are passed to the curation stage's string matcher.

---

### Stage 4: `RedditCurateStage`

Given the condition→subreddit mapping and the partitioned Parquet from Stage 2, this stage produces one Parquet file per clinical trial containing only posts that mention that trial's treatments.

1. **Builds an Aho-Corasick automaton per subreddit** — all treatment term variations across every trial relevant to a subreddit are compiled into a single multi-pattern string matcher. This allows scanning millions of posts in one pass without running a regex per term per post.
2. **Applies date filtering** — posts created after a trial's results publication date are excluded. This prevents temporal leakage: a Reddit user writing about a treatment *after* the RCT results were published may have been influenced by those results, which would bias the causal estimate.
3. **Scans Parquet partitions in parallel** — worker processes each take a batch of Parquet files, run the automaton, and write matching rows to a per-NCT-ID temp directory.
4. **Consolidates** — temp chunks for each NCT ID are merged into a single Parquet file.

Parallelism is tuned to available memory and CPU count (8 GB and a fixed thread count per worker by default).

---

### Stage 5: Anonymization

Before the curated files are written to their final location, all documents are passed through `presidio-analyzer` + `presidio-anonymizer` to detect and redact PII (names, emails, phone numbers, etc.). This is a safety measure given that Reddit posts can contain identifying personal health information.

---

### Storage Layout

```
{data_path}/
├── reddit_raw/
│   └── {subreddit}/
│       ├── content_type=submissions/
│       │   └── bucket=001/ *.parquet
│       └── content_type=comments/
│           └── bucket=001/ *.parquet
└── reddit_curated/
    └── {condition}/
        └── {nct_id}.parquet
```

Each final `{nct_id}.parquet` file contains posts/comments from relevant subreddits that mention that trial's treatments, with columns: `subreddit`, `title`, `initial_post`, `report_text`, `report_type`, `score`, `date_created`, `permalink`, `author_replies`.

---

## Pipeline Stages

All stages run async with configurable concurrency and operate on a `pd.DataFrame` of documents. Each stage filters or enriches rows; documents that fail a filter are dropped before the next stage.

### 1. `RelevanceFilterStage`

**Input:** all curated documents for a trial  
**Output:** documents relevant to the condition, treatment, and outcome of interest

The LLM is asked whether each document is relevant to the trial's condition and treatments. Documents that are off-topic (e.g., a Reddit post about an unrelated drug) are dropped here before any expensive extraction is done.

### 2. `TreatmentOutcomeFilterStage`

**Input:** relevant documents  
**Output:** documents where both treatment taken and outcome are identifiable

The LLM identifies which treatment arm the author took and whether the outcome is mentioned. Documents where the treatment is ambiguous or the outcome is not discussed are dropped. This is the primary quality gate — it ensures every remaining document can contribute a (T, Y) observation.

### 3. `KnownsStage`

**Input:** filtered documents  
**Output:** same documents with covariate columns populated (may contain `Unknown`)

The LLM extracts observed covariate values (e.g., age, severity, comorbidities) defined by the trial's eligibility criteria. It is permitted to return `Unknown` for any covariate it cannot determine from the text. These are the confounders X used by the estimators.

### 4. `ImputationsStage`

**Input:** documents with partially-known covariates  
**Output:** documents with all covariates filled in

For covariates marked `Unknown` by `KnownsStage`, the LLM makes a best-guess imputation based on contextual cues in the document. After this stage every row has a complete covariate vector X, which is required by IPW and OI.

### 5. `SampleTYStage` *(MC path only)*

**Input:** fully-imputed documents  
**Output:** documents with discrete sampled T and Y values

Instead of extracting log-probabilities, the LLM samples a treatment label and outcome label for each document given its covariates. The resulting hard assignments are treated as a synthetic observational dataset fed into the MC estimator.

### 6. `ConditionalExtractionStage`

**Input:** fully-imputed documents  
**Output:** documents with probability columns (`ty_given_x_probs` or `y_given_tx_probs`)

Uses a **log-probability-capable** LLM (typically a local vLLM model) to score answer tokens and derive:
- **P(T, Y | X)** — joint probability of treatment and outcome given covariates; used by `NaturalIPW` (after marginalizing out Y to get propensity scores) and `NaturalOI` (as outcome predictions)
- **P(Y | T, X)** — outcome probability conditioned on treatment and covariates; an alternative formulation for OI

Log-probs are optionally length-normalized before being stored. This is the only stage that requires a model capable of returning token log-probabilities rather than just text completions.

---

## Estimators

All three estimators take the pipeline output and produce an array of shape `(num_treatments, num_samples)` — the estimated response per treatment per document — which is then averaged (and bootstrapped for CIs) to produce the final ATE.

The RCT trials downloaded by `create_study` serve as **ground truth labels**: each trial has a known experimentally-determined ATE, and the framework's goal is to reproduce that estimate from observational text alone.

### `NaturalIPW` — Inverse Probability Weighting

**Uses:** `ty_given_x_probs` from `ConditionalExtractionStage`

Computes a propensity score P(T | X) by marginalizing Y out of P(T, Y | X). Each document's outcome contribution is then reweighted by `P(T,Y|X) / P(T|X)`. The intuition: upweight people who received a treatment they were unlikely to get (more informative counterfactual signal), downweight people who were always going to take it. Sensitive to poor propensity estimation when LLM probability calibration is off.

### `NaturalOI` — Outcome Imputation

**Uses:** `y_given_tx_probs` from `ConditionalExtractionStage`

For each covariate stratum X, averages the LLM's estimated E[Y | T, X] across all documents with matching covariates. No reweighting — it directly imputes what the outcome *would have been* under each treatment for each person. Sensitive to outcome model misspecification but not to propensity estimation errors.

### `NaturalMC` — Monte Carlo

**Uses:** sampled T/Y from `SampleTYStage`

Treats the LLM's hard-sampled (T, Y, X) tuples as a synthetic observational dataset and fits an off-the-shelf causal model from `causallib` (either IPW or OI variant). This decouples the causal estimation logic from the LLM's log-probability outputs, at the cost of information lost in discretizing soft probabilities into hard samples. Note: not suitable for APO estimation.

### Why three?

Each estimator makes different assumptions about where error originates. Running all three on the same trial lets the framework diagnose whether a poor ATE estimate is due to propensity errors (IPW degrades), outcome model errors (OI degrades), or sampling noise (MC degrades).

---

## Key Dependencies

| Concern | Library |
|---|---|
| LLM routing | `litellm` |
| Local inference | `vllm` |
| Config management | `hydra-core` |
| Causal inference | `causallib` |
| Data frames | `pandas`, `polars` |
| PII removal | `presidio` |
| Reddit data | `asyncpraw`, Pushshift `.zst` archives |
| Rate limiting | `aiolimiter`, `tenacity` |
| Experiment tracking | `weave` (optional) |
