# NATURAL-v2

This repository extends [NATURAL](https://arxiv.org/abs/2407.07018) to larger data and evaluation scales.

---

## Setup

It is recommended to use [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for dependency management and virtual environments.

```bash
git clone https://github.com/nikitadhawan/naturalv2.git
cd naturalv2
uv sync --no-cache --dev
```

> **Tip:** Add `--active` to `uv sync` to use your current virtual environment. Otherwise, `.venv` will be created in the project root.

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

---

## Retrospective Study

Create a retrospective study for a condition (e.g. "diabetes") with temporally split training and validation clinical trials:

```bash
uv run --active --env-file=.env create_study conditions=[<condition>] experiment_name=test
```

---

## Data Filtering and Curation

Filter and curate data for a source (e.g. Reddit):

```bash
uv run --active --env-file=.env filter_curate \
    sample_model.model_id="gemini/gemini-2.5-pro" \
    sample_model.rpm=5 \
    sample_model.tpm=250000 \
    sample_model.rpd=100 \
    +sample_model.thinking.type=enabled \
    +sample_model.thinking.budget_tokens=-1 \
    sources.reddit.stages.download_and_clean.max_download_workers=8 \
    experiment_name=test
```

> This will filter subreddits, download and clean relevant Reddit data, and curate experiment datasets.

---

## Estimating NATURAL ATEs

Convert curated Reddit data to NATURAL-IPW ATE:

```bash
uv run --active --env-file=.env estimate_ate \
    ~pipeline.stages.relevance_filter \
    cheap_model.model_id="gemini/gemini-2.5-flash" \
    cheap_model.rpm=10 \
    cheap_model.tpm=250000 \
    cheap_model.rpd=250 \
    cheap_model.max_parallel_requests=10 \
    +cheap_model.reasoning_effort=medium \
    sample_model.model_id="gemini/gemini-2.5-pro" \
    sample_model.rpm=5 \
    sample_model.tpm=250000 \
    sample_model.rpd=100 \
    sample_model.max_parallel_requests=10 \
    +sample_model.thinking.type=enabled \
    +sample_model.thinking.budget_tokens=-1 \
    imputations_model.model_id="gemini/gemini-2.5-pro" \
    imputations_model.rpm=10 \
    imputations_model.tpm=250000 \
    imputations_model.rpd=250 \
    imputations_model.max_parallel_requests=10 \
    +imputations_model.thinking.type=enabled \
    +imputations_model.thinking.budget_tokens=-1 \
    probs_model.model_id="meta-llama/Llama-3.3-70B-Instruct" \
    probs_model.model_kwargs.tensor_parallel_size=4 \
    +probs_model.model_kwargs.max_model_len=16384 \
    estimator=natural_ipw \
    split=train \
    experiment_name=test
```

> Model choices and parameters can be adjusted based on your budget and hardware.

---
